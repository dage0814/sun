from __future__ import annotations

import copy
import csv
from collections.abc import Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import nn

from .config import AblationConfig, TrainingConfig
from .losses import CombinedStockObjective, mixup_objective, temporal_mixup
from .pipeline import MultiModalASTGTN
from .text_encoder import ListMLELoss


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    validation_mcc: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_mcc: float = -1.0


class TwoStageTrainer:

    def __init__(
        self,
        model: MultiModalASTGTN,
        training: TrainingConfig,
        ablation: AblationConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.training = training
        self.ablation = ablation
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.listmle = ListMLELoss()
        self.objective = CombinedStockObjective(training, ablation)

    def fit_stage1(self, loader: Iterable[Mapping[str, torch.Tensor]]) -> List[float]:
        if not self.ablation.use_text or not self.ablation.use_listmle:
            return []
        if self.model.text_encoder is None:
            raise RuntimeError("Stage 1 requires a text encoder")
        if self.training.stage1_epochs and isinstance(loader, Sized) and len(loader) == 0:
            raise ValueError("Stage 1 loader contains no batches")
        optimizer = torch.optim.AdamW(
            self.model.text_encoder.parameters(),
            lr=self.training.stage1_lr,
            weight_decay=self.training.weight_decay,
        )
        history = []
        self.model.text_encoder.train()
        for _ in range(self.training.stage1_epochs):
            total, count = 0.0, 0
            for raw_batch in loader:
                batch = self._to_device(raw_batch)
                required = {"input_ids", "attention_mask", "industry_relevance"}
                missing = required.difference(batch)
                if missing:
                    raise KeyError(f"Stage 1 batch is missing {sorted(missing)}")
                optimizer.zero_grad(set_to_none=True)
                output = self.model.text_encoder(
                    batch["input_ids"], batch["attention_mask"], batch.get("news_mask")
                )
                loss = self.listmle(output.industry_scores, batch["industry_relevance"])
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.text_encoder.parameters(), self.training.grad_clip_norm
                )
                optimizer.step()
                total += float(loss.detach())
                count += 1
            if count == 0:
                raise ValueError("Stage 1 loader contains no batches")
            history.append(total / count)
        return history

    def fit_stage2(
        self,
        train_loader: Iterable[Mapping[str, torch.Tensor]],
        validation_loader: Optional[Iterable[Mapping[str, torch.Tensor]]] = None,
    ) -> TrainingHistory:
        if isinstance(train_loader, Sized) and len(train_loader) == 0:
            raise ValueError("Stage 2 training loader contains no batches")
        if isinstance(validation_loader, Sized) and len(validation_loader) == 0:
            raise ValueError("Validation loader contains no batches")
        optimizer = self._stage2_optimizer()
        history = TrainingHistory()
        best_state = None
        epochs_without_improvement = 0
        for epoch in range(self.training.stage2_epochs):
            self.model.train()
            total, count = 0.0, 0
            for raw_batch in train_loader:
                batch = self._to_device(raw_batch)
                self._pop_metadata(batch)
                targets = batch.pop("targets")
                batch.pop("industry_relevance", None)
                window_cutoffs = batch.pop("window_cutoffs", None)
                optimizer.zero_grad(set_to_none=True)
                encoded = self.model.encode_features(**batch)
                if self.ablation.use_mixup:
                    mixed, target_a, target_b, coefficient, permutation = temporal_mixup(
                        encoded.features, targets, self.training.mixup_alpha,
                        return_permutation=True,
                    )

                    mixed_cutoffs = None if window_cutoffs is None else torch.minimum(
                        window_cutoffs, window_cutoffs[permutation]
                    )
                    logits = self.model.backbone(mixed, mixed_cutoffs).logits
                    objective = mixup_objective(
                        self.objective, logits, target_a, target_b, coefficient
                    )
                else:
                    logits = self.model.backbone(encoded.features, window_cutoffs).logits
                    objective = self.objective(logits, targets)
                objective.total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training.grad_clip_norm)
                optimizer.step()
                total += float(objective.total.detach())
                count += 1
            if count == 0:
                raise ValueError("Stage 2 training loader contains no batches")
            history.train_loss.append(total / count)

            if validation_loader is None:
                continue
            mcc = self.evaluate_mcc(validation_loader)
            history.validation_mcc.append(mcc)
            if best_state is None or mcc > history.best_mcc:
                history.best_mcc = mcc
                history.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.training.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def evaluate_mcc(self, loader: Iterable[Mapping[str, torch.Tensor]]) -> float:

        self.model.eval()
        predictions, truths = [], []
        for raw_batch in loader:
            batch = self._to_device(raw_batch)
            self._pop_metadata(batch)
            targets = batch.pop("targets")
            batch.pop("industry_relevance", None)
            logits = self.model(**batch).prediction.logits
            if logits.ndim != 4 or logits.shape[-1] != 2 or targets.shape != logits.shape[:-1]:
                raise ValueError("Expected binary logits [B,H,N,2] and targets [B,H,N]")
            predictions.append(logits.argmax(dim=-1).reshape(-1))
            truths.append(targets.reshape(-1))
        if not predictions:
            raise ValueError("Evaluation loader contains no batches")
        predicted = torch.cat(predictions)
        truth = torch.cat(truths)
        if predicted.numel() == 0 or predicted.numel() != truth.numel():
            raise ValueError("Predictions and targets must have the same nonzero size")
        tp = ((predicted == 1) & (truth == 1)).sum().float()
        tn = ((predicted == 0) & (truth == 0)).sum().float()
        fp = ((predicted == 1) & (truth == 0)).sum().float()
        fn = ((predicted == 0) & (truth == 1)).sum().float()
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float(torch.where(denominator > 0, (tp * tn - fp * fn) / denominator, 0.0))

    @torch.no_grad()
    def export_predictions(
        self,
        loader: Iterable[Mapping[str, object]],
        output_path: Path,
        *,
        split: str,
        seed: int,
        method: str,
        stock_symbols: Optional[Sequence[str]] = None,
    ) -> int:


        self.model.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        fieldnames = [
            "method",
            "split",
            "seed",
            "sample_index",
            "date",
            "horizon",
            "stock_index",
            "stock_symbol",
            "y_true",
            "prob_down",
            "prob_up",
            "y_pred",
        ]
        row_count = 0
        fallback_index = 0
        with temporary_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for raw_batch in loader:
                batch = self._to_device(raw_batch)
                metadata = self._pop_metadata(batch)
                targets = batch.pop("targets")
                batch.pop("industry_relevance", None)
                logits = self.model(**batch).prediction.logits
                if logits.ndim != 4 or logits.shape[-1] != 2 or targets.shape != logits.shape[:-1]:
                    raise ValueError("Expected binary logits [B,H,N,2] and targets [B,H,N]")
                probabilities = torch.softmax(logits, dim=-1).detach().cpu()
                if not torch.isfinite(probabilities).all():
                    raise ValueError("Cannot export non-finite prediction probabilities")
                target_values = targets.detach().cpu()
                predictions = probabilities.argmax(dim=-1)
                batch_size, horizons, stocks = predictions.shape

                sample_indices = self._metadata_values(
                    metadata.get("sample_index"), batch_size, fallback_index
                )
                dates = self._metadata_values(
                    metadata.get("sample_date"), batch_size, fallback_index,
                    prefix="sample_",
                )
                if stock_symbols is not None and len(stock_symbols) != stocks:
                    raise ValueError(
                        f"Expected {stocks} stock symbols, got {len(stock_symbols)}"
                    )

                for batch_index in range(batch_size):
                    for horizon in range(horizons):
                        for stock_index in range(stocks):
                            symbol = (
                                str(stock_symbols[stock_index])
                                if stock_symbols is not None
                                else f"stock_{stock_index:03d}"
                            )
                            writer.writerow(
                                {
                                    "method": method,
                                    "split": split,
                                    "seed": seed,
                                    "sample_index": sample_indices[batch_index],
                                    "date": dates[batch_index],
                                    "horizon": horizon + 1,
                                    "stock_index": stock_index,
                                    "stock_symbol": symbol,
                                    "y_true": int(target_values[batch_index, horizon, stock_index]),
                                    "prob_down": f"{float(probabilities[batch_index, horizon, stock_index, 0]):.10f}",
                                    "prob_up": f"{float(probabilities[batch_index, horizon, stock_index, 1]):.10f}",
                                    "y_pred": int(predictions[batch_index, horizon, stock_index]),
                                }
                            )
                            row_count += 1
                fallback_index += batch_size
        if row_count == 0:
            temporary_path.unlink()
            raise ValueError("Prediction loader contains no observations")
        temporary_path.replace(output_path)
        return row_count

    def _stage2_optimizer(self) -> torch.optim.Optimizer:
        encoder_parameters = (
            list(self.model.text_encoder.parameters()) if self.model.text_encoder is not None else []
        )
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        main_parameters = [
            parameter for parameter in self.model.parameters() if id(parameter) not in encoder_ids
        ]
        groups = [{"params": main_parameters, "lr": self.training.stage2_lr}]
        if encoder_parameters:
            groups.append(
                {"params": encoder_parameters, "lr": self.training.encoder_finetune_lr}
            )
        return torch.optim.AdamW(groups, weight_decay=self.training.weight_decay)

    @staticmethod
    def _pop_metadata(batch: Dict[str, object]) -> Dict[str, object]:
        return {
            key: batch.pop(key)
            for key in ("sample_index", "sample_date")
            if key in batch
        }

    @staticmethod
    def _metadata_values(
        values: Optional[object],
        batch_size: int,
        fallback_start: int,
        *,
        prefix: str = "",
    ) -> List[object]:
        if values is None:
            return [f"{prefix}{fallback_start + index:06d}" for index in range(batch_size)]
        if torch.is_tensor(values):
            result = values.detach().cpu().reshape(-1).tolist()
        elif isinstance(values, (str, bytes)):
            raise ValueError("Metadata must contain one value per sample, not a scalar string")
        else:
            result = list(values)
        if len(result) != batch_size:
            raise ValueError("Metadata batch size does not match prediction batch size")
        return result

    def _to_device(self, batch: Mapping[str, object]) -> Dict[str, object]:
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
