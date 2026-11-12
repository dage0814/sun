"""Usage: python train.py --data data/bundle.pt --ablation full"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from astgtn.config import AblationConfig, ModelConfig, TextEncoderConfig, TrainingConfig
from astgtn.data import MultimodalWindowDataset, chronological_split
from astgtn.pipeline import MultiModalASTGTN
from astgtn.contract import validate_temporal_bundle, validate_split_boundaries
from astgtn.training import TwoStageTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AST-GTN")
    parser.add_argument("--data", type=Path, required=True, help="Aligned .pt tensor bundle")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--ablation", default="full", help="Ablation preset name")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--tcn-layers", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in (
        "batch_size", "hidden_dim", "graph_layers", "transformer_layers",
        "tcn_layers", "stage2_epochs",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.stage1_epochs < 0:
        parser.error("--stage1-epochs must be nonnegative")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    return args


def load_bundle(path: Path) -> dict:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "quantitative",
        "technical",
        "market_caps",
        "targets",
        "stock_industry",
    }
    missing = required.difference(bundle)
    if missing:
        raise KeyError(f"Bundle is missing required fields: {sorted(missing)}")
    return bundle


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ablation = AblationConfig.preset(args.ablation)
    if not ablation.use_genetic_factors or not ablation.use_ic_selection:
        raise ValueError(
            "GP and IC selection operate before tensor-bundle creation. "
            "Prepare the corresponding feature bundle separately; this training "
            "entry point cannot ablate precomputed features."
        )
    bundle = load_bundle(args.data)
    window_cutoffs, prior_graph, temporal_report = validate_temporal_bundle(
        bundle, use_prior_graph=ablation.use_prior_graph, use_text=ablation.use_text
    )
    quantitative = bundle["quantitative"]
    technical = bundle["technical"]
    stocks = quantitative.shape[2]
    pre_len = bundle["targets"].shape[1]
    stock_industry = bundle["stock_industry"].long()
    num_industries = int(stock_industry.max().item()) + 1

    optional_data = {
        key: bundle[key]
        for key in ("input_ids", "attention_mask", "news_mask", "industry_relevance")
        if key in bundle
    }
    if ablation.use_text:
        missing_text = {"input_ids", "attention_mask"}.difference(optional_data)
        if missing_text:
            raise KeyError(
                f"Text-enabled ablation {args.ablation!r} requires {sorted(missing_text)}"
            )
    if ablation.use_text and ablation.use_listmle and "industry_relevance" not in optional_data:
        raise KeyError("Two-stage full training requires 'industry_relevance' for ListMLE")
    dataset = MultimodalWindowDataset(
        quantitative,
        technical,
        bundle["market_caps"],
        bundle["targets"],
        sample_dates=bundle.get("target_dates", bundle.get("sample_dates", [row[0] for row in bundle["target_times"]])),
        window_cutoffs=window_cutoffs,
        **optional_data,
    )
    purge_gap = max(0, quantitative.shape[1] - 1)
    train_set, validation_set, test_set = chronological_split(dataset, gap=purge_gap)
    validate_split_boundaries(temporal_report, train_set, validation_set, test_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model_config = ModelConfig(
        num_stocks=stocks,
        num_input_features=1,  # Set to the fused feature width by MultiModalASTGTN.
        pre_len=pre_len,
        hidden_dim=args.hidden_dim,
        graph_heads=4,
        graph_layers=args.graph_layers,
        transformer_heads=4,
        transformer_layers=args.transformer_layers,
        tcn_layers=args.tcn_layers,
    )
    text_config = TextEncoderConfig(
        pretrained_model_name=bundle.get("finbert_model_name", "ProsusAI/finbert")
    )
    training_config = TrainingConfig(
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
    )
    model = MultiModalASTGTN(
        model_config=model_config,
        text_config=text_config,
        ablation=ablation,
        num_industries=num_industries,
        num_quant_features=quantitative.shape[-1],
        num_technical_features=technical.shape[-1],
        stock_industry=stock_industry,
        prior_graph=prior_graph,
        fusion_dim=args.hidden_dim,
        fusion_heads=4,
    )
    device = torch.device(args.device) if args.device else None
    trainer = TwoStageTrainer(model, training_config, ablation, device)
    stage1_history = trainer.fit_stage1(train_loader)
    stage2_history = trainer.fit_stage2(train_loader, validation_loader)
    test_mcc = trainer.evaluate_mcc(test_loader)

    run_dir = args.output / args.ablation / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "temporal_audit.json").write_text(
        json.dumps(temporal_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_prediction_path = run_dir / "validation_predictions.csv"
    validation_prediction_rows = trainer.export_predictions(
        validation_loader,
        validation_prediction_path,
        split="validation",
        seed=args.seed,
        method=args.ablation,
        stock_symbols=bundle.get("stock_symbols"),
    )
    prediction_path = run_dir / "test_predictions.csv"
    prediction_rows = trainer.export_predictions(
        test_loader,
        prediction_path,
        split="test",
        seed=args.seed,
        method=args.ablation,
        stock_symbols=bundle.get("stock_symbols"),
    )
    split_manifest = {
        "purge_gap": purge_gap,
        "train_indices": list(train_set.indices),
        "validation_indices": list(validation_set.indices),
        "test_indices": list(test_set.indices),
    }
    (run_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), run_dir / "best_model.pth")
    report = {
        "ablation": asdict(ablation),
        "model": asdict(model.backbone.config),
        "text": asdict(text_config),
        "training": asdict(training_config),
        "stage1_loss": stage1_history,
        "stage2_train_loss": stage2_history.train_loss,
        "validation_mcc": stage2_history.validation_mcc,
        "best_epoch": stage2_history.best_epoch,
        "best_validation_mcc": stage2_history.best_mcc,
        "test_mcc": test_mcc,
        "validation_prediction_file": validation_prediction_path.name,
        "validation_prediction_rows": validation_prediction_rows,
        "test_prediction_file": prediction_path.name,
        "test_prediction_rows": prediction_rows,
        "split_manifest_file": "split_manifest.json",
        "temporal_audit_file": "temporal_audit.json",
        "temporal_policy": temporal_report["policy"],
    }
    (run_dir / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), "test_mcc": test_mcc}, ensure_ascii=False))


if __name__ == "__main__":
    main()
