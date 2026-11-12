from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Subset


class MultimodalWindowDataset(Dataset):
    """Inputs must already be aligned, with the sample dimension first."""

    def __init__(
        self,
        quantitative: torch.Tensor,
        technical: torch.Tensor,
        market_caps: torch.Tensor,
        targets: torch.Tensor,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        news_mask: Optional[torch.Tensor] = None,
        industry_relevance: Optional[torch.Tensor] = None,
        sample_dates: Optional[Sequence[object]] = None,
        window_cutoffs: Optional[torch.Tensor] = None,
    ) -> None:
        tensors = {
            "quantitative": quantitative,
            "technical": technical,
            "market_caps": market_caps,
            "targets": targets,
        }
        optional = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "news_mask": news_mask,
            "industry_relevance": industry_relevance,
            "window_cutoffs": window_cutoffs,
        }
        tensors.update({key: value for key, value in optional.items() if value is not None})
        sample_counts = {key: value.shape[0] for key, value in tensors.items()}
        if len(set(sample_counts.values())) != 1:
            raise ValueError(f"All tensors must share the sample dimension: {sample_counts}")
        self.tensors = tensors
        self.sample_dates = None
        if sample_dates is not None:
            if len(sample_dates) != next(iter(sample_counts.values())):
                raise ValueError("sample_dates must contain one value per date-level sample")
            self.sample_dates = [str(value) for value in sample_dates]

    def __len__(self) -> int:
        return next(iter(self.tensors.values())).shape[0]

    def __getitem__(self, index: int) -> Dict[str, object]:
        item: Dict[str, object] = {
            key: value[index] for key, value in self.tensors.items()
        }
        item["sample_index"] = torch.tensor(index, dtype=torch.long)
        if self.sample_dates is not None:
            item["sample_date"] = self.sample_dates[index]
        return item


def chronological_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    gap: int = 0,
) -> Tuple[Subset, Subset, Subset]:
    """Split chronologically; gap removes overlapping windows at split boundaries."""

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid train/validation ratios")
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_indices = range(0, max(0, train_end - gap))
    val_indices = range(min(total, train_end + gap), max(train_end + gap, val_end - gap))
    test_indices = range(min(total, val_end + gap), total)
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)
