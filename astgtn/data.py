from __future__ import annotations

from datetime import date, datetime, time, timezone
import math
from numbers import Integral, Real
import re
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Subset

from .history import utc_seconds


def _date_label(value: str | date | datetime) -> str:
    if isinstance(value, str):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value).isoformat()
        utc_seconds(value)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        if value.tzinfo is None and value.time() == time():
            return value.date().isoformat()
        utc_seconds(value)
        parsed = value
    elif isinstance(value, date):
        return value.isoformat()
    else:
        raise ValueError("sample_dates must contain ISO dates or timezone-aware timestamps")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class MultimodalWindowDataset(Dataset):


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
        sample_dates: Optional[Sequence[str | date | datetime]] = None,
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
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise ValueError(f"{key} must be a tensor with a sample dimension")
        sample_counts = {key: value.shape[0] for key, value in tensors.items()}
        if len(set(sample_counts.values())) != 1:
            raise ValueError(f"All tensors must share the sample dimension: {sample_counts}")
        self.tensors = tensors
        self.sample_dates = None
        if sample_dates is not None:
            if len(sample_dates) != next(iter(sample_counts.values())):
                raise ValueError("sample_dates must contain one value per date-level sample")
            try:
                self.sample_dates = [_date_label(value) for value in sample_dates]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample_dates: {exc}") from exc
            if len({"T" in value for value in self.sample_dates}) > 1:
                raise ValueError("sample_dates must consistently use dates or UTC timestamps")
            if any(left >= right for left, right in zip(self.sample_dates, self.sample_dates[1:])):
                raise ValueError("sample_dates must increase strictly without duplicates")

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


    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
           for value in (train_ratio, val_ratio)):
        raise ValueError("Train/validation ratios must be finite numbers")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid train/validation ratios")
    if isinstance(gap, bool) or not isinstance(gap, Integral) or gap < 0:
        raise ValueError("gap must be a nonnegative integer")
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_stop, val_start, val_stop, test_start = (
        train_end - gap, train_end + gap, val_end - gap, val_end + gap
    )
    sizes = (train_stop, val_stop - val_start, total - test_start)
    if min(sizes) < 1:
        raise ValueError(
            f"Temporal split would be empty: samples={total}, gap={gap}, "
            f"train/validation/test sizes={sizes}. Supply more samples or revise the split settings."
        )
    train_indices = range(0, train_stop)
    val_indices = range(val_start, val_stop)
    test_indices = range(test_start, total)
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)
