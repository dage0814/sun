from __future__ import annotations

import re
from datetime import time
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch


class NewsWindowPreprocessor:

    def __init__(
        self,
        timezone: str = "America/New_York",
        close_time: time = time(16, 0),
        open_time: time = time(9, 30),
        max_news_per_day: int = 64,
    ) -> None:
        self.timezone = timezone
        self.close_time = close_time
        self.open_time = open_time
        self.max_news_per_day = max_news_per_day

    def transform(
        self,
        news: pd.DataFrame,
        trading_days: Iterable[pd.Timestamp],
        *,
        timestamp_col: str = "published_at",
        title_col: str = "title",
    ) -> Dict[pd.Timestamp, List[str]]:
        if timestamp_col not in news or title_col not in news:
            raise KeyError(f"news must contain {timestamp_col!r} and {title_col!r}")
        frame = news[[timestamp_col, title_col]].copy()
        timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(self.timezone)
        else:
            timestamps = timestamps.dt.tz_convert(self.timezone)
        frame[timestamp_col] = timestamps
        frame[title_col] = frame[title_col].fillna("").astype(str).map(self._normalize_title)
        frame = frame.dropna(subset=[timestamp_col])
        frame = frame[frame[title_col].str.len() > 0].sort_values(timestamp_col)
        frame = frame.drop_duplicates(subset=[timestamp_col, title_col], keep="first")

        days = pd.DatetimeIndex(pd.to_datetime(list(trading_days))).normalize().sort_values().unique()
        windows: Dict[pd.Timestamp, List[str]] = {}
        for index in range(1, len(days)):
            previous_day, prediction_day = days[index - 1], days[index]
            start = self._combine(previous_day, self.close_time)
            end = self._combine(prediction_day, self.open_time)
            selected = frame.loc[
                (frame[timestamp_col] > start) & (frame[timestamp_col] <= end), title_col
            ].drop_duplicates()
            windows[prediction_day] = selected.head(self.max_news_per_day).tolist()
        return windows

    def _combine(self, day: pd.Timestamp, clock: time) -> pd.Timestamp:
        value = pd.Timestamp.combine(day.date(), clock)
        return value.tz_localize(self.timezone)

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", title.strip())


def tokenize_news_windows(
    windows: Sequence[Sequence[str]],
    tokenizer,
    *,
    max_news: int = 64,
    max_length: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Output: [T, M, L] tokens. news_mask marks real headlines; empty days use an all-zero masked placeholder."""

    input_days, attention_days, day_masks = [], [], []
    for titles in windows:
        chosen = list(titles)[:max_news]
        encoded = tokenizer(
            chosen or [""],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        count = len(chosen)
        input_ids = torch.zeros(max_news, max_length, dtype=torch.long)
        attention_mask = torch.zeros(max_news, max_length, dtype=torch.long)
        if count:
            input_ids[:count] = encoded["input_ids"][:count]
            attention_mask[:count] = encoded["attention_mask"][:count]
        input_days.append(input_ids)
        attention_days.append(attention_mask)
        mask = torch.zeros(max_news, dtype=torch.bool)
        mask[:count] = True
        day_masks.append(mask)
    return torch.stack(input_days), torch.stack(attention_days), torch.stack(day_masks)
