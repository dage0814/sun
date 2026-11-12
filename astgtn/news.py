from __future__ import annotations

import re
from datetime import date, datetime, time
from numbers import Integral
from typing import Dict, Iterable, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


class NewsWindowPreprocessor:

    def __init__(
        self,
        timezone: str = "America/New_York",
        close_time: time = time(16, 0),
        open_time: time = time(9, 30),
        max_news_per_day: int = 64,
        source_timezone: str | None = None,
    ) -> None:
        ZoneInfo(timezone)
        if source_timezone is not None:
            ZoneInfo(source_timezone)
        if any(not isinstance(value, time) or value.tzinfo is not None
               for value in (close_time, open_time)):
            raise ValueError("close_time and open_time must be local wall-clock times")
        if isinstance(max_news_per_day, bool) or not isinstance(max_news_per_day, Integral) or max_news_per_day < 1:
            raise ValueError("max_news_per_day must be a positive integer")
        self.timezone = timezone
        self.source_timezone = source_timezone
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
        timestamps = [self._publication_time(value) for value in frame[timestamp_col]]
        frame[timestamp_col] = pd.to_datetime(timestamps, utc=True).tz_convert(self.timezone)
        frame[title_col] = frame[title_col].fillna("").astype(str).map(self._normalize_title)
        frame = frame.dropna(subset=[timestamp_col])
        frame = frame[frame[title_col].str.len() > 0].sort_values(timestamp_col, kind="stable")
        frame = frame.drop_duplicates(subset=[timestamp_col, title_col], keep="first")

        calendar = []
        for value in trading_days:
            if not isinstance(value, (str, date, datetime, pd.Timestamp)):
                raise ValueError("trading_days must contain explicit calendar dates")
            day = pd.Timestamp(value)
            if pd.isna(day):
                raise ValueError("trading_days cannot contain missing dates")
            if day.tzinfo is not None:
                day = day.tz_convert(self.timezone)
            calendar.append(pd.Timestamp(day.date()))
        days = pd.DatetimeIndex(calendar).sort_values().unique()
        if len(days) < 2:
            raise ValueError("trading_days must include the previous session and at least one prediction session")
        windows: Dict[pd.Timestamp, List[str]] = {}
        for index in range(1, len(days)):
            previous_day, prediction_day = days[index - 1], days[index]
            start = self._combine(previous_day, self.close_time)
            end = self._combine(prediction_day, self.open_time)
            selected = frame.loc[
                (frame[timestamp_col] > start) & (frame[timestamp_col] <= end), title_col
            ]
            windows[prediction_day] = selected.tail(self.max_news_per_day).tolist()
        return windows

    def _publication_time(self, value) -> pd.Timestamp:
        if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
            return pd.NaT
        if not isinstance(value, (str, datetime, pd.Timestamp)):
            raise ValueError("News publication times must be explicit timestamps, not numeric epochs or dates")
        if isinstance(value, str) and not re.search(r"[T ]\d{2}:\d{2}", value):
            raise ValueError("News publication timestamps must include a time of day")
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            return pd.NaT
        if pd.isna(stamp):
            return pd.NaT
        if stamp.tzinfo is None:
            if self.source_timezone is None:
                raise ValueError("Naive news timestamps require an explicit source_timezone")
            stamp = stamp.tz_localize(self.source_timezone, ambiguous="raise", nonexistent="raise")
        return stamp.tz_convert(self.timezone)

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


    for name, value in (("max_news", max_news), ("max_length", max_length)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0
    if isinstance(pad_id, bool) or not isinstance(pad_id, Integral) or pad_id < 0:
        raise ValueError("tokenizer.pad_token_id must be a nonnegative integer")
    input_days, attention_days, day_masks = [], [], []
    for titles in windows:
        if isinstance(titles, (str, bytes)):
            raise ValueError("Each news window must be a sequence of title strings")
        rows = list(titles)
        if any(not isinstance(title, str) for title in rows):
            raise ValueError("News titles must be strings")
        normalized = [NewsWindowPreprocessor._normalize_title(title) for title in rows]
        chosen = [title for title in normalized if title][-max_news:]
        count = len(chosen)
        input_ids = torch.full((max_news, max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(max_news, max_length, dtype=torch.long)
        if count:
            encoded = tokenizer(
                chosen, padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            for name, value in (("input_ids", ids), ("attention_mask", mask)):
                if not isinstance(value, torch.Tensor) or value.shape != (count, max_length):
                    raise ValueError(f"Tokenizer {name} must use shape [title count,max_length]")
                if value.is_complex() or not torch.isfinite(value).all().item():
                    raise ValueError(f"Tokenizer {name} must contain finite integer values")
                if (value < 0).any().item() or (value.is_floating_point() and not torch.equal(value, value.trunc())):
                    raise ValueError(f"Tokenizer {name} must contain nonnegative integer values")
            if (mask > 1).any().item() or not mask.bool().any(dim=-1).all().item():
                raise ValueError("Each real headline requires a binary attention mask with valid tokens")
            input_ids[:count] = ids.long().cpu()
            attention_mask[:count] = mask.long().cpu()
        input_days.append(input_ids)
        attention_days.append(attention_mask)
        mask = torch.zeros(max_news, dtype=torch.bool)
        mask[:count] = True
        day_masks.append(mask)
    if not input_days:
        return (torch.empty((0, max_news, max_length), dtype=torch.long),
                torch.empty((0, max_news, max_length), dtype=torch.long),
                torch.empty((0, max_news), dtype=torch.bool))
    return torch.stack(input_days), torch.stack(attention_days), torch.stack(day_masks)
