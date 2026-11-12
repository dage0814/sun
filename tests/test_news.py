from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd
import torch

from astgtn.data import MultimodalWindowDataset
from astgtn.news import NewsWindowPreprocessor, tokenize_news_windows


class TinyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen = []

    def __call__(self, rows, **kwargs):
        self.seen.append(list(rows))
        ids = torch.ones(len(rows), kwargs["max_length"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids.clone()}


class NewsWindowTests(unittest.TestCase):
    def test_latest_titles_keep_ascending_order(self):
        news = pd.DataFrame({
            "published_at": ["2024-01-03T09:00:00-05:00", "2024-01-02T17:00:00-05:00",
                             "2024-01-03T08:00:00-05:00", "2024-01-03T10:00:00-05:00"],
            "title": ["Latest", "Early", "Recent", "Too late"],
        })
        result = NewsWindowPreprocessor(max_news_per_day=2).transform(news, ["2024-01-02", "2024-01-03"])
        self.assertEqual(result[pd.Timestamp("2024-01-03")], ["Recent", "Latest"])

    def test_only_identical_timestamp_title_pairs_are_deduplicated(self):
        news = pd.DataFrame({
            "published_at": ["2024-01-03T08:00:00-05:00"] * 3 + ["2024-01-03T09:00:00-05:00"],
            "title": ["Same", " Same ", "Other", "Same"],
        })
        result = NewsWindowPreprocessor().transform(news, ["2024-01-02", "2024-01-03"])
        self.assertEqual(result[pd.Timestamp("2024-01-03")], ["Same", "Other", "Same"])

    def test_weekend_and_dst_use_each_local_session(self):
        news = pd.DataFrame({
            "published_at": ["2024-03-08T20:30:00Z", "2024-03-08T21:30:00Z",
                             "2024-03-11T09:00:00-04:00"],
            "title": ["Before close", "After close", "Monday morning"],
        })
        result = NewsWindowPreprocessor().transform(news, ["2024-03-08", "2024-03-11"])
        self.assertEqual(result[pd.Timestamp("2024-03-11")], ["After close", "Monday morning"])

    def test_naive_times_require_declared_source_timezone(self):
        news = pd.DataFrame({"published_at": ["2024-01-03 08:00:00"], "title": ["Title"]})
        with self.assertRaisesRegex(ValueError, "source_timezone"):
            NewsWindowPreprocessor().transform(news, ["2024-01-02", "2024-01-03"])
        result = NewsWindowPreprocessor(source_timezone="America/New_York").transform(
            news, ["2024-01-02", "2024-01-03"])
        self.assertEqual(result[pd.Timestamp("2024-01-03")], ["Title"])

    def test_empty_and_missing_news_preserve_session(self):
        news = pd.DataFrame({"published_at": [None], "title": ["Missing time"]})
        result = NewsWindowPreprocessor().transform(news, ["2024-01-02", "2024-01-03"])
        self.assertEqual(result[pd.Timestamp("2024-01-03")], [])

    def test_bad_capacity_and_incomplete_calendar(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NewsWindowPreprocessor(max_news_per_day=value)
        with self.assertRaisesRegex(ValueError, "previous session"):
            NewsWindowPreprocessor().transform(pd.DataFrame(columns=["published_at", "title"]), ["2024-01-03"])

    def test_tokenizer_latest_truncation_and_empty_days(self):
        tokenizer = TinyTokenizer()
        ids, attention, mask = tokenize_news_windows([[], ["Old", "New", "Latest"]], tokenizer, max_news=2, max_length=4)
        self.assertEqual(tokenizer.seen, [["New", "Latest"]])
        self.assertEqual(tuple(ids.shape), (2, 2, 4))
        self.assertFalse(mask[0].any())
        self.assertFalse(attention[0].any())
        self.assertTrue(mask[1].all())

    def test_empty_window_batch_has_consistent_shapes(self):
        result = tokenize_news_windows([], TinyTokenizer(), max_news=2, max_length=4)
        self.assertEqual([tuple(value.shape) for value in result], [(0, 2, 4), (0, 2, 4), (0, 2)])

    def test_empty_title_is_not_a_valid_headline(self):
        tokenizer = TinyTokenizer()
        _, _, mask = tokenize_news_windows([["  "]], tokenizer, max_news=2, max_length=4)
        self.assertFalse(mask.any())
        self.assertEqual(tokenizer.seen, [])


class DateLabelTests(unittest.TestCase):
    def dataset(self, values):
        count = len(values)
        return MultimodalWindowDataset(
            torch.zeros(count, 1, 1, 1), torch.zeros(count, 1, 1, 1),
            torch.ones(count, 1, 1), torch.zeros(count, 1, 1), sample_dates=values)

    def test_timezone_normalization_preserves_sort_order(self):
        result = self.dataset(["2024-01-01T20:00:00-05:00", "2024-01-02T02:00:00Z"])
        self.assertEqual(result.sample_dates, ["2024-01-02T01:00:00Z", "2024-01-02T02:00:00Z"])

    def test_dates_and_midnight_datetimes_become_iso_dates(self):
        result = self.dataset([datetime(2024, 1, 1), "2024-01-02"])
        self.assertEqual(result.sample_dates, ["2024-01-01", "2024-01-02"])

    def test_ambiguous_reversed_duplicate_or_mixed_labels_rejected(self):
        cases = [["1/2/2024"], ["2024-01-02", "2024-01-01"],
                 ["2024-01-01", "2024-01-01"], ["2024-01-01", "2024-01-02T00:00:00Z"],
                 [datetime(2024, 1, 1, 9, 30)]]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.dataset(values)


if __name__ == "__main__":
    unittest.main()
