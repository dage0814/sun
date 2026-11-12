import copy
from datetime import datetime, timedelta, timezone
import unittest

import torch

from astgtn.contract import validate_temporal_bundle, validate_split_boundaries
from astgtn.data import MultimodalWindowDataset, chronological_split


def bundle_fixture():
    samples, steps, stocks = 32, 3, 2
    start = datetime(2017, 1, 1, 21, tzinfo=timezone.utc)
    windows = [[(start + timedelta(days=i + j)).isoformat() for j in range(steps)]
               for i in range(samples)]
    return {
        "quantitative": torch.zeros(samples, steps, stocks, 2),
        "technical": torch.zeros(samples, steps, stocks, 1),
        "market_caps": torch.ones(samples, steps, stocks),
        "targets": torch.zeros(samples, 1, stocks, dtype=torch.long),
        "stock_industry": torch.tensor([0, 1]),
        "stock_symbols": ["AAA", "BBB"],
        "window_cutoffs": windows,
        "prediction_cutoffs": [row[-1] for row in windows],
        "target_times": [[(start + timedelta(days=i + steps)).isoformat()] for i in range(samples)],
        "relation_events": [],
        "stock_industry_provenance": [{
            "known_at": "2016-01-01T00:00:00Z", "effective_at": "2016-01-01T00:00:00Z",
            "source_available_at": "2016-01-01T00:00:00Z", "source_ref": "test fixture",
            "stock_symbol": symbol, "industry_id": i,
        } for i, symbol in enumerate(("AAA", "BBB"))],
    }


class SplitValidationTests(unittest.TestCase):
    def test_symmetric_gap_preserved(self):
        train, val, test = chronological_split(list(range(100)), gap=2)
        self.assertEqual(list(train.indices), list(range(68)))
        self.assertEqual(list(val.indices), list(range(72, 83)))
        self.assertEqual(list(test.indices), list(range(87, 100)))

    def test_empty_splits_are_rejected_before_return(self):
        for count, gap in ((0, 0), (1, 0), (32, 3), (100, 50)):
            with self.subTest(count=count, gap=gap), self.assertRaisesRegex(ValueError, "empty"):
                chronological_split(list(range(count)), gap=gap)

    def test_bad_gap_and_ratios(self):
        for gap in (-1, 1.5, True):
            with self.subTest(gap=gap), self.assertRaisesRegex(ValueError, "gap"):
                chronological_split(list(range(100)), gap=gap)
        for ratio in (0, float("nan"), float("inf"), True):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                chronological_split(list(range(100)), val_ratio=ratio)


class BundleValidationTests(unittest.TestCase):
    def validate(self, bundle, text=False):
        return validate_temporal_bundle(bundle, use_prior_graph=True, use_text=text)

    def test_valid_boundaries_and_integer_semantics(self):
        bundle = bundle_fixture()
        bundle["stock_industry"] = bundle["stock_industry"].float()
        bundle["targets"] = bundle["targets"].float()
        cutoffs, _, report = self.validate(bundle)
        subsets = chronological_split(list(range(32)), gap=2)
        validate_split_boundaries(report, *subsets)
        self.assertEqual(cutoffs.shape, (32, 3))

    def test_bad_shapes(self):
        changes = {
            "quantitative": torch.zeros(32, 3, 2),
            "technical": torch.zeros(32, 2, 2, 1),
            "market_caps": torch.ones(32, 3, 1),
            "targets": torch.zeros(32, 2),
            "stock_industry": torch.zeros(3),
        }
        for key, value in changes.items():
            bundle = bundle_fixture()
            bundle[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(bundle)

    def test_bad_values(self):
        changes = {
            "quantitative": float("nan"), "technical": float("inf"),
            "market_caps": 0, "targets": 0.5, "stock_industry": 0.5,
        }
        for key, value in changes.items():
            bundle = bundle_fixture()
            bundle[key] = bundle[key].float()
            bundle[key].reshape(-1)[0] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(bundle)

    def test_text_tensor_alignment(self):
        bundle = bundle_fixture()
        bundle.update({"input_ids": torch.zeros(32, 3, 2, 4, dtype=torch.long),
                       "attention_mask": torch.ones(32, 3, 2, 4, dtype=torch.long),
                       "news_mask": torch.ones(32, 3, 2, dtype=torch.bool),
                       "industry_relevance": torch.zeros(32, 3, 2)})
        self.validate(bundle, True)
        changes = {"industry_relevance": torch.zeros(32, 3, 3),
                   "attention_mask": torch.zeros(32, 3, 2, 4, dtype=torch.long),
                   "input_ids": torch.full((32, 3, 2, 4), 0.25)}
        for key, value in changes.items():
            candidate = copy.deepcopy(bundle)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(candidate, True)

    def test_dataset_date_labels_reject_arbitrary_objects(self):
        bundle = bundle_fixture()
        tensors = {key: bundle[key] for key in ("quantitative", "technical", "market_caps", "targets")}
        with self.assertRaisesRegex(ValueError, "sample_dates"):
            MultimodalWindowDataset(**tensors, sample_dates=[object()] * 32)


if __name__ == "__main__":
    unittest.main()
