"""Synthetic test fixtures."""

import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bootstrap import bootstrap_comparison, holm_adjust, load_predictions


class BootstrapTests(unittest.TestCase):
    def write_predictions(self, folder, name, change=None):
        rows = []
        for day in range(1, 5):
            for stock in range(2):
                rows.append({
                    "seed": 42, "date": f"2020-01-{day:02d}",
                    "horizon": 1, "stock_index": stock,
                    "stock_symbol": ["AAA", "BBB"][stock],
                    "split": "test", "y_true": stock, "y_pred": stock,
                })
        if change:
            change(rows)
        target = folder / name
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        return target

    def test_identical_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.write_predictions(folder, "full.csv")
            self.write_predictions(folder, "base.csv")
            result = bootstrap_comparison(
                "full.csv", "base.csv", 2, 10, random.Random(42), base=folder
            )
            self.assertEqual(result["delta_mcc"], 0)
            self.assertEqual(result["ci_low"], 0)
            self.assertEqual(result["p_raw"], 1)

    def test_mismatched_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            full = self.write_predictions(folder, "full.csv")
            for field, value in [("y_true", 1), ("stock_index", 4),
                                 ("stock_symbol", "CCC"), ("split", "validation")]:
                baseline = self.write_predictions(
                    folder, "base.csv", lambda rows: rows[0].update({field: value})
                )
                with self.subTest(field=field), self.assertRaises(ValueError):
                    bootstrap_comparison(full, baseline, 2, 10, random.Random(42))

    def test_invalid_labels_and_repetitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            invalid = self.write_predictions(
                folder, "invalid.csv", lambda rows: rows[0].update(y_pred=2)
            )
            with self.assertRaisesRegex(ValueError, "binary"):
                load_predictions(invalid)
            valid = self.write_predictions(folder, "valid.csv")
            with self.assertRaisesRegex(ValueError, "repetitions"):
                bootstrap_comparison(valid, valid, 2, 0, random.Random(42))

    def test_inconsistent_seed_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            first = self.write_predictions(folder, "first.csv")
            def add_seed(rows):
                rows.extend([dict(row, seed=43) for row in list(rows)])
                rows[-1]["stock_symbol"] = "CCC"
            full = self.write_predictions(folder, "full.csv", add_seed)
            with self.assertRaisesRegex(ValueError, "same observations"):
                bootstrap_comparison(full, full, 2, 10, random.Random(42))

    def test_cli_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.write_predictions(folder, "full.csv")
            self.write_predictions(folder, "base.csv")
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps([
                {"name": str(index), "full": "full.csv", "baseline": "base.csv"}
                for index in range(6)
            ]), encoding="utf-8")
            output = folder / "output.csv"
            completed = subprocess.run([
                sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "bootstrap.py"),
                "--manifest", str(manifest), "--output", str(output),
                "--block-length", "2", "--repetitions", "10",
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists())

    def test_holm_adjustment(self):
        values = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(values, [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
