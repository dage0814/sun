from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import torch


class CommandLineSmokeTests(unittest.TestCase):

    def test_undated_bundle_stops_before_training(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="astgtn_undated_") as temporary:
            folder = Path(temporary)
            bundle_path = folder / "undated.pt"
            torch.save({
                "quantitative": torch.ones(10, 3, 2, 2),
                "technical": torch.ones(10, 3, 2, 1),
                "market_caps": torch.ones(10, 3, 2),
                "targets": torch.zeros(10, 1, 2, dtype=torch.long),
                "stock_industry": torch.tensor([0, 0]),
                "prior_adjacency": torch.ones(2, 2),
            }, bundle_path)
            completed = subprocess.run([
                sys.executable, str(project_root / "train.py"),
                "--data", str(bundle_path), "--output", str(folder / "results"),
            ], cwd=project_root, capture_output=True, text=True, timeout=30, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Historical runs require", completed.stderr)
            self.assertFalse((folder / "results").exists())

    def test_two_stage_training_entry_point_writes_valid_artifacts(self) -> None:
        torch.manual_seed(17)
        samples, history, stocks = 32, 3, 3
        bundle = {
            "quantitative": torch.randn(samples, history, stocks, 2),
            "technical": torch.randn(samples, history, stocks, 2),
            "market_caps": torch.rand(samples, history, stocks) * 100.0 + 1.0,
            "targets": torch.randint(0, 2, (samples, 1, stocks)),
            "stock_industry": torch.tensor([0, 0, 1]),
            "input_ids": torch.randint(0, 100, (samples, history, 1, 4)),
            "attention_mask": torch.ones(samples, history, 1, 4, dtype=torch.long),
            "news_mask": torch.ones(samples, history, 1, dtype=torch.bool),
            "industry_relevance": torch.randn(samples, history, 2),
            "sample_dates": [(datetime(2017, 1, 4) + timedelta(days=index)).date().isoformat() for index in range(samples)],
            "stock_symbols": ["AAA", "BBB", "CCC"],
        }

        origin = datetime(2017, 1, 1, 21, tzinfo=timezone.utc)
        bundle["window_cutoffs"] = [
            [(origin + timedelta(days=i+j)).isoformat() for j in range(history)]
            for i in range(samples)
        ]
        bundle["prediction_cutoffs"] = [row[-1] for row in bundle["window_cutoffs"]]
        bundle["target_times"] = [[(origin + timedelta(days=i+history)).isoformat()] for i in range(samples)]
        bundle["relation_events"] = [{
            "relation_id": "synthetic-industry", "source": 0, "target": 1,
            "known_at": "2017-01-01T00:00:00Z", "effective_at": "2017-01-01T00:00:00Z",
            "source_available_at": "2017-01-01T00:00:00Z", "source_ref": "synthetic test fixture",
            "action": "add",
        }]
        bundle["stock_industry_provenance"] = [{
            "known_at": "2017-01-01T00:00:00Z", "effective_at": "2017-01-01T00:00:00Z",
            "source_available_at": "2017-01-01T00:00:00Z", "source_ref": "synthetic test fixture",
            "stock_symbol": symbol, "industry_id": int(bundle["stock_industry"][i]),
        } for i, symbol in enumerate(bundle["stock_symbols"])]

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="astgtn_cli_") as temporary:
            temporary_path = Path(temporary)
            bundle_path = temporary_path / "bundle.pt"
            output_path = temporary_path / "results"
            torch.save(bundle, bundle_path)
            environment = os.environ.copy()
            environment.setdefault("MKL_THREADING_LAYER", "TBB")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "train.py"),
                    "--data",
                    str(bundle_path),
                    "--output",
                    str(output_path),
                    "--ablation",
                    "no_finbert",
                    "--device",
                    "cpu",
                    "--batch-size",
                    "4",
                    "--hidden-dim",
                    "8",
                    "--graph-layers",
                    "1",
                    "--transformer-layers",
                    "1",
                    "--tcn-layers",
                    "1",
                    "--stage1-epochs",
                    "1",
                    "--stage2-epochs",
                    "1",
                ],
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            run_directory = output_path / "no_finbert" / "seed_42"
            model_path = run_directory / "best_model.pth"
            report_path = run_directory / "run.json"
            predictions_path = run_directory / "test_predictions.csv"
            validation_predictions_path = run_directory / "validation_predictions.csv"
            split_manifest_path = run_directory / "split_manifest.json"
            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(predictions_path.is_file())
            self.assertTrue(validation_predictions_path.is_file())
            self.assertTrue(split_manifest_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["stage1_loss"]), 1)
            self.assertEqual(len(report["stage2_train_loss"]), 1)
            self.assertIsInstance(report["test_mcc"], float)
            self.assertGreater(report["test_prediction_rows"], 0)
            self.assertGreater(report["validation_prediction_rows"], 0)
            audit = json.loads((run_directory / "temporal_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["policy"], "point_in_time_v1")
            self.assertEqual(len(audit["window_cutoffs"]), samples)
            self.assertEqual(audit["relation_events"][0]["relation_id"], "synthetic-industry")
            self.assertEqual(audit["relation_events"][0]["known_at"], "2017-01-01T00:00:00Z")

            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self.assertTrue(state)
            self.assertTrue(all(torch.isfinite(tensor).all() for tensor in state.values()))


if __name__ == "__main__":
    unittest.main()
