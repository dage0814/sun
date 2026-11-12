"""Synthetic test fixtures."""
import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

import torch

from astgtn.config import AblationConfig, ModelConfig, TextEncoderConfig, TrainingConfig
from astgtn.data import MultimodalWindowDataset, chronological_split
from astgtn.model import AdaptiveSpatioTemporalGraphTransformer
from astgtn.pipeline import MultiModalASTGTN
from astgtn.history import PointInTimePrior, utc_seconds
from astgtn.contract import validate_temporal_bundle, validate_split_boundaries
from astgtn.topology import DualTopology, PriorGraphBuilder
from astgtn.training import TwoStageTrainer


def event():
    return {"relation_id": "test-edge", "source": 0, "target": 1,
            "known_at": "2018-01-05T00:00:00Z", "effective_at": "2014-01-01T00:00:00Z",
            "source_available_at": "2018-01-05T00:00:00Z", "source_ref": "synthetic fixture",
            "action": "add"}


def bundle_fixture():
    samples, steps, stocks = 32, 3, 2
    start = datetime(2017, 1, 1, 21, tzinfo=timezone.utc)
    windows = [[(start + timedelta(days=i+j)).isoformat() for j in range(steps)] for i in range(samples)]
    return {
        "quantitative": torch.randn(samples, steps, stocks, 2),
        "technical": torch.randn(samples, steps, stocks, 1),
        "market_caps": torch.ones(samples, steps, stocks),
        "targets": torch.zeros(samples, 1, stocks, dtype=torch.long),
        "stock_industry": torch.tensor([0, 0]),
        "stock_symbols": ["AAA", "BBB"],
        "window_cutoffs": windows,
        "prediction_cutoffs": [row[-1] for row in windows],
        "target_times": [[(start + timedelta(days=i+steps)).isoformat()] for i in range(samples)],
        "relation_events": [event()],
        "stock_industry_provenance": [{"known_at": "2016-01-01T00:00:00Z",
            "effective_at": "2016-01-01T00:00:00Z", "source_available_at": "2016-01-01T00:00:00Z",
            "source_ref": "synthetic fixture", "stock_symbol": symbol, "industry_id": 0}
            for symbol in ("AAA", "BBB")],
    }


class TemporalIntegrationTests(unittest.TestCase):
    def config(self):
        return ModelConfig(num_stocks=2, num_input_features=3, hidden_dim=8,
            graph_heads=2, graph_layers=1, transformer_heads=2, transformer_layers=1,
            transformer_ff_dim=16, tcn_channels=8, tcn_layers=1, dropout=0)

    def test_bundle_and_dataset_keep_window_cutoffs(self):
        bundle = bundle_fixture()
        cutoffs, graph, report = validate_temporal_bundle(bundle, use_prior_graph=True, use_text=True)
        self.assertTrue(torch.equal(graph(cutoffs)[..., 0, 1], torch.zeros(32, 3)))
        dataset = MultimodalWindowDataset(**{k: bundle[k] for k in
            ("quantitative", "technical", "market_caps", "targets")}, window_cutoffs=cutoffs)
        self.assertTrue(torch.equal(dataset[0]["window_cutoffs"], cutoffs[0]))
        splits = chronological_split(dataset, gap=2)
        validate_split_boundaries(report, *splits)
        self.assertEqual(report["policy"], "point_in_time_v1")

    def test_missing_times_and_legacy_matrices_fail_closed(self):
        for key in ("window_cutoffs", "prediction_cutoffs", "target_times", "relation_events"):
            bundle = bundle_fixture()
            del bundle[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_temporal_bundle(bundle, use_prior_graph=True, use_text=False)
        for key in ("prior_adjacency", "concept_membership"):
            bundle = bundle_fixture()
            bundle[key] = torch.ones(2, 2)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_temporal_bundle(bundle, use_prior_graph=True, use_text=False)

    def test_bad_sample_boundaries_rejected(self):
        for change in ("date_only", "naive", "future_window", "target", "order"):
            bundle = bundle_fixture()
            if change == "date_only": bundle["window_cutoffs"][0][0] = "2017-01-01"
            if change == "naive": bundle["window_cutoffs"][0][0] = "2017-01-01T21:00:00"
            if change == "future_window": bundle["window_cutoffs"][0][-1] = "2020-01-01T00:00:00Z"
            if change == "target": bundle["target_times"][0][0] = bundle["prediction_cutoffs"][0]
            if change == "order": bundle["prediction_cutoffs"][1] = bundle["prediction_cutoffs"][0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_temporal_bundle(bundle, use_prior_graph=True, use_text=False)

    def test_late_industry_mapping_cannot_bypass_via_text(self):
        bundle = bundle_fixture()
        bundle["stock_industry_provenance"][0]["source_available_at"] = "2018-01-05T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "Industry mapping"):
            validate_temporal_bundle(bundle, use_prior_graph=False, use_text=True)

    def test_split_label_overlap_rejected(self):
        bundle = bundle_fixture()
        _, _, report = validate_temporal_bundle(bundle, use_prior_graph=True, use_text=False)
        splits = chronological_split(list(range(32)), gap=2)
        report["target_times"][0] = [report["prediction_cutoffs"][-1]]
        with self.assertRaisesRegex(ValueError, "Split boundary"):
            validate_split_boundaries(report, *splits)

    def test_static_prior_and_old_checkpoint_rejected(self):
        old = PriorGraphBuilder.from_metadata(torch.tensor([0, 0]))
        with self.assertRaisesRegex(ValueError, "Undated"):
            DualTopology(self.config(), AblationConfig(), old)
        layer = DualTopology(self.config(), AblationConfig())
        state = layer.state_dict()
        state["prior_adjacency"] = old
        layer.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            layer(torch.randn(1, 3, 2, 8))

    def test_model_uses_each_cutoff_and_requires_dates(self):
        ablation = replace(AblationConfig(), use_adaptive_graph=False, use_dynamic_graph=False)
        model = AdaptiveSpatioTemporalGraphTransformer(self.config(), ablation,
            prior_graph=PointInTimePrior(2, [event()]))
        features = torch.randn(1, 3, 2, 3)
        times = torch.tensor([[utc_seconds(x) for x in
            ("2014-01-01T00:00:00Z", "2017-01-01T00:00:00Z", "2018-01-05T00:00:00Z")]])
        with self.assertRaises(ValueError): model(features)
        output = model(features, times)
        self.assertEqual(float(output.adjacency[0, 0, 0, 1]), 0)
        self.assertEqual(float(output.adjacency[0, 1, 0, 1]), 0)
        self.assertGreater(float(output.adjacency[0, 2, 0, 1]), 0)
        output.logits.square().sum().backward()
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_mixup_and_evaluation_propagate_cutoffs(self):
        ablation = AblationConfig.preset("no_text")
        model = MultiModalASTGTN(model_config=self.config(), text_config=TextEncoderConfig(),
            ablation=ablation, num_industries=1, num_quant_features=2, num_technical_features=1,
            stock_industry=torch.tensor([0, 0]), prior_graph=PointInTimePrior(2, [event()]),
            fusion_dim=8, fusion_heads=2)
        bundle = bundle_fixture()
        cutoffs, _, _ = validate_temporal_bundle(bundle, use_prior_graph=True, use_text=False)
        batch = {k: bundle[k][:2] for k in ("quantitative", "technical", "market_caps", "targets")}
        batch["window_cutoffs"] = cutoffs[:2]
        seen = []
        handle = model.backbone.topology.prior_graph.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        trainer = TwoStageTrainer(model, TrainingConfig(stage2_epochs=1), ablation, torch.device("cpu"))
        with patch("astgtn.losses.torch.randperm", return_value=torch.tensor([1, 0])):
            trainer.fit_stage2([batch])
        self.assertTrue(torch.equal(seen[0], cutoffs[0:1].expand(2, -1)))
        trainer.evaluate_mcc([batch])
        self.assertTrue(torch.equal(seen[-1], cutoffs[:2]))
        handle.remove()


if __name__ == "__main__":
    unittest.main()
