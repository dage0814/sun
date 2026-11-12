import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

import train
from astgtn.config import ModelConfig, TextEncoderConfig, TrainingConfig


class ConfigReviewTests(unittest.TestCase):
    def test_invalid_model_dimensions(self):
        config = ModelConfig(num_stocks=3, num_input_features=9)
        for changes in (
            {"graph_heads": 0}, {"num_input_features": 0}, {"pre_len": -1},
            {"num_classes": 3}, {"dropout": 1.0}, {"prior_graph_weight": -1},
            {"hidden_dim": 3}, {"graph_layers": 1.5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)

    def test_invalid_training_config(self):
        config = TrainingConfig()
        for changes in (
            {"stage2_epochs": 0}, {"stage1_epochs": -1}, {"patience": 0},
            {"stage2_lr": float("nan")}, {"mcc_weight": -1},
            {"focal_alpha": 1.1}, {"mixup_alpha": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)

    def test_invalid_text_config(self):
        for changes in ({"fallback_vocab_size": 0}, {"freeze_finbert_layers": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TextEncoderConfig(**changes)

    def test_bundle_requires_dictionary_and_tensors(self):
        with patch("train.torch.load", return_value=[]), self.assertRaises(TypeError):
            train.load_bundle("unused.pt")
        bundle = {name: torch.zeros(1) for name in (
            "quantitative", "technical", "market_caps", "targets", "stock_industry",
        )}
        with patch("train.torch.load", return_value=bundle) as load:
            self.assertIs(train.load_bundle("unused.pt"), bundle)
            self.assertTrue(load.call_args.kwargs["weights_only"])
        bundle["targets"] = [0]
        with patch("train.torch.load", return_value=bundle), self.assertRaises(TypeError):
            train.load_bundle("unused.pt")


if __name__ == "__main__":
    unittest.main()
