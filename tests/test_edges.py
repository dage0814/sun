import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from astgtn.config import AblationConfig, ModelConfig, TextEncoderConfig, TrainingConfig
from astgtn.fusion import MarketCapAdapter
from astgtn.pipeline import MultiModalASTGTN
from astgtn.text_encoder import FinFEBERTEncoder
from astgtn.topology import DualTopology
from astgtn.training import TwoStageTrainer


class ModelReviewTests(unittest.TestCase):
    def test_prior_conflict_is_rejected_before_text_initialization(self):
        with patch("astgtn.pipeline.FinFEBERTEncoder") as encoder:
            with self.assertRaisesRegex(ValueError, "not both"):
                MultiModalASTGTN(
                    model_config=ModelConfig(num_stocks=2, num_input_features=3),
                    text_config=TextEncoderConfig(), ablation=AblationConfig(),
                    num_industries=1, num_quant_features=2, num_technical_features=1,
                    stock_industry=torch.tensor([0, 0]),
                    prior_adjacency=torch.eye(2), prior_graph=object(),
                )
            encoder.assert_not_called()

    def test_direct_topology_rejects_two_priors(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            DualTopology(
                ModelConfig(num_stocks=2, num_input_features=3),
                AblationConfig(), torch.eye(2), object(),
            )

    def test_single_stock_industry_preserves_news(self):
        adapter = MarketCapAdapter()
        industry = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        mapping = torch.tensor([0, 1, 1])
        caps = torch.tensor([[[10.0, 20.0, 40.0]]])
        news, weights = adapter(industry, mapping, caps)
        torch.testing.assert_close(weights, torch.tensor([[[1.0, 1.6, 0.4]]]))
        torch.testing.assert_close(news[:, :, 0], industry[:, :, 0])

    def test_invalid_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            MarketCapAdapter()(torch.ones(1, 1, 1, 2), torch.tensor([0]), torch.zeros(1, 1, 1))

    def test_tensor_metadata_count_is_checked(self):
        with self.assertRaisesRegex(ValueError, "batch size"):
            TwoStageTrainer._metadata_values(torch.tensor([1, 2]), 1, 0)
        self.assertEqual(TwoStageTrainer._metadata_values(torch.tensor([4]), 1, 0), [4])

    def test_scalar_metadata_is_not_split_into_characters(self):
        with self.assertRaisesRegex(ValueError, "scalar string"):
            TwoStageTrainer._metadata_values("2024-01-01", 10, 0)

    def test_no_attention_keeps_one_global_residual(self):
        config = TextEncoderConfig(
            fallback_vocab_size=10, fallback_hidden_size=2,
            lstm_hidden_size=1, industry_feature_dim=2,
            attention_key_dim=2, dropout=0.0,
        )
        ablation = replace(AblationConfig(), use_finbert=False, use_bilstm=False, use_industry_attention=False)
        encoder = FinFEBERTEncoder(1, config, ablation).eval()
        with torch.no_grad():
            encoder.context_encoder.weight.copy_(torch.eye(2))
            encoder.context_encoder.bias.zero_()
            encoder.pool_projection.weight.copy_(torch.eye(2))
            encoder.industry_weight.copy_(torch.eye(2).unsqueeze(0))
            encoder.industry_bias.zero_()
        ids = torch.ones(1, 1, 2, 2, dtype=torch.long)
        sentences = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with patch.object(encoder, "_encode_sentences", return_value=sentences):
            output = encoder(ids, ids)
        expected = torch.nn.functional.gelu(torch.tensor([[[[2.0, 3.0]]]]))
        torch.testing.assert_close(output.industry_features, expected)

    def test_empty_news_keeps_mapping_bias(self):
        config = TextEncoderConfig(
            fallback_vocab_size=10, fallback_hidden_size=2,
            lstm_hidden_size=1, industry_feature_dim=2,
            attention_key_dim=2, dropout=0.0,
        )
        encoder = FinFEBERTEncoder(2, config, AblationConfig.preset("no_finbert")).eval()
        with torch.no_grad():
            encoder.industry_bias.fill_(2.0)
            encoder.score_head.bias.fill_(3.0)
        ids = torch.zeros(1, 1, 2, 2, dtype=torch.long)
        output = encoder(ids, ids)
        expected = torch.nn.functional.gelu(encoder.industry_bias)[None, None]
        torch.testing.assert_close(output.industry_features, expected)
        torch.testing.assert_close(output.industry_scores, encoder.score_head(expected).squeeze(-1))
        self.assertEqual(output.news_attention.count_nonzero().item(), 0)

    def trainer(self):
        model = nn.Linear(1, 1)
        model.text_encoder = None
        return TwoStageTrainer(model, TrainingConfig(stage2_epochs=1), AblationConfig(), torch.device("cpu"))

    def test_empty_training_and_evaluation_are_rejected(self):
        trainer = self.trainer()
        with self.assertRaisesRegex(ValueError, "no batches"):
            trainer.fit_stage2([])
        with self.assertRaisesRegex(ValueError, "no batches"):
            trainer.evaluate_mcc([])

    def test_empty_export_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "predictions.csv"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no observations"):
                self.trainer().export_predictions([], output, split="test", seed=42, method="full")
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
