import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn

from astgtn.config import AblationConfig, TextEncoderConfig, TrainingConfig
from astgtn.factors import FinancialFactorBuilder, GeneticFactorMiner
from astgtn.news import NewsWindowPreprocessor
from astgtn.text_encoder import FinFEBERTEncoder
from astgtn.training import TwoStageTrainer


class RegressionTests(unittest.TestCase):
    def test_rank_ties(self):
        values = np.array([[2.0, 2.0, 5.0, 9.0]])
        ranked = GeneticFactorMiner()._apply_unary("rank", values)
        np.testing.assert_allclose(ranked, [[1 / 6, 1 / 6, 2 / 3, 1.0]])

    def test_drawdown_window(self):
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5), ["A"]],
            names=["date", "symbol"],
        )
        frame = pd.DataFrame(
            {"close": [100, 50, 60, 70, 80], "eps": 1.0,
             "free_cash_flow": 1.0, "market_cap": 10.0, "enterprise_value": 1.0, "ebitda": 1.0},
            index=index,
        )
        output = FinancialFactorBuilder(window=4).build(frame)
        self.assertAlmostEqual(output["max_drawdown"].iloc[3], -0.5)
        self.assertAlmostEqual(output["max_drawdown"].iloc[4], 0.0)

    def make_encoder(self):
        torch.manual_seed(5)
        config = TextEncoderConfig(
            fallback_vocab_size=20, fallback_hidden_size=8,
            lstm_hidden_size=4, industry_feature_dim=8,
            attention_key_dim=4, dropout=0.0,
        )
        ablation = replace(AblationConfig(), use_finbert=False)
        return FinFEBERTEncoder(2, config, ablation).eval()

    def test_padding_invariance(self):
        encoder = self.make_encoder()
        ids = torch.tensor([[[[1, 2, 0], [3, 4, 0]]]])
        attention = ids.ne(0).long()
        real = encoder(ids, attention)
        padded_ids = torch.cat([ids, torch.zeros(1, 1, 3, 3, dtype=torch.long)], dim=2)
        padded = encoder(padded_ids, padded_ids.ne(0).long())
        torch.testing.assert_close(real.industry_features, padded.industry_features)
        torch.testing.assert_close(real.news_attention, padded.news_attention[..., :2])
        gapped_ids = padded_ids[:, :, [0, 2, 1, 3, 4]]
        gapped = encoder(gapped_ids, gapped_ids.ne(0).long())
        torch.testing.assert_close(real.industry_features, gapped.industry_features)

    def test_empty_news(self):
        encoder = self.make_encoder()
        with torch.no_grad():
            encoder.industry_bias.fill_(2.0)
            encoder.score_head.bias.fill_(3.0)
        ids = torch.zeros(1, 1, 4, 3, dtype=torch.long)
        output = encoder(ids, ids)
        expected = torch.nn.functional.gelu(encoder.industry_bias).unsqueeze(0).unsqueeze(0)
        torch.testing.assert_close(output.industry_features, expected)
        torch.testing.assert_close(output.industry_scores, encoder.score_head(expected).squeeze(-1))
        self.assertEqual(output.news_attention.count_nonzero().item(), 0)

    def test_first_checkpoint(self):
        class CheckpointModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.text_encoder = None

            def encode_features(self, quantitative):
                return SimpleNamespace(features=quantitative)

            def backbone(self, features, window_cutoffs=None):
                value = features * self.weight
                return SimpleNamespace(logits=torch.cat([value, -value], dim=-1))

        model = CheckpointModel()
        trainer = TwoStageTrainer(
            model, TrainingConfig(stage2_epochs=2),
            AblationConfig.preset("no_mixup"), torch.device("cpu")
        )
        trainer.evaluate_mcc = lambda _: -1.0
        batch = {"quantitative": torch.ones(1, 1, 1, 1), "targets": torch.zeros(1, 1, 1, dtype=torch.long)}
        history = trainer.fit_stage2([batch], [batch])
        self.assertEqual(history.best_epoch, 0)
        self.assertEqual(history.best_mcc, -1.0)

    def test_us_news_window(self):
        news = pd.DataFrame({
            "published_at": ["2024-03-08T20:30:00Z", "2024-03-08T21:30:00Z", "2024-03-11T13:00:00Z"],
            "title": ["Before Friday close", "After Friday close", "Before Monday open"],
        })
        windows = NewsWindowPreprocessor().transform(news, ["2024-03-08", "2024-03-11"])
        self.assertEqual(windows[pd.Timestamp("2024-03-11")], ["After Friday close", "Before Monday open"])


if __name__ == "__main__":
    unittest.main()
