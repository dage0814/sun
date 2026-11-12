from __future__ import annotations

import unittest
from datetime import time

import torch
import pandas as pd

from astgtn.activations import sgfrelu
from astgtn.ablation import DEFAULT_ABLATIONS
from astgtn.config import AblationConfig, ModelConfig, TextEncoderConfig, TrainingConfig
from astgtn.data import MultimodalWindowDataset, chronological_split
from astgtn.fusion import DualPathCrossModalFusion
from astgtn.losses import CombinedStockObjective
from astgtn.model import AdaptiveSpatioTemporalGraphTransformer
from astgtn.news import NewsWindowPreprocessor, tokenize_news_windows
from astgtn.pipeline import MultiModalASTGTN
from astgtn.revin import RevIN
from astgtn.text_encoder import EmbeddingTextBackbone, FinFEBERTEncoder, ListMLELoss
from astgtn.topology import PriorGraphBuilder
from astgtn.training import TwoStageTrainer


class ThesisModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_ablation_presets_are_single_source_of_truth(self) -> None:
        config = AblationConfig.preset("no_text")
        self.assertFalse(config.use_text)
        self.assertFalse(config.use_cross_attention_ca3)
        self.assertTrue(AblationConfig.preset("full").use_adaptive_graph)

    def test_sgfrelu_has_hard_zero_region(self) -> None:
        values = sgfrelu(torch.tensor([-2.0, 0.0, 0.75, 2.0]))
        self.assertTrue(torch.equal(values[:3], torch.zeros(3)))
        self.assertGreater(float(values[-1]), 0.0)

    def test_revin_is_reversible_per_stock(self) -> None:
        values = torch.randn(2, 6, 4, 3)
        layer = RevIN(3)
        normalized = layer(values, "norm")
        reconstructed = layer(normalized, "denorm")
        self.assertTrue(torch.allclose(values, reconstructed, atol=2e-4))
        self.assertTrue(torch.allclose(normalized.mean(dim=1), torch.zeros(2, 4, 3), atol=1e-5))

    def test_text_encoder_and_listmle(self) -> None:
        ablation = AblationConfig.preset("no_finbert")
        config = TextEncoderConfig(
            fallback_vocab_size=50,
            fallback_hidden_size=8,
            lstm_hidden_size=4,
            industry_feature_dim=8,
            attention_key_dim=4,
        )
        encoder = FinFEBERTEncoder(3, config, ablation)
        input_ids = torch.randint(1, 50, (2, 3, 4, 5))
        attention = torch.ones_like(input_ids)
        news_mask = torch.ones(2, 3, 4, dtype=torch.bool)
        output = encoder(input_ids, attention, news_mask)
        self.assertEqual(output.industry_features.shape, (2, 3, 3, 8))
        relevance = torch.randn(2, 3, 3)
        loss = ListMLELoss()(output.industry_scores, relevance)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_cross_modal_fusion(self) -> None:
        ablation = AblationConfig.preset("no_finbert")
        fusion = DualPathCrossModalFusion(3, 8, model_dim=8, num_heads=2, ablation=ablation)
        quantitative = torch.randn(2, 5, 4, 3)
        industry_news = torch.randn(2, 5, 2, 8, requires_grad=True)
        stock_industry = torch.tensor([0, 0, 1, 1])
        caps = torch.rand(2, 5, 4) * 100 + 1
        output = fusion(quantitative, industry_news, stock_industry, caps)
        self.assertEqual(output.combined.shape, (2, 5, 4, 8))
        self.assertTrue(torch.isfinite(output.combined).all())
        output.combined.square().mean().backward()
        self.assertIsNotNone(industry_news.grad)
        self.assertGreater(float(industry_news.grad.abs().sum()), 0.0)

    def test_news_window_preprocessing_and_token_batch(self) -> None:
        news = pd.DataFrame(
            {
                "published_at": [
                    "2024-01-02 16:00:00",
                    "2024-01-03 08:00:00",
                    "2024-01-03 10:00:00",
                ],
                "title": ["政策发布", "行业解读", "开盘后的未来信息"],
            }
        )
        processor = NewsWindowPreprocessor(
            close_time=time(15, 0), open_time=time(9, 30), max_news_per_day=4,
            source_timezone="America/New_York",
        )
        windows = processor.transform(
            news, pd.to_datetime(["2024-01-02", "2024-01-03"])
        )
        titles = windows[pd.Timestamp("2024-01-03")]
        self.assertEqual(titles, ["政策发布", "行业解读"])

        class TinyTokenizer:
            def __call__(self, rows, **kwargs):
                length = kwargs["max_length"]
                ids = torch.ones(len(rows), length, dtype=torch.long)
                return {"input_ids": ids, "attention_mask": ids.clone()}

        ids, attention, mask = tokenize_news_windows(
            [titles], TinyTokenizer(), max_news=4, max_length=6
        )
        self.assertEqual(ids.shape, (1, 4, 6))
        self.assertEqual(int(mask.sum()), 2)

    def test_spatio_temporal_model_forward_and_gradient(self) -> None:
        ablation = AblationConfig()
        config = self._small_model_config(num_input_features=5)
        prior = torch.eye(4)
        model = AdaptiveSpatioTemporalGraphTransformer(config, ablation, prior)
        output = model(torch.randn(2, 6, 4, 5))
        self.assertEqual(output.logits.shape, (2, 1, 4, 2))
        self.assertEqual(output.adjacency.shape, (2, 6, 4, 4))
        objective = CombinedStockObjective(TrainingConfig(), ablation)
        targets = torch.randint(0, 2, (2, 1, 4))
        loss = objective(output.logits, targets).total
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None), 0)

    def test_multimodal_pipeline_and_chronological_split(self) -> None:
        ablation = AblationConfig.preset("no_finbert")
        text_config = TextEncoderConfig(
            fallback_vocab_size=40,
            fallback_hidden_size=8,
            lstm_hidden_size=4,
            industry_feature_dim=8,
            attention_key_dim=4,
        )
        model_config = self._small_model_config(num_input_features=10)
        prior = torch.eye(4)
        model = MultiModalASTGTN(
            model_config=model_config,
            text_config=text_config,
            ablation=ablation,
            num_industries=2,
            num_quant_features=3,
            num_technical_features=2,
            stock_industry=torch.tensor([0, 0, 1, 1]),
            prior_adjacency=prior,
            fusion_dim=8,
            fusion_heads=2,
        )
        batch = {
            "quantitative": torch.randn(2, 6, 4, 3),
            "technical": torch.randn(2, 6, 4, 2),
            "market_caps": torch.rand(2, 6, 4) + 1,
            "input_ids": torch.randint(1, 40, (2, 6, 3, 5)),
            "attention_mask": torch.ones(2, 6, 3, 5, dtype=torch.long),
            "news_mask": torch.ones(2, 6, 3, dtype=torch.bool),
        }
        output = model(**batch)
        self.assertEqual(output.prediction.logits.shape, (2, 1, 4, 2))

        dataset = MultimodalWindowDataset(
            batch["quantitative"].repeat(10, 1, 1, 1),
            batch["technical"].repeat(10, 1, 1, 1),
            batch["market_caps"].repeat(10, 1, 1),
            torch.randint(0, 2, (20, 1, 4)),
        )
        train, validation, test = chronological_split(dataset, gap=1)
        self.assertLess(len(train) + len(validation) + len(test), len(dataset))
        self.assertLess(max(train.indices), min(test.indices))

    def test_two_stage_trainer_runs_both_stages(self) -> None:
        ablation = AblationConfig.preset("no_finbert")
        text_config = TextEncoderConfig(
            fallback_vocab_size=40,
            fallback_hidden_size=8,
            lstm_hidden_size=4,
            industry_feature_dim=8,
            attention_key_dim=4,
        )
        model = MultiModalASTGTN(
            model_config=self._small_model_config(num_input_features=10),
            text_config=text_config,
            ablation=ablation,
            num_industries=2,
            num_quant_features=3,
            num_technical_features=2,
            stock_industry=torch.tensor([0, 0, 1, 1]),
            prior_adjacency=torch.eye(4),
            fusion_dim=8,
            fusion_heads=2,
        )
        batch = {
            "quantitative": torch.randn(2, 6, 4, 3),
            "technical": torch.randn(2, 6, 4, 2),
            "market_caps": torch.rand(2, 6, 4) + 1,
            "input_ids": torch.randint(1, 40, (2, 6, 3, 5)),
            "attention_mask": torch.ones(2, 6, 3, 5, dtype=torch.long),
            "news_mask": torch.ones(2, 6, 3, dtype=torch.bool),
            "industry_relevance": torch.randn(2, 6, 2),
            "targets": torch.randint(0, 2, (2, 1, 4)),
        }
        training = TrainingConfig(stage1_epochs=1, stage2_epochs=1)
        trainer = TwoStageTrainer(model, training, ablation, torch.device("cpu"))
        stage1 = trainer.fit_stage1([batch])
        stage2 = trainer.fit_stage2([batch])
        self.assertEqual(len(stage1), 1)
        self.assertEqual(len(stage2.train_loss), 1)

    def test_all_standard_ablation_presets_forward(self) -> None:
        text_config = TextEncoderConfig(
            fallback_vocab_size=40,
            fallback_hidden_size=8,
            lstm_hidden_size=4,
            industry_feature_dim=8,
            attention_key_dim=4,
        )
        inputs = {
            "quantitative": torch.randn(1, 4, 4, 3),
            "technical": torch.randn(1, 4, 4, 2),
            "market_caps": torch.rand(1, 4, 4) + 1,
            "input_ids": torch.randint(1, 40, (1, 4, 2, 5)),
            "attention_mask": torch.ones(1, 4, 2, 5, dtype=torch.long),
            "news_mask": torch.ones(1, 4, 2, dtype=torch.bool),
        }
        for name in DEFAULT_ABLATIONS:
            with self.subTest(ablation=name):
                ablation = AblationConfig.preset(name)
                model = MultiModalASTGTN(
                    model_config=self._small_model_config(num_input_features=10),
                    text_config=text_config,
                    ablation=ablation,
                    num_industries=2,
                    num_quant_features=3,
                    num_technical_features=2,
                    stock_industry=torch.tensor([0, 0, 1, 1]),
                    prior_adjacency=torch.eye(4),
                    fusion_dim=8,
                    fusion_heads=2,
                    text_backbone=EmbeddingTextBackbone(40, 8),
                )
                model_inputs = inputs if ablation.use_text else {
                    key: value
                    for key, value in inputs.items()
                    if key in {"quantitative", "technical", "market_caps"}
                }
                output = model(**model_inputs)
                self.assertTrue(torch.isfinite(output.prediction.logits).all())

    @staticmethod
    def _small_model_config(num_input_features: int) -> ModelConfig:
        return ModelConfig(
            num_stocks=4,
            num_input_features=num_input_features,
            hidden_dim=16,
            graph_heads=4,
            graph_layers=1,
            graph_top_k=3,
            node_embedding_dim=4,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ff_dim=32,
            tcn_channels=16,
            tcn_layers=2,
            tcn_kernel_size=3,
            dropout=0.0,
        )


if __name__ == "__main__":
    unittest.main()
