import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from astgtn.factors import FinancialFactorBuilder, FormulaNode, GeneticFactorMiner, ICSelector


class FactorReviewTests(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(5)
        values = rng.normal(size=(9, 5))
        return {"price": values, "volume": rng.normal(size=(9, 5))}, values

    def test_fcf_yield_uses_total_market_value(self):
        index = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=5), ["A"]],
                                          names=["date", "symbol"])
        prices = np.array([10., 12., 11., 14., 15.])
        frame = pd.DataFrame({"close": prices, "eps": 2., "free_cash_flow": 30.,
                              "market_cap": 600., "enterprise_value": 100., "ebitda": 10.},
                             index=index)
        result = FinancialFactorBuilder(window=4).build(frame)
        np.testing.assert_allclose(result.free_cash_flow_yield, 0.05)
        expected = np.log(prices[1:] / prices[:-1]).var() * 252
        self.assertAlmostEqual(result.annualized_variance.iloc[-1], expected)

    def test_ic_skips_unusable_factors(self):
        index = pd.MultiIndex.from_product([range(3), range(4)], names=["date", "symbol"])
        signal = np.tile(np.arange(4), 3)
        frame = pd.DataFrame({"good": signal, "constant": 1., "invalid": np.inf}, index=index)
        selector = ICSelector(3).fit(frame, pd.Series(signal, index=index))
        self.assertEqual(selector.selected_, ["good"])
        with self.assertRaises(ValueError):
            ICSelector(1).fit(frame[["constant"]], pd.Series(signal, index=index))

    def test_initialization_mixes_full_and_grow_at_each_depth(self):
        features, returns = self.inputs()
        miner = GeneticFactorMiner(population_size=8, elite_size=1, generations=0, max_depth=4)
        with patch.object(miner, "_random_tree", wraps=miner._random_tree) as generate:
            miner.mine(features, returns, top_k=1)
        root_calls = [call for call in generate.call_args_list if "full" in call.kwargs]
        observed = [(call.kwargs["depth"], call.kwargs["full"]) for call in root_calls]
        self.assertEqual(observed, [(d, full) for d in range(1, 5) for full in (False, True)])

    def test_frozen_formulas_and_serialization(self):
        features, returns = self.inputs()
        miner = GeneticFactorMiner(population_size=12, elite_size=2, generations=1, max_depth=2)
        miner.fit(features, returns, top_k=2)
        transformed = miner.transform(features)
        self.assertEqual(transformed.shape, (9, 5, len(miner.formulas_)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "factors.json"
            miner.save(path)
            restored = GeneticFactorMiner.load(path)
            np.testing.assert_array_equal(restored.transform(features), transformed)
            self.assertEqual([str(x) for x in restored.formulas_], [str(x) for x in miner.formulas_])

    def test_transform_preserves_rolling_history(self):
        features, _ = self.inputs()
        miner = GeneticFactorMiner(rolling_window=5)
        miner.formulas_ = (FormulaNode("ts_mean", (FormulaNode("feature", feature="price"),)),)
        miner.feature_names_, miner.num_stocks_ = ("price",), 5
        full = miner.transform(features)
        test = miner.transform({k: v[6:] for k, v in features.items()},
                               history={k: v[:6] for k, v in features.items()})
        np.testing.assert_array_equal(test, full[6:])

    def test_filter_uses_panel_correlation_and_can_be_ablated(self):
        values = np.arange(15, dtype=float).reshape(3, 5)
        miner = GeneticFactorMiner()
        self.assertTrue(miner._is_redundant(-values, [values]))
        self.assertFalse(GeneticFactorMiner(use_similarity_filter=False)._is_redundant(-values, [values]))
        self.assertTrue(miner._is_redundant(np.ones((3, 5)), []))

    def test_invalid_search_inputs_are_rejected(self):
        for options in [{"max_depth": 0}, {"elite_size": 101}, {"hoist_probability": 2},
                        {"similarity_threshold": 0}, {"parsimony_coefficient": -1}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                GeneticFactorMiner(**options)
        features, returns = self.inputs()
        features["price"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            GeneticFactorMiner().fit(features, returns)
        with self.assertRaises(ValueError):
            FormulaNode.from_dict({"op": "eval", "children": []})


if __name__ == "__main__":
    unittest.main()
