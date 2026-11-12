from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from astgtn.factors import FinancialFactorBuilder, GeneticFactorMiner, ICSelector


class FactorTests(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(7)

    def test_ic_selector_and_gp_factor_miner(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6)
        symbols = ["A", "B", "C", "D"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        signal = np.tile(np.arange(4, dtype=float), 6)
        factors = pd.DataFrame(
            {"signal": signal, "noise": np.random.randn(len(index))}, index=index
        )
        returns = pd.Series(signal + np.random.randn(len(index)) * 0.01, index=index)
        selector = ICSelector(top_k=1).fit(factors, returns)
        self.assertEqual(selector.selected_, ["signal"])

        feature_arrays = {
            "price": np.random.randn(8, 5),
            "volume": np.random.randn(8, 5),
        }
        future = feature_arrays["price"] + np.random.randn(8, 5) * 0.01
        miner = GeneticFactorMiner(population_size=10, generations=1, max_depth=2, seed=3)
        results = miner.mine(feature_arrays, future, top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].values.shape, future.shape)

    def test_financial_and_risk_factor_builder(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10)
        symbols = ["A", "B"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        count = len(index)
        frame = pd.DataFrame(
            {
                "close": np.linspace(10, 20, count),
                "eps": np.full(count, 2.0),
                "free_cash_flow": np.linspace(1, 2, count),
                "market_cap": np.linspace(100, 200, count),
                "enterprise_value": np.linspace(100, 120, count),
                "ebitda": np.full(count, 10.0),
            },
            index=index,
        )
        factors = FinancialFactorBuilder(window=4).build(frame)
        expected = {
            "pe",
            "free_cash_flow_yield",
            "ev_to_ebitda",
            "sharpe",
            "max_drawdown",
            "kurtosis",
            "annualized_variance",
        }
        self.assertEqual(set(factors.columns), expected)
        self.assertTrue(np.isfinite(factors["pe"]).all())


if __name__ == "__main__":
    unittest.main()
