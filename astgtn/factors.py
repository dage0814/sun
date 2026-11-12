from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class FinancialFactorBuilder:
    """Input index: (date, symbol). Accounting column names are configurable; missing fields raise an error."""

    def __init__(self, window: int = 20, annualization: int = 252) -> None:
        self.window = window
        self.annualization = annualization

    def build(
        self,
        frame: pd.DataFrame,
        *,
        price_col: str = "close",
        eps_col: str = "eps",
        free_cash_flow_col: str = "free_cash_flow",
        enterprise_value_col: str = "enterprise_value",
        ebitda_col: str = "ebitda",
        return_col: str = "return",
        symbol_level: str = "symbol",
    ) -> pd.DataFrame:
        if not isinstance(frame.index, pd.MultiIndex):
            raise ValueError("factor frame must use a (date, symbol) MultiIndex")
        required = {
            price_col,
            eps_col,
            free_cash_flow_col,
            enterprise_value_col,
            ebitda_col,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"Missing factor columns: {missing}")

        data = frame.sort_index().copy()
        if return_col not in data:
            data[return_col] = data.groupby(level=symbol_level)[price_col].pct_change(fill_method=None)
        grouped_return = data.groupby(level=symbol_level, group_keys=False)[return_col]
        grouped_price = data.groupby(level=symbol_level, group_keys=False)[price_col]

        output = pd.DataFrame(index=data.index)
        output["pe"] = self._safe_divide(data[price_col], data[eps_col])
        output["free_cash_flow_yield"] = self._safe_divide(
            data[free_cash_flow_col], data[price_col]
        )
        output["ev_to_ebitda"] = self._safe_divide(
            data[enterprise_value_col], data[ebitda_col]
        )

        rolling_mean = grouped_return.transform(
            lambda s: s.rolling(self.window, min_periods=max(3, self.window // 4)).mean()
        )
        rolling_std = grouped_return.transform(
            lambda s: s.rolling(self.window, min_periods=max(3, self.window // 4)).std(ddof=0)
        )
        output["sharpe"] = self._safe_divide(rolling_mean, rolling_std) * math.sqrt(
            self.annualization
        )
        # Peak and trough must lie in the same window.
        output["max_drawdown"] = grouped_price.transform(
            lambda s: s.rolling(self.window, min_periods=2).apply(
                self._window_drawdown, raw=True
            )
        )
        output["kurtosis"] = grouped_return.transform(
            lambda s: s.rolling(self.window, min_periods=4).kurt()
        )
        output["annualized_variance"] = grouped_return.transform(
            lambda s: s.rolling(self.window, min_periods=3).var(ddof=0)
        ) * self.annualization
        return output.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _window_drawdown(prices: np.ndarray) -> float:
        values = prices[np.isfinite(prices)]
        if len(values) < 2 or (values <= 0).any():
            return np.nan
        return float(np.min(values / np.maximum.accumulate(values) - 1.0))

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        safe_denominator = denominator.where(denominator.abs() > 1e-12)
        return numerator / safe_denominator


class ICSelector:

    def __init__(self, top_k: int, min_observations: int = 3, use_absolute_ic: bool = True) -> None:
        self.top_k = top_k
        self.min_observations = min_observations
        self.use_absolute_ic = use_absolute_ic
        self.ic_: Optional[pd.Series] = None
        self.selected_: List[str] = []

    def fit(
        self,
        factors: pd.DataFrame,
        future_returns: pd.Series,
        *,
        date_level: str = "date",
    ) -> "ICSelector":
        aligned = factors.join(future_returns.rename("__future_return__"), how="inner")
        scores: Dict[str, float] = {}
        for column in factors.columns:
            daily = aligned.groupby(level=date_level).apply(
                lambda part: self._corr_if_valid(
                    part[column], part["__future_return__"], self.min_observations
                ),
                include_groups=False,
            )
            scores[column] = float(daily.mean(skipna=True))
        self.ic_ = pd.Series(scores, dtype=float).sort_values(
            key=lambda s: s.abs() if self.use_absolute_ic else s,
            ascending=False,
        )
        self.selected_ = self.ic_.head(self.top_k).index.tolist()
        return self

    def transform(self, factors: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_:
            raise RuntimeError("ICSelector.fit must be called before transform")
        return factors.loc[:, self.selected_].copy()

    def fit_transform(self, factors: pd.DataFrame, future_returns: pd.Series, **kwargs) -> pd.DataFrame:
        return self.fit(factors, future_returns, **kwargs).transform(factors)

    @staticmethod
    def _corr_if_valid(left: pd.Series, right: pd.Series, minimum: int) -> float:
        valid = left.notna() & right.notna()
        if int(valid.sum()) < minimum:
            return np.nan
        if left[valid].nunique() < 2 or right[valid].nunique() < 2:
            return np.nan
        return float(left[valid].corr(right[valid]))


@dataclass(frozen=True)
class FormulaNode:
    op: str
    children: Tuple["FormulaNode", ...] = ()
    feature: Optional[str] = None
    constant: Optional[float] = None

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    def __str__(self) -> str:
        if self.op == "feature":
            return str(self.feature)
        if self.op == "constant":
            return f"{self.constant:.4g}"
        if len(self.children) == 1:
            return f"{self.op}({self.children[0]})"
        return f"({self.children[0]} {self.op} {self.children[1]})"


@dataclass(frozen=True)
class FactorResult:
    formula: FormulaNode
    fitness: float
    mean_ic: float
    values: np.ndarray


class GeneticFactorMiner:
    """Input shape: [T, N]. Fitness averages daily cross-sectional IC."""

    BINARY_OPS: Mapping[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: np.clip(a * b, -1e6, 1e6),
        "/": lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-8),
    }
    UNARY_OPS = ("rank", "zscore", "abs", "logabs", "sqrtabs", "ts_mean", "ts_std")

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 20,
        crossover_probability: float = 0.7,
        mutation_probability: float = 0.2,
        similarity_threshold: float = 0.7,
        parsimony_coefficient: float = 1e-3,
        elite_size: int = 5,
        tournament_size: int = 4,
        max_depth: int = 4,
        rolling_window: int = 5,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.similarity_threshold = similarity_threshold
        self.parsimony_coefficient = parsimony_coefficient
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.rolling_window = rolling_window
        self.rng = random.Random(seed)

    def mine(
        self,
        features: Mapping[str, np.ndarray],
        future_returns: np.ndarray,
        top_k: int = 20,
    ) -> List[FactorResult]:
        self._validate_inputs(features, future_returns)
        feature_names = tuple(features)
        population = [
            self._random_tree(feature_names, depth=1 + i % self.max_depth, full=bool(i % 2))
            for i in range(self.population_size)
        ]
        evaluated = self._evaluate_population(population, features, future_returns)
        for _ in range(self.generations):
            evaluated.sort(key=lambda item: item.fitness, reverse=True)
            next_population = [item.formula for item in evaluated[: self.elite_size]]
            accepted_values = [item.values for item in evaluated[: self.elite_size]]
            attempts = 0
            while len(next_population) < self.population_size and attempts < self.population_size * 20:
                attempts += 1
                parent = self._tournament(evaluated).formula
                child = parent
                if self.rng.random() < self.crossover_probability:
                    other = self._tournament(evaluated).formula
                    child = self._crossover(parent, other)
                if self.rng.random() < self.mutation_probability:
                    child = self._mutate(child, feature_names)
                values = self.evaluate(child, features)
                if self._is_redundant(values, accepted_values):
                    continue
                next_population.append(child)
                accepted_values.append(values)
            while len(next_population) < self.population_size:
                next_population.append(self._random_tree(feature_names, self.max_depth, full=False))
            evaluated = self._evaluate_population(next_population, features, future_returns)

        evaluated.sort(key=lambda item: item.fitness, reverse=True)
        selected: List[FactorResult] = []
        for candidate in evaluated:
            if not self._is_redundant(candidate.values, [item.values for item in selected]):
                selected.append(candidate)
            if len(selected) >= top_k:
                break
        return selected

    def evaluate(self, node: FormulaNode, features: Mapping[str, np.ndarray]) -> np.ndarray:
        if node.op == "feature":
            return np.asarray(features[str(node.feature)], dtype=float)
        if node.op == "constant":
            shape = next(iter(features.values())).shape
            return np.full(shape, float(node.constant), dtype=float)
        values = [self.evaluate(child, features) for child in node.children]
        if node.op in self.BINARY_OPS:
            result = self.BINARY_OPS[node.op](values[0], values[1])
        else:
            result = self._apply_unary(node.op, values[0])
        return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

    def _apply_unary(self, op: str, value: np.ndarray) -> np.ndarray:
        if op == "abs":
            return np.abs(value)
        if op == "logabs":
            return np.log1p(np.abs(value))
        if op == "sqrtabs":
            return np.sqrt(np.abs(value))
        if op == "rank":
            ranks = pd.DataFrame(value).rank(axis=1, method="average").to_numpy()
            return (ranks - 1.0) / max(value.shape[1] - 1, 1)
        if op == "zscore":
            mean = value.mean(axis=1, keepdims=True)
            std = value.std(axis=1, keepdims=True)
            return (value - mean) / np.maximum(std, 1e-8)
        if op in {"ts_mean", "ts_std"}:
            frame = pd.DataFrame(value)
            rolling = frame.rolling(self.rolling_window, min_periods=1)
            return (rolling.mean() if op == "ts_mean" else rolling.std(ddof=0)).to_numpy()
        raise ValueError(f"Unknown operator: {op}")

    def _evaluate_population(
        self,
        population: Iterable[FormulaNode],
        features: Mapping[str, np.ndarray],
        future_returns: np.ndarray,
    ) -> List[FactorResult]:
        results = []
        for formula in population:
            values = self.evaluate(formula, features)
            mean_ic = self._mean_cross_sectional_ic(values, future_returns)
            fitness = abs(mean_ic) - self.parsimony_coefficient * formula.size
            results.append(FactorResult(formula, fitness, mean_ic, values))
        return results

    def _random_tree(self, features: Sequence[str], depth: int, full: bool) -> FormulaNode:
        if depth <= 0 or (not full and self.rng.random() < 0.3):
            if self.rng.random() < 0.9:
                return FormulaNode("feature", feature=self.rng.choice(features))
            return FormulaNode("constant", constant=self.rng.uniform(-2.0, 2.0))
        if self.rng.random() < 0.55:
            op = self.rng.choice(tuple(self.BINARY_OPS))
            return FormulaNode(
                op,
                (
                    self._random_tree(features, depth - 1, full),
                    self._random_tree(features, depth - 1, full),
                ),
            )
        op = self.rng.choice(self.UNARY_OPS)
        return FormulaNode(op, (self._random_tree(features, depth - 1, full),))

    def _tournament(self, population: Sequence[FactorResult]) -> FactorResult:
        candidates = self.rng.sample(list(population), min(self.tournament_size, len(population)))
        return max(candidates, key=lambda item: item.fitness)

    def _crossover(self, left: FormulaNode, right: FormulaNode) -> FormulaNode:
        left_paths = self._paths(left)
        right_paths = self._paths(right)
        left_path = self.rng.choice(left_paths)
        right_subtree = self._subtree(right, self.rng.choice(right_paths))
        return self._replace(left, left_path, right_subtree)

    def _mutate(self, node: FormulaNode, features: Sequence[str]) -> FormulaNode:
        paths = self._paths(node)
        target = self.rng.choice(paths)
        if self.rng.random() < 0.25 and target:
            subtree = self._subtree(node, target)
            descendants = self._paths(subtree)
            replacement = self._subtree(subtree, self.rng.choice(descendants))
        else:
            replacement = self._random_tree(features, self.rng.randint(1, self.max_depth), False)
        return self._replace(node, target, replacement)

    @staticmethod
    def _paths(node: FormulaNode, prefix: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
        paths = [prefix]
        for index, child in enumerate(node.children):
            paths.extend(GeneticFactorMiner._paths(child, prefix + (index,)))
        return paths

    @staticmethod
    def _subtree(node: FormulaNode, path: Tuple[int, ...]) -> FormulaNode:
        current = node
        for index in path:
            current = current.children[index]
        return current

    @staticmethod
    def _replace(node: FormulaNode, path: Tuple[int, ...], replacement: FormulaNode) -> FormulaNode:
        if not path:
            return replacement
        index = path[0]
        children = list(node.children)
        children[index] = GeneticFactorMiner._replace(children[index], path[1:], replacement)
        return FormulaNode(node.op, tuple(children), node.feature, node.constant)

    def _is_redundant(self, values: np.ndarray, accepted: Sequence[np.ndarray]) -> bool:
        flat = values.reshape(-1)
        if np.std(flat) < 1e-10:
            return True
        for other in accepted:
            other_flat = other.reshape(-1)
            valid = np.isfinite(flat) & np.isfinite(other_flat)
            if valid.sum() < 3 or np.std(other_flat[valid]) < 1e-10:
                continue
            similarity = abs(float(np.corrcoef(flat[valid], other_flat[valid])[0, 1]))
            if similarity >= self.similarity_threshold:
                return True
        return False

    @staticmethod
    def _mean_cross_sectional_ic(values: np.ndarray, returns: np.ndarray) -> float:
        correlations = []
        for date_index in range(values.shape[0]):
            left, right = values[date_index], returns[date_index]
            valid = np.isfinite(left) & np.isfinite(right)
            if valid.sum() < 3 or np.std(left[valid]) < 1e-10 or np.std(right[valid]) < 1e-10:
                continue
            correlations.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
        return float(np.mean(correlations)) if correlations else 0.0

    @staticmethod
    def _validate_inputs(features: Mapping[str, np.ndarray], future_returns: np.ndarray) -> None:
        if not features:
            raise ValueError("features cannot be empty")
        shape = np.asarray(future_returns).shape
        if len(shape) != 2:
            raise ValueError("future_returns must use shape [T,N]")
        for name, values in features.items():
            if np.asarray(values).shape != shape:
                raise ValueError(f"Feature {name!r} has shape {np.asarray(values).shape}, expected {shape}")
