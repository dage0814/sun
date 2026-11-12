from __future__ import annotations

import math
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class FinancialFactorBuilder:


    def __init__(self, window: int = 20, annualization: int = 252) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 4:
            raise ValueError("window must be an integer of at least 4")
        if not math.isfinite(annualization) or annualization <= 0:
            raise ValueError("annualization must be positive")
        self.window = window
        self.annualization = annualization

    def build(
        self,
        frame: pd.DataFrame,
        *,
        price_col: str = "close",
        eps_col: str = "eps",
        free_cash_flow_col: str = "free_cash_flow",
        market_cap_col: str = "market_cap",
        enterprise_value_col: str = "enterprise_value",
        ebitda_col: str = "ebitda",
        return_col: str = "return",
        symbol_level: str = "symbol",
    ) -> pd.DataFrame:
        if not isinstance(frame.index, pd.MultiIndex):
            raise ValueError("factor frame must use a (date, symbol) MultiIndex")
        if frame.index.nlevels != 2 or symbol_level not in frame.index.names or not frame.index.is_unique:
            raise ValueError("factor rows must have unique date and symbol indices")
        required = {
            price_col,
            eps_col,
            free_cash_flow_col,
            market_cap_col,
            enterprise_value_col,
            ebitda_col,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"Missing factor columns: {missing}")

        data = frame.sort_index().copy()
        for column in required | ({return_col} if return_col in data else set()):
            data[column] = pd.to_numeric(data[column], errors="raise")
            if np.isinf(data[column]).any():
                raise ValueError(f"Factor input {column!r} contains infinite values")
        if (data[price_col].dropna() <= 0).any() or (data[market_cap_col].dropna() <= 0).any():
            raise ValueError("Prices and market capitalization must be positive")
        if return_col not in data:
            previous = data.groupby(level=symbol_level)[price_col].shift(1)
            data[return_col] = np.log(data[price_col] / previous)
        grouped_return = data.groupby(level=symbol_level, group_keys=False)[return_col]
        grouped_price = data.groupby(level=symbol_level, group_keys=False)[price_col]

        output = pd.DataFrame(index=data.index)
        output["pe"] = self._safe_divide(data[price_col], data[eps_col])
        output["free_cash_flow_yield"] = self._safe_divide(
            data[free_cash_flow_col], data[market_cap_col]
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
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(min_observations, int) or min_observations < 3:
            raise ValueError("min_observations must be at least 3")
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
        if not factors.index.is_unique or not future_returns.index.is_unique:
            raise ValueError("Factors and targets must have unique sample indices")
        if not factors.columns.is_unique or "__future_return__" in factors.columns:
            raise ValueError("Factor column names must be unique and not use __future_return__")
        if date_level not in factors.index.names:
            raise ValueError(f"Factors must have a {date_level!r} index level")
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
            kind="stable",
        )
        self.selected_ = self.ic_.dropna().head(self.top_k).index.tolist()
        if not self.selected_:
            raise ValueError("No factor has a valid training-period cross-sectional IC")
        return self

    def transform(self, factors: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_:
            raise RuntimeError("ICSelector.fit must be called before transform")
        return factors.loc[:, self.selected_].copy()

    def fit_transform(self, factors: pd.DataFrame, future_returns: pd.Series, **kwargs) -> pd.DataFrame:
        return self.fit(factors, future_returns, **kwargs).transform(factors)

    @staticmethod
    def _corr_if_valid(left: pd.Series, right: pd.Series, minimum: int) -> float:
        valid = np.isfinite(left) & np.isfinite(right)
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

    def to_dict(self) -> dict:
        return {"op": self.op, "children": [child.to_dict() for child in self.children],
                "feature": self.feature, "constant": self.constant}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FormulaNode":
        op = value["op"]
        children = tuple(cls.from_dict(child) for child in value.get("children", []))
        feature, constant = value.get("feature"), value.get("constant")
        if op == "feature":
            if children or not isinstance(feature, str) or not feature or constant is not None:
                raise ValueError("Invalid feature node")
        elif op == "constant":
            if children or feature is not None or isinstance(constant, bool) or not isinstance(constant, (int, float)) or not math.isfinite(constant):
                raise ValueError("Invalid constant node")
        elif op in GeneticFactorMiner.BINARY_OPS:
            if len(children) != 2 or feature is not None or constant is not None:
                raise ValueError("Invalid binary node")
        elif op in GeneticFactorMiner.UNARY_OPS:
            if len(children) != 1 or feature is not None or constant is not None:
                raise ValueError("Invalid unary node")
        else:
            raise ValueError(f"Unknown formula operator: {op}")
        return cls(op, children, feature, constant)

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
        hoist_probability: float = 0.25,
        use_similarity_filter: bool = True,
    ) -> None:
        for name, value in {"population_size": population_size, "tournament_size": tournament_size,
                            "max_depth": max_depth, "rolling_window": rolling_window}.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(generations, int) or isinstance(generations, bool) or generations < 0:
            raise ValueError("generations must be a nonnegative integer")
        if not isinstance(elite_size, int) or isinstance(elite_size, bool) or not 0 <= elite_size <= population_size:
            raise ValueError("elite_size must be between 0 and population_size")
        for name, value in {"crossover_probability": crossover_probability,
                            "mutation_probability": mutation_probability,
                            "hoist_probability": hoist_probability}.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not math.isfinite(similarity_threshold) or not 0 < similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if not math.isfinite(parsimony_coefficient) or parsimony_coefficient < 0:
            raise ValueError("parsimony_coefficient must be nonnegative")
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
        self.hoist_probability = hoist_probability
        self.use_similarity_filter = use_similarity_filter
        self.rng = random.Random(seed)
        self.formulas_: Tuple[FormulaNode, ...] = ()
        self.feature_names_: Tuple[str, ...] = ()
        self.num_stocks_: Optional[int] = None
        self.results_: List[FactorResult] = []

    def fit(
        self, features: Mapping[str, np.ndarray], future_returns: np.ndarray, top_k: int = 20,
    ) -> "GeneticFactorMiner":
        results = self.mine(features, future_returns, top_k)
        if not results:
            raise ValueError("No nonconstant factors survived selection")
        self.results_ = results
        self.formulas_ = tuple(item.formula for item in results)
        self.feature_names_ = tuple(features)
        self.num_stocks_ = np.asarray(future_returns).shape[1]
        return self

    def transform(
        self, features: Mapping[str, np.ndarray], *, history: Optional[Mapping[str, np.ndarray]] = None,
    ) -> np.ndarray:

        if not self.formulas_:
            raise RuntimeError("Fit or load a formula set before transform")
        self._validate_features(features)
        missing = set(self.feature_names_).difference(features)
        if missing:
            raise KeyError(f"Missing frozen factor inputs: {sorted(missing)}")
        current = {name: np.asarray(features[name], dtype=float) for name in self.feature_names_}
        rows, stocks = next(iter(current.values())).shape
        if stocks != self.num_stocks_:
            raise ValueError("Stock count must match the fitted cross-sectional universe")
        prefix = 0
        if history is not None:
            self._validate_features(history)
            if set(self.feature_names_).difference(history):
                raise KeyError("History is missing a fitted feature")
            prefix, history_stocks = np.asarray(history[self.feature_names_[0]]).shape
            if history_stocks != stocks:
                raise ValueError("History and current inputs must use the same stock order and count")
            current = {name: np.concatenate([np.asarray(history[name], dtype=float), values], axis=0)
                       for name, values in current.items()}
        return np.stack([self.evaluate(formula, current)[prefix:prefix + rows]
                         for formula in self.formulas_], axis=-1)

    def save(self, path: str | Path) -> None:
        if not self.formulas_:
            raise RuntimeError("Fit a formula set before save")
        payload = {"version": 1, "rolling_window": self.rolling_window,
                   "feature_names": list(self.feature_names_), "num_stocks": self.num_stocks_,
                   "formulas": [formula.to_dict() for formula in self.formulas_]}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GeneticFactorMiner":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported formula file")
        names, stocks = payload.get("feature_names"), payload.get("num_stocks")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Formula inputs must be unique nonempty names")
        if isinstance(stocks, bool) or not isinstance(stocks, int) or stocks < 3:
            raise ValueError("Formula file must specify at least three stocks")
        model = cls(rolling_window=payload["rolling_window"])
        model.formulas_ = tuple(FormulaNode.from_dict(value) for value in payload["formulas"])
        if not model.formulas_:
            raise ValueError("Formula file contains no selected factors")
        for formula in model.formulas_:
            for node_path in model._paths(formula):
                node = model._subtree(formula, node_path)
                if node.op == "feature" and node.feature not in names:
                    raise ValueError(f"Unknown saved input: {node.feature}")
        model.feature_names_, model.num_stocks_ = tuple(names), stocks
        return model

    def mine(
        self,
        features: Mapping[str, np.ndarray],
        future_returns: np.ndarray,
        top_k: int = 20,
    ) -> List[FactorResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= self.population_size:
            raise ValueError("top_k must be between 1 and population_size")
        self._validate_inputs(features, future_returns)
        features = {name: np.asarray(value, dtype=float) for name, value in features.items()}
        future_returns = np.asarray(future_returns, dtype=float)
        feature_names = tuple(features)
        population = [
            self._random_tree(feature_names, depth=1 + (i // 2) % self.max_depth, full=bool(i % 2))
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
        if self.rng.random() < self.hoist_probability and target:
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
        finite = np.isfinite(flat)
        if finite.sum() < 3 or np.std(flat[finite]) < 1e-10:
            return True
        if not self.use_similarity_filter:
            return False
        for other in accepted:
            other_flat = other.reshape(-1)
            valid = np.isfinite(flat) & np.isfinite(other_flat)
            if valid.sum() < 3 or np.std(flat[valid]) < 1e-10 or np.std(other_flat[valid]) < 1e-10:
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
        GeneticFactorMiner._validate_features(features)
        returns = np.asarray(future_returns, dtype=float)
        shape = returns.shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 3:
            raise ValueError("future_returns must use nonempty shape [T,N] with at least three stocks")
        if np.isinf(returns).any():
            raise ValueError("future_returns cannot contain infinite values")
        if not np.any(np.isfinite(returns).sum(axis=1) >= 3):
            raise ValueError("At least one training date must have three finite target returns")
        for name, values in features.items():
            if np.asarray(values).shape != shape:
                raise ValueError(f"Feature {name!r} has shape {np.asarray(values).shape}, expected {shape}")

    @staticmethod
    def _validate_features(features: Mapping[str, np.ndarray]) -> None:
        if not features:
            raise ValueError("features cannot be empty")
        shape = np.asarray(next(iter(features.values()))).shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 3:
            raise ValueError("Features must use nonempty shape [T,N] with at least three stocks")
        for name, values in features.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Feature names must be nonempty strings")
            if np.asarray(values).shape != shape:
                raise ValueError(f"Feature {name!r} has shape {np.asarray(values).shape}, expected {shape}")
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise ValueError(f"Feature {name!r} must be finite before GP evaluation")
