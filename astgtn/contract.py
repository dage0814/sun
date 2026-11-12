from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch

from .history import PointInTimePrior, utc_seconds


def _tensor(bundle, key, *, shape=None, ndim=None):
    value = bundle.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{key} must have {ndim} dimensions, got {value.ndim}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{key} must use shape {shape}, got {tuple(value.shape)}")
    if value.is_complex() or not torch.isfinite(value).all().item():
        raise ValueError(f"{key} must contain finite real values")
    return value


def _integer_values(value, name, *, binary=False):
    if value.is_floating_point() and not torch.equal(value, value.trunc()):
        raise ValueError(f"{name} must contain integer values")
    if (value < 0).any().item() or (binary and (value > 1).any().item()):
        qualifier = "binary 0/1" if binary else "nonnegative integer"
        raise ValueError(f"{name} must contain {qualifier} values")


def _validate_tensor_shapes(bundle, *, use_text):
    quant = _tensor(bundle, "quantitative", ndim=4)
    samples, history, stocks, features = quant.shape
    if min(quant.shape) < 1:
        raise ValueError("quantitative requires positive [samples,history,stocks,features] dimensions")
    technical = _tensor(bundle, "technical", ndim=4)
    if technical.shape[:3] != quant.shape[:3]:
        raise ValueError("technical and quantitative must share sample/history/stock dimensions")
    caps = _tensor(bundle, "market_caps", shape=(samples, history, stocks))
    for key, value in (("quantitative", quant), ("technical", technical), ("market_caps", caps)):
        if not value.is_floating_point():
            raise ValueError(f"{key} must be floating-point")
    if (caps <= 0).any().item():
        raise ValueError("market_caps must be positive; missing or nonpositive values require preprocessing")
    labels = _tensor(bundle, "targets", ndim=3)
    if labels.shape[0] != samples or labels.shape[2] != stocks or labels.shape[1] < 1:
        raise ValueError("targets must use shape [samples,positive horizon,stocks]")
    _integer_values(labels, "targets", binary=True)
    industries = _tensor(bundle, "stock_industry", shape=(stocks,))
    _integer_values(industries, "stock_industry")
    num_industries = int(industries.max().item()) + 1

    if use_text:
        has_ids, has_mask = "input_ids" in bundle, "attention_mask" in bundle
        if has_ids != has_mask:
            raise ValueError("input_ids and attention_mask must be supplied together")
        if has_ids:
            ids = _tensor(bundle, "input_ids", ndim=4)
            if ids.shape[:2] != (samples, history) or min(ids.shape[2:]) < 1:
                raise ValueError("input_ids must use [samples,history,positive news count,positive token count]")
            _integer_values(ids, "input_ids")
            mask = _tensor(bundle, "attention_mask", shape=ids.shape)
            _integer_values(mask, "attention_mask", binary=True)
            if "news_mask" in bundle:
                news = _tensor(bundle, "news_mask", shape=ids.shape[:-1])
                _integer_values(news, "news_mask", binary=True)
                if (news.bool() & ~mask.bool().any(dim=-1)).any().item():
                    raise ValueError("news_mask marks a headline valid without any unmasked tokens")
        elif "news_mask" in bundle:
            raise ValueError("news_mask requires input_ids and attention_mask")
        if "industry_relevance" in bundle:
            _tensor(bundle, "industry_relevance", shape=(samples, history, num_industries))
    return samples, history, stocks, labels.shape[1]


def _time_grid(values, shape, name):
    try:
        if len(shape) == 1:
            result = torch.tensor([utc_seconds(value) for value in values], dtype=torch.long)
        else:
            result = torch.tensor(
                [[utc_seconds(value) for value in row] for row in values], dtype=torch.long
            )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: explicit timezone-aware ISO timestamps are required: {exc}") from exc
    if tuple(result.shape) != tuple(shape):
        raise ValueError(f"{name} must use shape {shape}, got {tuple(result.shape)}")
    return result


def validate_temporal_bundle(bundle: Mapping, *, use_prior_graph: bool, use_text: bool):

    required = {"window_cutoffs", "prediction_cutoffs", "target_times", "stock_symbols"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Historical runs require {sorted(missing)}; dates cannot be inferred safely")
    if "prior_adjacency" in bundle or "concept_membership" in bundle:
        raise ValueError(
            "Legacy undated prior_adjacency/concept_membership is rejected. "
            "Rebuild relation_events from dated evidence; do not backdate a recent snapshot."
        )
    samples, history, stocks, horizons = _validate_tensor_shapes(bundle, use_text=use_text)
    symbols = bundle["stock_symbols"]
    if not isinstance(symbols, list) or len(symbols) != stocks or any(
        not isinstance(symbol, str) or not symbol.strip() for symbol in symbols
    ) or len(set(symbols)) != stocks:
        raise ValueError("stock_symbols must identify each stock once in the tensor/edge node order")
    cutoffs = _time_grid(bundle["window_cutoffs"], (samples, history), "window_cutoffs")
    predictions = _time_grid(bundle["prediction_cutoffs"], (samples,), "prediction_cutoffs")
    targets = _time_grid(bundle["target_times"], (samples, horizons), "target_times")
    if (cutoffs[:, 1:] <= cutoffs[:, :-1]).any():
        raise ValueError("Window cutoffs must increase strictly within each history window")
    if (predictions[1:] <= predictions[:-1]).any():
        raise ValueError("Samples must be in strictly increasing prediction-time order")
    if (cutoffs > predictions[:, None]).any():
        raise ValueError("A history window contains information after its prediction cutoff")
    if (targets <= predictions[:, None]).any() or (targets[:, 1:] <= targets[:, :-1]).any():
        raise ValueError("Target times must increase strictly and occur after the prediction cutoff")

    industry_provenance = bundle.get("stock_industry_provenance")
    if use_text:
        if not isinstance(industry_provenance, list) or len(industry_provenance) != stocks:
            raise ValueError("Text adaptation requires stock_industry_provenance for every stock")
        for index, record in enumerate(industry_provenance):
            if not isinstance(record, dict) or not isinstance(record.get("source_ref"), str) or not record["source_ref"].strip():
                raise ValueError(f"Industry mapping {index} has no source_ref")
            if record.get("stock_symbol") != symbols[index] or record.get("industry_id") != int(bundle["stock_industry"][index]):
                raise ValueError(f"Industry mapping {index} evidence does not identify the supplied stock/category")
            try:
                available = max(utc_seconds(record[key]) for key in (
                    "known_at", "effective_at", "source_available_at"
                ))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Industry mapping {index} lacks dated evidence") from exc
            if available > int(cutoffs.min()):
                raise ValueError(
                    f"Industry mapping {index} was not available throughout the history. "
                    "Provide a historically valid mapping or shorten the sample period."
                )

    events = bundle.get("relation_events")
    if use_prior_graph and not isinstance(events, list):
        raise ValueError(
            "Prior-graph runs require explicit relation_events. Use [] only for an "
            "intentional self-loop-only prior, never fabricate missing edge dates."
        )
    prior = PointInTimePrior(stocks, events) if use_prior_graph else None
    audited_events = prior.events if prior is not None else []
    canonical_events = json.dumps(audited_events, sort_keys=True, ensure_ascii=False)
    report = {
        "policy": "point_in_time_v1",
        "stock_symbols": symbols,
        "relation_gate": "max(known_at,effective_at,source_available_at) <= window_cutoff",
        "cutoff_unit": "UTC epoch seconds; source input requires explicit timezone",
        "window_cutoffs": cutoffs.tolist(),
        "prediction_cutoffs": predictions.tolist(),
        "target_times": targets.tolist(),
        "relation_events": audited_events,
        "relation_event_sha256": hashlib.sha256(canonical_events.encode("utf-8")).hexdigest(),
        "prior_mode": ("dated_events" if events else "self_loops_only") if use_prior_graph else "disabled",
        "stock_industry_provenance": industry_provenance if use_text else None,
        "mixup_prior_policy": "elementwise minimum of both donor window cutoffs; training only",
        "scope": "Declared relation availability and sample boundaries; raw source authenticity and financial feature provenance require separate verification",
    }
    return cutoffs, prior, report


def validate_split_boundaries(report, *subsets):
    for subset in subsets:
        if not len(subset):
            raise ValueError("Temporal purge produced an empty split; supply more aligned samples")
    for left, right in zip(subsets, subsets[1:]):
        latest_label = max(max(report["target_times"][i]) for i in left.indices)
        next_prediction = min(report["prediction_cutoffs"][i] for i in right.indices)
        if latest_label >= next_prediction:
            raise ValueError("Split boundary uses labels not available before the next evaluation period")
