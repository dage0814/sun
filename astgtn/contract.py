from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch

from .history import PointInTimePrior, utc_seconds


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
    """Check declared time boundaries, not the authenticity of source evidence or feature construction."""
    required = {"window_cutoffs", "prediction_cutoffs", "target_times", "stock_symbols"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Historical runs require {sorted(missing)}; dates cannot be inferred safely")
    if "prior_adjacency" in bundle or "concept_membership" in bundle:
        raise ValueError(
            "Legacy undated prior_adjacency/concept_membership is rejected. "
            "Rebuild relation_events from dated evidence; do not backdate a recent snapshot."
        )
    samples, history, stocks, _ = bundle["quantitative"].shape
    symbols = bundle["stock_symbols"]
    if not isinstance(symbols, list) or len(symbols) != stocks or any(
        not isinstance(symbol, str) or not symbol.strip() for symbol in symbols
    ) or len(set(symbols)) != stocks:
        raise ValueError("stock_symbols must identify each stock once in the tensor/edge node order")
    horizons = bundle["targets"].shape[1]
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
