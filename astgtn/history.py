from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
import re
from typing import Any

import torch
from torch import nn


_ISO_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MIN_TIMESTAMP = -62135596800
_MAX_TIMESTAMP = 253402300799
_OPEN_END = torch.iinfo(torch.int64).max


def utc_seconds(value: str | datetime) -> int:

    if isinstance(value, str):
        if not _ISO_TIMESTAMP.fullmatch(value):
            raise ValueError("timestamp must be ISO 8601 with T, seconds and an explicit timezone")

        fraction = re.search(r"\.(\d+)", value)
        if fraction is not None and any(digit != "0" for digit in fraction.group(1)):
            raise ValueError("timestamp must use whole seconds; do not round publication times")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid timestamp: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("timestamp must be an explicit ISO string or aware datetime, not a numeric epoch")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an explicit timezone")
    if parsed.microsecond != 0:
        raise ValueError("timestamp must use whole seconds; do not round publication times")
    try:
        delta = parsed.astimezone(timezone.utc) - _EPOCH
    except (OverflowError, ValueError) as exc:
        raise ValueError("timestamp is outside the supported UTC datetime range") from exc
    if delta.microseconds:
        raise ValueError("timestamp timezone offset must preserve whole-second UTC precision")
    return delta.days * 86400 + delta.seconds


def _canonical_timestamp(value: str | datetime) -> str:
    utc_seconds(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class PointInTimePrior(nn.Module):


    def __init__(self, num_stocks: int, events: Sequence[Mapping[str, Any]]) -> None:
        super().__init__()
        if isinstance(num_stocks, bool) or not isinstance(num_stocks, int) or num_stocks < 1:
            raise ValueError("num_stocks must be a positive integer")
        if not isinstance(events, (list, tuple)):
            raise TypeError("events must be an explicit list or tuple of evidence-dated relation events")
        self.num_stocks = num_stocks
        required = {
            "relation_id", "source", "target", "known_at", "effective_at",
            "source_available_at", "source_ref", "action",
        }
        groups: dict[str, list[dict[str, Any]]] = {}
        identities: dict[str, tuple[int, int, bool]] = {}
        for position, raw in enumerate(events):
            if not isinstance(raw, Mapping):
                raise TypeError(f"relation event {position} must be a mapping")
            missing = required.difference(raw)
            if missing:
                raise ValueError(f"relation event {position} is missing required provenance: {sorted(missing)}")
            relation_id = raw["relation_id"]
            if not isinstance(relation_id, str) or not relation_id.strip():
                raise ValueError(f"relation event {position}: relation_id must be a nonempty string")
            source_ref = raw["source_ref"]
            if not isinstance(source_ref, str) or not source_ref.strip():
                raise ValueError(f"relation event {position}: source_ref must identify supporting evidence")
            source, target = raw["source"], raw["target"]
            for name, index in (("source", source), ("target", target)):
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < num_stocks:
                    raise ValueError(f"relation event {position}: {name} must be a stock index in [0,{num_stocks})")
            bidirectional = raw.get("bidirectional", True)
            if not isinstance(bidirectional, bool):
                raise ValueError(f"relation event {position}: bidirectional must be a boolean")
            identity = (source, target, bidirectional)
            if relation_id in identities and identities[relation_id] != identity:
                raise ValueError(f"relation {relation_id!r}: endpoints and bidirectional identity cannot change")
            identities[relation_id] = identity
            action = raw["action"]
            if action not in ("add", "remove"):
                raise ValueError(f"relation event {position}: action must be add or remove")
            weight = raw.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"relation event {position}: weight must be finite and positive")
            try:
                weight = float(weight)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"relation event {position}: weight must be finite and positive") from exc
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"relation event {position}: weight must be finite and positive")
            representable_weight = torch.tensor(weight, dtype=torch.float32)
            if not torch.isfinite(representable_weight).item() or representable_weight.item() <= 0:
                raise ValueError(f"relation event {position}: weight is outside positive float32 range")
            normalized: dict[str, Any] = {
                "relation_id": relation_id, "source": source, "target": target,
                "bidirectional": bidirectional, "source_ref": source_ref.strip(),
                "action": action, "weight": weight,
            }
            for key in ("known_at", "effective_at", "source_available_at"):
                try:
                    normalized[key] = _canonical_timestamp(raw[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"relation event {position}: invalid {key}: {exc}") from exc
            normalized["activation"] = max(
                utc_seconds(normalized[key])
                for key in ("known_at", "effective_at", "source_available_at")
            )
            groups.setdefault(relation_id, []).append(normalized)

        starts: list[int] = []
        ends: list[int] = []
        edge_indices: list[int] = []
        weights: list[float] = []
        self.events: list[dict[str, Any]] = []
        for relation_id in sorted(groups):
            ordered = sorted(groups[relation_id], key=lambda event: event["activation"])
            unique: list[dict[str, Any]] = []
            for event in ordered:
                if unique and event["activation"] == unique[-1]["activation"]:
                    if event != unique[-1]:
                        raise ValueError(f"relation {relation_id!r}: conflicting events at the same activation time")
                    continue
                unique.append(event)
            self.events.extend({key: value for key, value in event.items() if key != "activation"} for event in unique)
            for index, event in enumerate(unique):
                if event["action"] != "add":
                    continue
                end = unique[index + 1]["activation"] if index + 1 < len(unique) else _OPEN_END
                source, target = event["source"], event["target"]
                edges = [(source, target)]
                if event["bidirectional"] and source != target:
                    edges.append((target, source))
                for first, second in edges:
                    starts.append(event["activation"])
                    ends.append(end)
                    edge_indices.append(first * num_stocks + second)
                    weights.append(event["weight"])
        self.register_buffer("interval_starts", torch.tensor(starts, dtype=torch.int64))
        self.register_buffer("interval_ends", torch.tensor(ends, dtype=torch.int64))
        self.register_buffer("edge_indices", torch.tensor(edge_indices, dtype=torch.int64))
        self.register_buffer("edge_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, cutoffs: torch.Tensor) -> torch.Tensor:
        if not isinstance(cutoffs, torch.Tensor) or cutoffs.dtype != torch.int64 or cutoffs.ndim != 2:
            raise ValueError("cutoffs must be an int64 tensor of explicit UTC seconds with shape [B,T]")
        if cutoffs.shape[0] < 1 or cutoffs.shape[1] < 1:
            raise ValueError("cutoffs must contain at least one sample and one history step")
        if cutoffs.device != self.interval_starts.device:
            raise ValueError("cutoffs and PointInTimePrior must be on the same device")
        if torch.any((cutoffs < _MIN_TIMESTAMP) | (cutoffs > _MAX_TIMESTAMP)).item():
            raise ValueError("cutoffs contain an unknown/sentinel or out-of-range UTC timestamp")
        unique_cutoffs, inverse = torch.unique(cutoffs.reshape(-1), return_inverse=True)
        count, stocks = unique_cutoffs.numel(), self.num_stocks
        adjacency = torch.eye(stocks, device=cutoffs.device, dtype=self.edge_weights.dtype)
        adjacency = adjacency.reshape(1, stocks * stocks).expand(count, -1).clone()
        if self.edge_indices.numel():
            active = (unique_cutoffs[:, None] >= self.interval_starts[None, :]) & (
                unique_cutoffs[:, None] < self.interval_ends[None, :]
            )
            contributions = active.to(self.edge_weights.dtype) * self.edge_weights[None, :]
            adjacency.scatter_add_(1, self.edge_indices[None, :].expand(count, -1), contributions)
        adjacency = adjacency.reshape(count, stocks, stocks)
        row_sums = adjacency.sum(dim=-1, keepdim=True)
        if not torch.isfinite(row_sums).all().item():
            raise ValueError("relation weights overflow the graph dtype; use smaller finite weights")
        adjacency = adjacency / row_sums
        return adjacency[inverse].reshape(*cutoffs.shape, stocks, stocks)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        mismatches = []
        for name in ("interval_starts", "interval_ends", "edge_indices", "edge_weights"):
            key = prefix + name
            if key not in state_dict:
                continue
            incoming, expected = state_dict[key], getattr(self, name)
            if (
                not isinstance(incoming, torch.Tensor)
                or incoming.shape != expected.shape
                or (name != "edge_weights" and incoming.dtype != torch.int64)
                or not torch.equal(incoming.detach().cpu(), expected.detach().cpu())
            ):
                mismatches.append(key)
        if mismatches:
            error_msgs.append(
                "PointInTimePrior immutable provenance-derived graph mismatch: "
                + ", ".join(mismatches)
                + ". Rebuild explicitly from verified events instead of replacing temporal buffers."
            )
            return
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
