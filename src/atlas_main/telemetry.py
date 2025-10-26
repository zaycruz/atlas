"""Lightweight in-process telemetry for Atlas agent."""
from __future__ import annotations

import os
import threading
from collections import deque
from statistics import mean
from typing import Deque, Dict, Iterable, Tuple


def _percentile(samples: Iterable[float], percentile: float) -> float:
    data = sorted(samples)
    if not data:
        return 0.0
    k = (len(data) - 1) * percentile
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[int(k)]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1


class Telemetry:
    """Aggregate basic metrics for inspection."""

    _instance: "Telemetry" | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.enabled = os.getenv("ATLAS_METRICS", "1").strip().lower() not in {"0", "false", "off"}
        self._data_lock = threading.Lock()
        self._turn_durations: Deque[float] = deque(maxlen=200)
        self._tool_metrics: Dict[str, Dict[str, Deque[float] | int]] = {}
        self._compactions = 0
        self._snapshot_records: Deque[Tuple[int, int]] = deque(maxlen=200)
        self._layer_breakdown: Deque[Tuple[int, int, int, int, int, int, bool]] = deque(maxlen=200)

    @classmethod
    def instance(cls) -> "Telemetry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def observe_turn(self, duration: float) -> None:
        if not self.enabled:
            return
        with self._data_lock:
            self._turn_durations.append(max(0.0, float(duration)))

    def observe_tool(self, name: str, duration: float, status: str) -> None:
        if not self.enabled:
            return
        with self._data_lock:
            metrics = self._tool_metrics.setdefault(
                name,
                {
                    "durations": deque(maxlen=200),
                    "runs": 0,
                    "errors": 0,
                },
            )
            metrics["runs"] += 1  # type: ignore[assignment]
            if status != "success":
                metrics["errors"] += 1  # type: ignore[assignment]
            durations: Deque[float] = metrics["durations"]  # type: ignore[assignment]
            durations.append(max(0.0, float(duration)))

    def record_compaction(self) -> None:
        if not self.enabled:
            return
        with self._data_lock:
            self._compactions += 1

    def record_snapshot_tokens(self, before: int, after: int) -> None:
        if not self.enabled:
            return
        before = max(0, int(before))
        after = max(0, int(after))
        with self._data_lock:
            self._snapshot_records.append((before, after))

    def record_memory_layers(
        self,
        *,
        before: int,
        episodic: int,
        facts: int,
        reflections: int,
        total: int,
        budget: int,
        trimmed: bool,
    ) -> None:
        if not self.enabled:
            return
        before = max(0, int(before))
        episodic = max(0, int(episodic))
        facts = max(0, int(facts))
        reflections = max(0, int(reflections))
        total = max(0, int(total))
        budget = max(1, int(budget))
        trimmed = bool(trimmed)
        with self._data_lock:
            self._layer_breakdown.append((before, episodic, facts, reflections, total, budget, trimmed))

    def reset(self) -> None:
        with self._data_lock:
            self._turn_durations.clear()
            self._tool_metrics.clear()
            self._compactions = 0
            self._snapshot_records.clear()
            self._layer_breakdown.clear()

    def stats(self) -> Dict[str, Dict]:
        with self._data_lock:
            turn_list = list(self._turn_durations)
            snapshot_list = list(self._snapshot_records)
            tool_snapshot = {
                name: {
                    "runs": metrics["runs"],
                    "errors": metrics["errors"],
                    "p50": _percentile(metrics["durations"], 0.5) if metrics["durations"] else 0.0,  # type: ignore[arg-type]
                    "p95": _percentile(metrics["durations"], 0.95) if metrics["durations"] else 0.0,  # type: ignore[arg-type]
                }
                for name, metrics in self._tool_metrics.items()
            }
            layer_list = list(self._layer_breakdown)

        turn_stats = {
            "count": len(turn_list),
            "p50": _percentile(turn_list, 0.5) if turn_list else 0.0,
            "p95": _percentile(turn_list, 0.95) if turn_list else 0.0,
            "avg": mean(turn_list) if turn_list else 0.0,
        }

        snapshot_stats: Dict[str, float] = {"count": len(snapshot_list)}
        if snapshot_list:
            before_vals = [float(b) for b, _ in snapshot_list]
            after_vals = [float(a) for _, a in snapshot_list]
            savings = [max(b - a, 0.0) for b, a in snapshot_list]
            snapshot_stats.update(
                {
                    "avg_before": mean(before_vals),
                    "avg_after": mean(after_vals),
                    "avg_saved": mean(savings),
                }
            )
        else:
            snapshot_stats.update({"avg_before": 0.0, "avg_after": 0.0, "avg_saved": 0.0})

        layer_stats: Dict[str, float] = {"count": len(layer_list)}
        if layer_list:
            episodic_vals = []
            fact_vals = []
            reflection_vals = []
            total_vals = []
            budget_vals = []
            saved_vals = []
            trimmed_count = 0
            for before, episodic, facts, reflections, total, budget, trimmed in layer_list:
                episodic_vals.append(float(episodic))
                fact_vals.append(float(facts))
                reflection_vals.append(float(reflections))
                total_vals.append(float(total))
                budget_vals.append(max(1.0, float(budget)))
                saved_vals.append(max(float(before) - float(total), 0.0))
                if trimmed:
                    trimmed_count += 1
            layer_stats.update(
                {
                    "avg_episodic": mean(episodic_vals),
                    "avg_facts": mean(fact_vals),
                    "avg_reflections": mean(reflection_vals),
                    "avg_total": mean(total_vals),
                    "avg_saved": mean(saved_vals),
                    "avg_budget_utilization_pct": mean((total / budget) * 100.0 for total, budget in zip(total_vals, budget_vals)),
                    "trimmed_ratio": trimmed_count / len(layer_list),
                }
            )
        else:
            layer_stats.update(
                {
                    "avg_episodic": 0.0,
                    "avg_facts": 0.0,
                    "avg_reflections": 0.0,
                    "avg_total": 0.0,
                    "avg_saved": 0.0,
                    "avg_budget_utilization_pct": 0.0,
                    "trimmed_ratio": 0.0,
                }
            )

        return {
            "turns": turn_stats,
            "tools": tool_snapshot,
            "compactions": {"count": self._compactions},
            "snapshots": snapshot_stats,
            "memory_layers": layer_stats,
        }
