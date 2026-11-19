"""In-process counters, exposed in Prometheus text format at `GET /metrics`.

Per worker process (like the breakers). Labels are fixed enums only, never tenant, user or
query values, so the endpoint leaks nothing about who asked what.
"""

import threading
from collections import Counter

_HELP = {
    "rag_citation_mismatch_total": "Answers whose [n] citations point outside the contexts.",
    "rag_refusals_total": "Refusals by reason (low_score / no_results / model).",
    "rag_rerank_gate_total": "Low-score gate decisions (passed / refused).",
}

_lock = threading.Lock()
_counts: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()


def inc(name: str, value: int = 1, **labels: str) -> None:
    if name not in _HELP:
        raise KeyError(f"unknown metric: {name}")
    with _lock:
        _counts[(name, tuple(sorted(labels.items())))] += value


def get(name: str, **labels: str) -> int:
    with _lock:
        return _counts[(name, tuple(sorted(labels.items())))]


def reset() -> None:
    """Tests only."""
    with _lock:
        _counts.clear()


def render() -> str:
    with _lock:
        items = sorted(_counts.items())
    lines: list[str] = []
    for name, help_text in _HELP.items():
        lines += [f"# HELP {name} {help_text}", f"# TYPE {name} counter"]
        for (metric, labels), value in items:
            if metric != name:
                continue
            label_str = ",".join(f'{k}="{v}"' for k, v in labels)
            lines.append(f"{name}{{{label_str}}} {value}" if label_str else f"{name} {value}")
    return "\n".join(lines) + "\n"
