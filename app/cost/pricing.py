from app.gateway.base import Usage
from app.gateway.router import ModelSpec


def compute_cost_usd(usage: Usage, spec: ModelSpec) -> float | None:
    """Cost from per-1M-token prices. Returns None if the price is not configured."""
    if spec.input_per_mtok is None or spec.output_per_mtok is None:
        return None
    cost = (
        usage.prompt_tokens * spec.input_per_mtok + usage.completion_tokens * spec.output_per_mtok
    ) / 1_000_000
    return round(cost, 8)
