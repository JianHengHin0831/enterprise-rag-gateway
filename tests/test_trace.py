import asyncio

import pytest

from app.observability.trace import Span, current_spans, current_trace_id, start_trace, steps_ms


async def test_spans_recorded_under_trace():
    tid = start_trace()
    async with Span("retrieve") as s:
        s.set("hits", 3)
    async with Span("rerank"):
        pass
    assert current_trace_id() == tid
    assert [s.name for s in current_spans()] == ["retrieve", "rerank"]
    assert set(steps_ms()) == {"retrieve", "rerank"}


async def test_span_records_error_status():
    start_trace()
    with pytest.raises(ValueError):
        async with Span("llm"):
            raise ValueError("boom")
    assert current_spans()[0].status == "error"


async def test_concurrent_requests_isolated():
    async def request(name: str) -> tuple[str, list[str]]:
        tid = start_trace()
        async with Span(name):
            await asyncio.sleep(0.01)
        return tid, [s.name for s in current_spans()]

    (t1, s1), (t2, s2) = await asyncio.gather(
        asyncio.create_task(request("a")), asyncio.create_task(request("b"))
    )
    assert t1 != t2 and s1 == ["a"] and s2 == ["b"]
