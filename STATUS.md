# Status

**Current milestone:** M1 — 基础链路 + 权限 + 日志
**Last updated:** 2026-11-18

## Done
- M0 skeleton: 纯逻辑模块（auth / ACL filter / sanitizer / trace / circuit breaker / cache key / citations / injection wrapping / pricing / prompt loader）已实现并有单测。

## In progress
- (none)

## Next
- M1 第一项：补齐 mock 文档，接入 bge-m3，实现入库脚本。

## Known gaps
- `/v1/chat` 当前返回 501（链路未接通）。
- Provider adapters、retriever、reranker、budget、L1/L2 cache 均为 `TODO(Mx)` 桩。
- `configs/models.yaml` 中单价为 `null`，需按官方当前价格填写。
