# Roadmap

每个 Milestone 完成后：跑通测试 → 更新本文件勾选框与 `STATUS.md` → `git commit`。
未完成的代码位置用 `TODO(Mx)` 标注；未来里程碑的验收测试已用 `pytest.mark.skip(reason="Mx: ...")` 写好。

---

## M0 · 项目骨架 ✅

- [x] 目录结构、`pyproject.toml`、Docker Compose（API + Qdrant + Redis）、Dockerfile（含 HEALTHCHECK）
- [x] JWT 身份校验 → `Identity`（`tenant_id / role / user_id`）
- [x] ACL Filter 构造器（tenant + role，数据库查询级）
- [x] PII / 密钥脱敏器 + 单行 JSON 日志
- [x] `contextvars` trace_id + `Span` 异步上下文管理器
- [x] 三态熔断器（Closed / Open / Half-Open，可注入时钟）
- [x] 防串味缓存 Key + `should_cache` 规则
- [x] 引用编号越界校验
- [x] 检索内容随机 nonce 包裹 + 标签逃逸防护 + 注入特征扫描
- [x] 模型单价计算、Prompt 版本化加载
- [x] Mock 文档样例（2 租户、3 级权限、恶意注入文档、模拟 PII）
- [x] Golden set 样例与 schema 校验

---

## M1 · 基础链路 + 权限 + 日志（第 1~3 天）

目标：文档带 `tenant_id / allowed_roles` 入库；`/v1/chat` 跑通「检索 → 回答」最小链路（暂时非流式、单模型）。

- [ ] 补齐 mock 文档至 15~20 份（acme_corp + beta_inc，all / eng / hr，含 2 份恶意注入文档）
- [ ] `app/rag/embeddings.py`：接入 bge-m3（dense + sparse；本地 FlagEmbedding 或 API）
- [ ] `app/rag/ingest.py`：切片 → 向量化 → upsert，Payload 含 `tenant_id / allowed_roles / doc_id / chunk_index / index_version`；入库时用 `screen_text` 标记 `suspicious`
- [ ] Qdrant collection：named vectors `dense` + `sparse`，对 `tenant_id`、`allowed_roles` 建 payload index
- [ ] `app/rag/retriever.py`：Query API `prefetch(dense, sparse)` + RRF 融合，**每个 prefetch 都带 ACL filter**
- [ ] 单一 OpenAI 兼容 Provider 的非流式 `chat()`（经 router，不直连）
- [ ] `/v1/chat` 串起：鉴权 → trace → 检索 → 包裹文档 → 生成 → 返回 `answer + citations + trace_id`
- [ ] `/ready` 检查 Qdrant + Redis

**验收**
- intern 和 hr 问同一个薪资问题：日志中 trace_id 不同，intern 的 `retrieved_doc_ids` 不含任何 hr 文档
- beta_inc 用户检索不到任何 acme_corp 切片（集成测试，断言检索层结果）

---

## M2 · 多模型网关 + 熔断降级 + 预算 + 缓存 + SSE（第 4~6 天）

- [ ] Provider Adapter：OpenAI 兼容 + Anthropic（system 单独传参、usage 字段名差异）
- [ ] 错误分类：可重试（超时 / 429 / 5xx）vs 不可重试（400 / 401 / 403 / 内容拒绝）；不可重试不计入熔断
- [ ] `ModelRouter`：按 `configs/models.yaml` 路由链降级；指数退避 + full jitter；尊重 `Retry-After`；总 deadline
- [ ] 熔断器按 `provider:model` 维度隔离；响应 metadata 含 `model_used`、`degraded`
- [ ] **SSE 流式**：首 token 前可重试/降级，首 token 后失败发 `event: error` 结束
- [ ] `BudgetGuard`：Redis `budget:{tenant}:{YYYY-MM-DD}` 预扣 → 实际 usage 补差；超额 429 或降级小模型
- [ ] L1 `TTLCache(500, 60s)` + L2 Redis（1h）；`index_version` 存 Redis，L1 短 TTL 缓存版本号
- [ ] Query 改写（小模型）：仅多轮或短 query 触发；失败/超时回退原始 query

**验收**
- FakeLLM 主模型连续 5 次 429 → 第 6 次不再请求主模型，直接降级，`degraded=true`（`tests/test_router_fallback.py`）
- 同一问题连问两次：第二次 `cache_hit=true`、`cost_usd=0`
- 并发 20 个请求打满预算时，实际花费不超过日预算 + 单请求上限

---

## M3 · 防幻觉 + 安全审计（第 7~9 天）

- [ ] Rerank（外部 API 或本地 bge-reranker），Top-20 → Top-5
- [ ] 低分拒答门控：Top-1 分数 < 阈值 → 固定话术拒答，不调用大模型
- [ ] **阈值用 golden set 扫描标定**（输出阈值 vs 准确率/拒答率曲线），写入 README
- [ ] 流式结束后聚合全文做引用校验，`citation_mismatch=true` 时追加免责提示 + 记指标
- [ ] 审计日志：`who / tenant / role / query_hash / retrieved_doc_ids / decision / ts`（append-only JSONL）
- [ ] `rag_request_summary` 事件：`steps_ms`、`model_used`、`degraded`、`prompt/completion tokens`、`estimated vs actual`、`cost_usd`
- [ ] 可选：对接 LangFuse

**验收**
- 问库里不存在的问题：低分拒答，trace 中没有 `llm` span
- 召回恶意注入文档时：输出不包含注入载荷（钓鱼链接 / 篡改数字），不出现标签逃逸

---

## M4 · 评估集 + 交付（第 10~12 天）

- [ ] `evals/golden_set.json` 扩充至 30~50 题：answerable 50% / no_answer 20% / unauthorized + cross_tenant 15% / injection 15%
- [ ] `evals/run_eval.py`：关键词判定 + 引用准确率 + 检索层越权断言 +（可选）LLM-as-judge faithfulness
- [ ] 与 `evals/baselines/latest.json` 对比：成功率、拒答准确率、越权泄露数（必须为 0）、P50/P95 延迟、总花费
- [ ] README：架构图、实测指标截图、阈值曲线、设计取舍

**验收**
- `python -m evals.run_eval` 一键输出对比报告；越权泄露 = 0；注入成功 = 0

---

## 非目标（明确不做）

- 真实 SSO / 用户管理（JWT 由脚本签发，模拟上游网关）
- 多进程间共享熔断状态（每个 worker 独立，README 中说明取舍）
- 生产级 PDF 解析（mock 文档以 Markdown 为主）
