# Enterprise RAG Gateway

企业内部知识库智能问答网关：**多租户 + 角色权限隔离 + LLM 网关容错 + 防幻觉闭环 + 成本控制 + 全链路可观测**。

模拟一家科技公司的 HR / 研发 / 全员文档库，两个租户（`acme_corp`、`beta_inc`），四种角色（`intern`、`employee`、`eng`、`hr`）。

> 状态：M0 骨架完成，M1 进行中。详见 [ROADMAP.md](ROADMAP.md) / [STATUS.md](STATUS.md)。

---

## 核心特性

| 能力 | 实现要点 |
|---|---|
| 身份与权限 | JWT 验签得到 `tenant_id / role`；Qdrant **查询级** Payload 预过滤（每个 prefetch 都带 filter），绝不后置过滤 |
| 防注入 | 检索内容用随机 nonce 标签包裹并转义伪造标签；入库时扫描注入特征；System Prompt 声明资料不可执行 |
| 防幻觉 | 混合检索（dense + sparse，RRF）→ Rerank → 低分拒答门控（不调大模型）→ 强制 `[n]` 引用 → 引用越界校验 |
| 多模型容错 | Provider Adapter 抽象；可重试/不可重试错误分类；按 `provider:model` 隔离的三态熔断器；指数退避 + jitter + 总 deadline；降级链 |
| 流式 | SSE；**首 token 前**可重试/降级，首 token 后失败发 `event: error` |
| 成本 | tiktoken 预估 + 实际 usage 对账；Redis 日预算「预扣 → 补差」防并发超额 |
| 缓存 | L1 进程内 TTLCache + L2 Redis；Key 绑定 tenant / role / prompt 版本 / 改写 prompt 版本 / index 版本；降级结果不缓存 |
| 可观测 | `contextvars` trace_id；`Span` 记录 rewrite / retrieve / rerank / llm；单行 JSON 日志自动脱敏；`rag_request_summary` 汇总事件；独立审计日志 |
| 评估 | Golden set（正常 / 无答案陷阱 / 越权 / 跨租户 / 注入），与基线对比 |

---

## 请求链路

```
Client ──(Authorization: Bearer <JWT>)──►
 ① 鉴权：验签 → Identity；生成 trace_id
 ② 预算：按估算 token 预扣 budget:{tenant}:{date}（超额 → 429 / 降级小模型）
 ③ 输入安全：注入特征初筛
 ④ Query 改写（小模型，仅多轮/短 query；失败回退原 query）
 ⑤ 缓存查找：key = sha256(tenant, role, prompt_v, rewrite_v, index_v, query)
      └─ 命中 → 流式返回（cache_hit=true, cost=0）
 ⑥ 混合检索：Qdrant prefetch[dense, sparse] + RRF，ACL filter 下推到每个 prefetch → Top-20
 ⑦ Rerank → Top-5；Top-1 < 阈值 → 固定话术拒答（不调大模型）
 ⑧ ModelRouter：熔断器 + 退避 + 降级链 → SSE 流式（带 [n] 引用）
 ⑨ 后置：引用越界校验 → should_cache → 写缓存 → 预算补差 → 审计日志 + summary 事件
```

---

## 快速开始

```bash
cp .env.example .env            # 填入 API Key
docker compose up -d qdrant redis
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"     # Linux/macOS: .venv/bin/pip
.venv/Scripts/pytest tests/ -v
.venv/Scripts/uvicorn app.main:app --reload --port 8000
```

签发测试身份并调用：

```bash
TOKEN=$(.venv/Scripts/python -m scripts.mint_token --tenant acme_corp --role hr --user u_hr_01)
curl -s localhost:8000/v1/chat -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"query": "绩效考核分几档？"}'
```

一键起全部服务：`docker compose up --build`

---

## 目录结构

```
app/
  main.py            FastAPI 入口：/health /ready /v1/chat，trace 中间件
  config.py          环境变量配置（RAG_ 前缀）
  schemas.py         Identity、请求/响应、检索结果等共享模型
  security/          auth（JWT）· acl（Qdrant filter）· injection（包裹/扫描）· audit
  observability/     trace（contextvars + Span）· sanitizer（脱敏）· logging（JSON）
  gateway/           base（Provider 抽象/错误分类）· circuit_breaker · router · providers/
  cost/              tokens（tiktoken 估算）· pricing · budget（预扣/补差）
  cache/             keys（防串味 Key + should_cache）· layered（L1/L2）
  rag/               ingest · embeddings · retriever · reranker · citations · prompts · pipeline
configs/             models.yaml（路由/单价/熔断/重试）· prompts/（版本化 Prompt）
data/mock_docs/      <tenant>/<access>/*.md，front matter 声明 tenant_id / allowed_roles
evals/               golden_set.json · run_eval.py · baselines/
scripts/             mint_token.py（签发开发用 JWT）
tests/               单测 + FakeLLM 集成测试（未来里程碑的验收测试以 skip 形式预置）
```

---

## 设计取舍（面试问答速查）

- **为什么不信任 `X-User-Role` Header？** 客户端可以随便伪造。身份只能来自验签后的 JWT（生产中由上游 SSO/网关签发）；ACL filter 只能由该身份生成，LLM 改写结果永远不能影响 filter。
- **为什么必须在查询级过滤？** 先取 Top-K 再在 Python 里过滤，有权限的结果会被挤出 Top-K，而且无权文档已经被读出内存。
- **为什么缓存放在 Query 改写之后？** 多轮对话里「它的上限是多少」依赖上下文，按原始 query 缓存会把 A 对话的答案返回给 B。
- **为什么降级结果不缓存？** 否则一次故障期间的兜底答案会被缓存 1 小时。
- **为什么流式只在首 token 前降级？** 已发出的 token 无法撤回；中途换模型会拼出两种风格、互相矛盾的答案。
- **为什么预算要预扣？** 「先查后扣」在并发下会超额。
- **熔断状态为何不跨进程共享？** Demo 取舍；多 worker 各自熔断仍能保护下游，跨进程共享需要 Redis 且会引入新的故障点。
- **拒答阈值怎么定？** 不同 Reranker 的分数分布不同，阈值由 golden set 扫描标定（见 M3 阈值曲线）。

---

## 指标（M4 填充）

| 指标 | 值 |
|---|---|
| 正常问答命中率 | – |
| 无答案拒答准确率 | – |
| 越权泄露数 | – （目标 0） |
| 注入成功数 | – （目标 0） |
| P50 / P95 延迟 | – |
| 平均单次成本 | – |
