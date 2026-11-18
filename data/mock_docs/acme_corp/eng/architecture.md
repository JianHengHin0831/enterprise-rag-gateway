---
doc_id: acme_eng_architecture
tenant_id: acme_corp
allowed_roles: [eng]
title: Core System Architecture v2
---
# Core System Architecture v2 (Restricted: Engineering)

> **Access Scope**: Confidential internal technical specification for `acme_corp` Engineering (`eng`) personnel only. Do not share internal topology, port bindings, or error runbooks outside authorized engineering channels.

---

## 1. Service Topology & Communication

### 1.1 Core Microservices
Acme's core transactional platform consists of three primary domain services running on Kubernetes (EKS, `prod-us-east-1`):
* **`order-svc`**: Manages shopping carts, order state machines, inventory reservation, and checkout orchestration.
* **`pay-svc`**: Handles payment gateway integrations, webhook verification, ledger entries, and refunds.
* **`user-svc`**: Manages authentication tokens, account preferences, and profile synchronization.

### 1.2 Internal vs. External Protocols
* **External Access**: All external client traffic enters through the **API Gateway** (Kong/Envoy) and is exposed strictly as **RESTful JSON over HTTPS (TLS 1.3)** with JWT authentication and per-tenant rate limiting.
* **Internal Communication**: `order-svc`, `pay-svc`, and `user-svc` communicate with each other exclusively over **internal gRPC** (Protocol Buffers v3) secured with **mTLS** inside the `*.internal.acme` service mesh.
* **Timeouts & Idempotency**: All inter-service gRPC calls enforce a default **800 ms deadline** (2,000 ms for `pay-svc` settlement calls). Mutating calls between `order-svc` and `pay-svc` must include a UUIDv4 `idempotency_key` header valid for 24 hours.

---

## 2. Data Storage & Caching Layer

### 2.1 Relational Database (PostgreSQL 16)
* Both `order-svc` and `pay-svc` persist transactional records in a shared **PostgreSQL 16 primary/replica cluster** (1 primary writer + 2 asynchronous read replicas) with **read/write splitting** at the connection pooler (PgBouncer) level.
* **Replication Lag Fallback**: Read queries normally hit replicas; however, immediate post-checkout status checks (`read_your_writes=true`) or replica lag exceeding **500 ms** automatically route reads back to the primary node.
* Cross-service joins are strictly forbidden; `order-svc` queries `pay-svc` via gRPC rather than reading `pay_db` tables directly.

### 2.2 Analytical & Profile Storage (ClickHouse)
* **User Profiles**: User profiles, behavioral attributes, and event timelines owned by `user-svc` are stored in **ClickHouse** (`ReplacingMergeTree` partitioned by `toYYYYMM(updated_at)`).
* Profile lookups by `user_id` use a secondary materialized projection to keep P99 read latency under **25 ms**.

### 2.3 Distributed Cache (Redis Cluster)
* The shared caching layer is a **Redis Cluster** (6 shards, 1 replica per shard).
* **Mandatory Key Prefixing**: Every cache key **must be prefixed with the owning service name** to prevent namespace collisions and simplify memory auditing (e.g., `order-svc:cart:{user_id}`, `pay-svc:idemp:{key}`, `user-svc:profile:{user_id}`).
* Cache entries must specify a finite TTL (default 15 minutes for profiles, 24 hours for idempotency locks). Unbounded keys without TTL are blocked by CI linting.

---

## 3. Release Policy & Rollback SOP

### 3.1 Production Release Windows
* **Standard Release Schedule**: Production deployments take place strictly on **Tuesdays and Thursdays between 14:00 and 17:00** local time.
* **Staging Soak Requirement**: Every release candidate must pass automated integration tests and complete **at least 24 hours of observation in the `staging` environment** with zero P1/P2 alerts before production promotion.
* **Freeze Rules**: No routine production releases are permitted on Mondays, Fridays, or on Wednesday ("no-meeting day") evenings. Emergency P0 hotfixes outside the standard window require explicit sign-off from the On-Call Incident Commander and VP of Engineering.

### 3.2 Canary Rollout & Auto-Rollback Thresholds
Deployments follow a 3-stage Argo Rollouts canary progression: **5% (15 min) $\rightarrow$ 25% (20 min) $\rightarrow$ 100%**.
Automated rollback triggers immediately if any of the following occur during canary observation:
* HTTP 5xx or gRPC `INTERNAL` error rate exceeds **0.5%** over a 3-minute sliding window;
* P95 latency for `order-svc` checkout exceeds **400 ms** for 3 consecutive minutes;
* Payment callback verification failure rate spikes above **1%**.

---

## 4. Secrets, Key Rotation & Security Operations

### 4.1 Webhook Key Rotation (`pay-svc`)
* `pay-svc` verifies external payment processor callbacks using **HMAC-SHA256** signatures.
* Signing keys are managed in HashiCorp Vault and rotated every **90 days**. During rotation, `pay-svc` supports a **48-hour dual-key overlap window** (accepting both `current` and `previous` key versions) before revoking the old secret.

### 4.2 Environment Access
Debugging production pods or viewing internal Grafana/Prometheus dashboards requires **Acme-VPN (WireGuard)** connected via Okta MFA (refer to the *Employee Handbook* for VPN error codes). Direct `kubectl exec` in production is disabled; use read-only ephemeral debug containers logged to the security audit stream.

---

## 5. Internal Error Code Runbook (Troubleshooting Reference)

| Error Code | Owning Service | Meaning & Probable Cause | Immediate Remediation |
| :--- | :--- | :--- | :--- |
| **`E-4031`** | `pay-svc` | **Payment callback signature check failed**. Usually caused by an expired webhook secret or failed Vault sync during key rotation. | Check the key rotation status and dual-key overlap config of `pay-svc` in Vault; notify `#eng-payments` if third-party PSP mismatch persists. |
| **`E-4012`** | `order-svc` | **Downstream gRPC deadline exceeded** calling `pay-svc` (> 800 ms / 2,000 ms). | Verify `pay-svc` circuit breaker state and PG16 connection pool saturation; ensure client retries reuse the same `idempotency_key`. |
| **`E-5024`** | `order-svc` / `pay-svc` | **PostgreSQL replica lag > 500 ms**; read/write splitter fell back to primary. | Inspect long-running analytical queries accidentally hitting PG16 replicas; verify heavy profile queries are routed to ClickHouse instead. |
| **`E-5039`** | Shared / Redis | **Redis Cluster slot migration timeout or connection pool exhaustion**. | Check `redis-cluster` shard CPU and verify no unprefixed large keys (`> 1 MB`) were deployed in the recent release. |