---
doc_id: beta_eng_architecture
tenant_id: beta_inc
allowed_roles: [eng]
title: Beta Engineering Runbook & Architecture
---
# Beta Engineering Runbook & Architecture (Restricted: Engineering)

> **Access Scope**: Internal technical specification for `beta_inc` Engineering (`eng`) personnel only. Do not share internal topologies, Kafka topics, or runbook error codes outside authorized engineering channels.

---

## 1. Service Topology & Communication

### 1.1 Core Services
Beta Inc.'s SaaS platform runs on **AWS ECS (Fargate)** in `ap-southeast-1` using **Go 1.23** microservices:
* **`auth-gw`**: Edge authentication gateway handling JWT validation, tenant resolution, and rate limiting.
* **`billing-worker`**: Asynchronous subscription invoicing, usage metering, and dunning processor.
* **`analytics-api`**: Customer reporting and dashboard query service.

### 1.2 Communication Patterns (Event-Driven + REST)
* **Synchronous Calls**: External clients and internal synchronous service lookups use **REST over HTTPS/JSON** (Beta does **not** use gRPC internally).
* **Asynchronous Messaging**: `auth-gw`, `billing-worker`, and `analytics-api` communicate asynchronously via **Apache Kafka** (AWS MSK, 3 brokers) using Avro schemas registered in Confluent Schema Registry.
* **Consumer Retry Policy**: Failed Kafka messages retry 3 times with exponential backoff before moving to the Dead Letter Queue (`beta.dlq.*`).

---

## 2. Data Storage & Caching Layer

### 2.1 Relational & Document Databases
* **MySQL 8.0 (InnoDB)**: Primary transactional database (Amazon RDS Aurora MySQL) storing tenant accounts, subscriptions, and invoices used by `auth-gw` and `billing-worker`.
* **MongoDB 7.0**: Document store used by `analytics-api` to store flexible dashboard schemas, raw usage snapshots, and audit event logs.

### 2.2 Cache Layer (Memcached)
* Beta uses **Amazon ElastiCache for Memcached** (not Redis) for ephemeral session and feature-flag caching.
* **Key Convention**: All Memcached keys must start with the prefix **`beta:`** followed by service and tenant ID (e.g., `beta:auth-gw:session:{tenant}:{id}`), with a hard maximum TTL of **6 hours**.

---

## 3. Release Policy & Deployment Windows

### 3.1 Production Deployment Schedule
* **Standard Release Window**: Production releases happen strictly on **Mondays and Wednesdays between 14:00 and 16:00** local time (avoiding Wednesday morning 10:00–12:00 All-Hands & Sprint Planning).
* **Staging Requirement**: Release builds must run cleanly in `staging` for **at least 12 hours** and pass automated smoke tests before production promotion.
* **Rollback Rule**: Blue/Green ECS task-set shifts revert automatically if HTTP 5xx exceeds **1.0%** or P99 latency exceeds **600 ms** for 2 consecutive minutes.

### 3.2 Bastion & Prod Access
Engineers accessing staging/prod dashboards or Aurora read-only consoles must connect via **BetaVPN client version 5.2 or later** (see *Beta IT FAQ* at `it.beta.example`).

---

## 4. Beta Engineering Error Code Runbook (BM25 Reference)

| Error Code | Owning Service | Meaning & Probable Cause | Immediate Remediation |
| :--- | :--- | :--- | :--- |
| **`BETA-GW-502`** | `auth-gw` | **Upstream REST timeout or ECS task healthcheck failure**. | Check target group unhealthy host count in `ap-southeast-1` and inspect recent ECS task rollouts. |
| **`BETA-KAFKA-109`** | `billing-worker` | **Avro schema compatibility rejection** on topic `beta.billing.events`. | Verify Schema Registry version compatibility (`BACKWARD`) and replay records from `beta.dlq.billing`. |
| **`BETA-MYSQL-301`** | `auth-gw` / `billing` | **Aurora MySQL lock wait timeout exceeded** (`> 5s`) during batch invoicing. | Inspect long-running transactions via `SHOW PROCESSLIST`; throttle `billing-worker` batch concurrency. |
| **`BETA-MEM-404`** | Shared / Cache | **Memcached slab eviction spike or key prefix violation**. | Check ElastiCache `evictions` metric and audit recent PRs for oversized payloads (`> 512 KB`). |