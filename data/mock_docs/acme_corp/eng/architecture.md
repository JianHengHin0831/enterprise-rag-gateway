---
doc_id: acme_eng_architecture
tenant_id: acme_corp
allowed_roles: [eng]
title: 核心系统架构说明 v2
---
# 核心系统架构说明 v2

## 服务拆分

订单服务（order-svc）、支付服务（pay-svc）、用户服务（user-svc）通过内部 gRPC 通信，对外统一经 API 网关暴露 REST 接口。

## 数据存储

订单与支付使用 PostgreSQL 16 主从集群，读写分离；用户画像存储在 ClickHouse。缓存层使用 Redis Cluster，Key 统一以服务名为前缀。

## 发版规范

生产发版窗口为每周二、周四 14:00–17:00，必须先经过 staging 环境 24 小时观察。错误码 E-4031 表示支付回调签名校验失败，需检查 pay-svc 的密钥轮换状态。
