---
doc_id: acme_eng_architecture
tenant_id: acme_corp
allowed_roles: [eng]
title: Core System Architecture v2
---
# Core system architecture v2

## Services

The order service (order-svc), payment service (pay-svc) and user service (user-svc) talk to each other over internal gRPC. Externally they are exposed as REST through the API gateway.

## Data storage

Orders and payments use a PostgreSQL 16 primary/replica cluster with read/write splitting. User profiles are stored in ClickHouse. The cache layer is a Redis Cluster, and every key is prefixed with the service name.

## Release policy

Production releases happen on Tuesdays and Thursdays, 14:00-17:00, after at least 24 hours of observation in staging. Error code E-4031 means the payment callback signature check failed; check the key rotation status of pay-svc.
