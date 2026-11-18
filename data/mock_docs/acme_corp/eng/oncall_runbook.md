---
doc_id: acme_eng_oncall
tenant_id: acme_corp
allowed_roles: [eng]
title: Acme Engineering On-Call & Incident Runbook
---
# Acme Engineering On-Call & Incident Runbook (Restricted: Engineering)

> **Restricted — Engineering only.** Contains internal hostnames and escalation paths. All hostnames are fictional examples.

---

## 1. On-Call Rotation

* Rotations are **one week long**, Monday 10:00 to Monday 10:00 local time, with a primary and a secondary engineer per service group.
* Engineers join the rotation after completing **3 shadow shifts**. Interns never hold the pager.
* On-call compensation: **150 USD per weekday** and **300 USD per weekend day or public holiday** on the primary pager; the secondary receives half of these amounts.
* Swaps must be recorded in the paging tool at least **24 hours** in advance.

---

## 2. Severity Levels

| Severity | Definition | Ack target | Update cadence |
| :--- | :--- | :--- | :--- |
| **SEV1** | Full outage of checkout or login, or data loss | 5 minutes | Every 30 minutes |
| **SEV2** | Partial outage or error rate above 5% for 10 minutes | 15 minutes | Every 60 minutes |
| **SEV3** | Degraded non-critical feature, workaround exists | 4 business hours | Daily |

* If the primary does not acknowledge a SEV1 page within **5 minutes**, it escalates to the secondary, then to the Engineering Manager after another 10 minutes.

---

## 3. Incident Procedure

1. Acknowledge the page and open an incident channel named `#inc-YYYYMMDD-short-name`.
2. Assign an **Incident Commander** (IC). The IC coordinates and does not debug.
3. Mitigate first: roll back the last deploy, fail over, or disable the feature flag before root-causing.
4. Post status updates to `#status-internal` at the cadence in the table above.
5. Close the incident only after the error rate has been normal for **30 minutes**.

---

## 4. Rollback & Deploy Freeze

* Every service must support a one-command rollback: `deployctl rollback <service> --to previous`.
* A **deploy freeze** applies from the last Friday of December to January 2 and during any open SEV1. Exceptions need VP Engineering approval.

---

## 5. Postmortems

* SEV1 and SEV2 incidents require a blameless postmortem within **5 business days**.
* Postmortems are reviewed in the Thursday reliability review. Action items get an owner and a due date in the tracker.
