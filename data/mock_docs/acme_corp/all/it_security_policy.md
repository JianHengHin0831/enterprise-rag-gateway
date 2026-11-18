---
doc_id: acme_it_security
tenant_id: acme_corp
allowed_roles: [all]
title: Acme Information Security & Acceptable Use Policy
---
# Acme Information Security & Acceptable Use Policy (2026 Edition)

This policy applies to every Acme Corp employee, intern, and contractor who uses company devices, accounts, or networks. Policy owner: Security Operations Center (SOC), `security-soc@acme-corp.example`.

---

## 1. Accounts & Multi-Factor Authentication

* Every Acme account must have **multi-factor authentication (MFA)** enabled with the approved authenticator app. SMS codes are **not** accepted as a second factor since January 2026.
* Passwords must be at least **14 characters**. Acme does not force periodic rotation; a password is changed only after a suspected compromise or when the SOC requests it.
* Shared accounts are prohibited. Service accounts must be registered in the access catalog with a named human owner.

---

## 2. Device Rules

* Company laptops must run full-disk encryption and the managed endpoint agent. Disabling the agent is a policy violation (`SEC-DEV-03`).
* The screen must lock automatically after **5 minutes** of inactivity.
* Personal devices may access email and chat only through the managed mobile app; they may **not** connect to the engineering VPN.
* A lost or stolen laptop or phone must be reported to the SOC hotline `+1-555-0199` within **1 hour** of discovery.

---

## 3. Data Classification

| Class | Examples | Allowed storage | External sharing |
| :--- | :--- | :--- | :--- |
| **Public** | Press releases, published docs | Anywhere | Allowed |
| **Internal** | Handbook, team wikis | Company drive, wiki | Not allowed without manager approval |
| **Confidential** | Compensation data, architecture docs | Restricted drive folders | Not allowed without VP + SOC approval |
| **Restricted** | Customer data, credentials, keys | Approved vaults only | Never |

* Uploading Confidential or Restricted data to unapproved AI tools or personal cloud storage is prohibited (`SEC-DATA-07`).

---

## 4. Phishing & Incident Reporting

* Forward suspicious emails with the **Report Phish** button; do not click links or open attachments first.
* The IT Help Desk will **never** ask for your password or MFA code by email, chat, or phone.
* Security incidents of any size are reported to the SOC immediately. The SOC acknowledges P1 reports within **15 minutes**, 24/7.

---

## 5. Software Installation

* Only software from the self-service catalog may be installed. Requests for new software go through the IT ticket queue and take **3 business days** for a security review.
* Browser extensions are limited to the approved allowlist published on the IT wiki.
