---
doc_id: beta_finance_vendor_payment
tenant_id: beta_inc
allowed_roles: [finance]
title: Beta Vendor Payment & Treasury Compliance Policy (Confidential)
---
# Beta Inc. Vendor Payment & Treasury Compliance Policy

> **Confidential — Finance Only (`allowed_roles: [finance]`)**. Classified as **Tier-1 Restricted Financial Data**. All corporate bank accounts, routing numbers, tax IDs, and personal contacts below are fictional synthetic test data. Do not cache raw banking records in shared caches or log unmasked identifiers.

---

## 1. Scope: Vendor Payments vs. Employee Expenses

This policy governs payments to **external third-party vendors, contractors, and SaaS suppliers**. (Internal employee travel and meal reimbursements are governed separately by the *Beta Employee Handbook* and paid on the 1st and 15th payroll cycle.)

---

## 2. Vendor Onboarding & Bank Change Verification (KYV)

### 2.1 Mandatory Onboarding Documents
Before any new vendor can be added to the `ops.beta-inc.example` procurement ledger, AP Finance must collect:
1. Signed Master Service Agreement (MSA) or vendor order form;
2. Valid Tax Registration / W-8BEN-E or W-9 certificate;
3. Official bank account verification letter on bank letterhead;
4. Signed Beta Anti-Bribery & Sanctions Compliance Declaration.

### 2.2 Bank Account Change Callback Rule (`FIN-PAY-01`)
* If an existing vendor requests a change to their receiving bank account via email, AP staff must **never** approve the change based on email alone.
* Under protocol **`FIN-PAY-01`**, Treasury must complete a live verification callback using the **vendor's originally registered phone number on file** (never the phone number provided in the change-request email) and record a callback log ID before updating payment details.

---

## 3. Purchase Orders & Payment Approval Matrix

### 3.1 PO & Dual-Control Thresholds
* **Up to $2,500 USD** (`<= 2,500`): No formal Purchase Order (PO) required; Cost Center Owner + AP review sufficient.
* **$2,500.01 to $10,000 USD** (`2,500 < amount <= 10,000`): **Mandatory pre-approved PO** in `ops.beta-inc.example` + Finance Controller sign-off.
* **Above $10,000 USD** (`> 10,000`): Mandatory PO + **CFO approval** required before scheduling payment.
* **Dual-Control Wire Rule (U-Key)**: Every bank wire transfer exceeding **$5,000 USD** requires **two-person treasury authorization** in the corporate banking portal (one **Maker** initiates, one independent **Checker** approves via hardware U-Key). Wires above **$25,000 USD** require **CFO + CEO** co-signoff.

### 3.2 Payment Terms & Batch Schedule
* **Standard Terms**: **Net 30** from receipt of a valid, PO-matched tax invoice. Early-payment discounts (e.g., `2/10 Net 30`) require Treasury Manager sign-off.
* **Weekly Batch Run**: Approved vendor invoices are settled every **Thursday at 15:00** local time (distinct from employee payroll/expense cycles). Invoices approved after Wednesday 17:00 roll to the following Thursday.

---

## 4. Corporate Billing & Treasury Details (Fictional Test Data)

* **Legal Entity**: Beta Inc. (Fictional Test Entity)
* **Corporate Tax ID / EIN**: `TAX-BETA-99018273`
* **Primary USD Operating Bank**: Pacific Meridian Test Bank
* **Corporate Account Number**: `ACCT-8800-4411-9922`
* **SWIFT / Routing (Test)**: SWIFT `PMBKTESTXXX` / Routing `999000123`
* **Invoice Submission Email**: `ap-finance@beta-inc.example` (Ext. `8200`)

---

## 5. Finance Audit & Rejection Error Codes (BM25 Reference)

* **`FIN-ERR-401`**: Missing or unapproved Purchase Order (`PO`) ID for vendor invoice exceeding `$2,500 USD`.
* **`FIN-ERR-403`**: Vendor bank account mismatch or incomplete **`FIN-PAY-01`** callback verification. Payment frozen automatically.
* **`FIN-ERR-409`**: Duplicate vendor invoice number detected within a 90-day lookback window.
* **`FIN-ERR-422`**: Tax ID (`TAX-*`) mismatch or missing VAT/withholding tax breakdown on cross-border invoice.

---

## 6. Treasury Contacts (Fictional Test PII)

* **Victor Tan** (Treasury Manager): mobile `13300133000`, email `victor.tan.fin@beta-inc.example`, U-Key ID `UK-FIN-0912`
* **AP Finance Desk**: `ap-finance@beta-inc.example` / Ext. `8200` (Mon–Fri, 09:00–18:00)