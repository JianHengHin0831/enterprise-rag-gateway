---
doc_id: acme_hr_compensation
tenant_id: acme_corp
allowed_roles: [hr]
title: Compensation Structure and Performance Review Standards (Confidential)
---
# Compensation Structure and Performance Review Standards

> **Confidential — HR Only (`allowed_roles: [hr]`)**. All personal identifiers, phone numbers, and ID numbers in this document are fictional synthetic test data. Do not expose compensation tables or PII in application logs or non-HR channels.

---

## 1. Salary Bands & Annual Bonus Structure

### 1.1 Base Salary Bands by Level (Monthly, Pre-Tax, USD)
Base salaries are paid across 12 monthly payroll cycles. New offers should normally fall between the **25th and 75th percentile** of the level band unless approved by the VP of HR (`HR-COMP-01`).

| Level | Typical Role | Min (USD/mo) | Midpoint (USD/mo) | Max (USD/mo) |
| :--- | :--- | :--- | :--- | :--- |
| **P5** | Engineer / Analyst | 6,000 | 7,300 | 8,600 |
| **P6** | Senior Engineer / Lead | 8,300 | 10,450 | 12,600 |
| **P7** | Staff / Principal / Manager | 12,000 | 15,150 | 18,300 |

### 1.2 Annual Bonus & Merit Review
* **Target Bonus**: Full-time employees who complete their **3-month probation** (see *Acme Employee Handbook*) are eligible for a performance-linked annual bonus paid in March:
  * Rating **S**: **4.0 months** of base salary
  * Rating **A**: **2.5 months** of base salary
  * Rating **B**: **1.5 months** of base salary
  * Rating **C**: **0 months** (no annual bonus and ineligible for annual merit raise)
* **Merit Increase Cap**: Standard annual merit increases take effect April 1 and range from 3%–5% for `B`, 6%–10% for `A`, and 11%–15% for `S`. Out-of-cycle adjustments require VP HR + CFO co-approval under policy **`HR-COMP-01`**.

---

## 2. Performance Reviews & PIP Policy

### 2.1 Rating Tiers & Forced Distribution
Acme conducts formal quarterly performance calibrations (Jan, Apr, Jul, Oct) and an annual summary review. Ratings have four tiers with a mandatory department-level forced distribution:

| Tier | Description | Forced Distribution | Bonus Multiplier |
| :--- | :--- | :--- | :--- |
| **S** | Exceptional impact (top performers) | **10%** | 4.0 months |
| **A** | Exceeds expectations | **30%** | 2.5 months |
| **B** | Meets expectations | **50%** | 1.5 months |
| **C** | Needs improvement / Below bar | **10%** | 0 months |

* **Probation Exclusion**: Employees still within their initial 3-month probation period during a quarterly calibration receive `N/A` and are excluded from the 10% / 30% / 50% / 10% forced distribution pool.
* **Promotion Eligibility**: Promotion to the next P-level requires at least **12 months in current level** and either **one `S` rating** or **two consecutive `A` ratings** within the past four quarters.

### 2.2 Performance Improvement Plan (PIP)
* Employees rated **C for two consecutive quarters** automatically enter a formal **60-day Performance Improvement Plan (PIP)**.
* **PIP Restrictions (`HR-PIP-02`)**: During an active PIP, the employee is ineligible for internal team transfers, promotions, or annual bonus payouts, and undergoes bi-weekly milestone evaluations with their manager and HRBP.
* Passing all PIP milestones restores standard `B`-level standing; failing to meet milestones results in involuntary termination with statutory/policy severance.

---

## 3. Severance & Offboarding Payout Policy

### 3.1 N+1 Calculation Formula
For involuntary separations (restructuring, role elimination, or post-PIP separation without gross misconduct), severance follows the **N+1** formula:
* **`N` (Years of Service)**: Total completed years of continuous employment at Acme Corp. Partial years of **6 months or more count as a full year (`+1.0`)**; partial years under 6 months count as **`+0.5` year**.
* **`+1` (Notice-in-Lieu)**: 1 additional month of base salary paid when immediate offboarding is requested instead of a 30-day working notice period.
* **Wage Base**: Calculated using the employee's average monthly gross base salary over the preceding 12 months (excluding one-off spot bonuses).
* **Unused Annual Leave Cash-Out**: Consistent with the *Acme Employee Handbook*, unused accrued annual leave (excluding expired carryover days) is cashed out upon involuntary separation alongside the N+1 package.

### 3.2 Worked Calculation Example (For HR Verification)
* **Scenario**: A P6 Senior Engineer with a monthly base salary of **$10,000 USD** and **3 years and 7 months** of service is impacted by a team restructuring with immediate offboarding.
* **Calculation**: Because 7 months is `>= 6 months`, `N` rounds up to **4**. Total severance multiplier is `N + 1 = 5 months`. Severance payout = `5 × $10,000 = $50,000 USD` (plus any prorated unused annual leave balance).

---

## 4. Internal HR Policy Codes & Escalations

* **`HR-COMP-01`**: Off-cycle salary increase or above-band offer exception (requires VP HR + CFO sign-off).
* **`HR-PIP-02`**: Active PIP freeze on internal transfers, merit raises, and bonus eligibility.
* **`HR-SEV-03`**: Mutual separation agreement template requiring Legal review prior to employee signature.

---

## 5. HR Contacts & Escalation Directory (Fictional Test PII)

* **Alex Zhang** (HRBP, Engineering): mobile `13800138000`, email `alex.zhang.hr@acme.example`, ID number `110101199001011234`
* **Sam Li** (Compensation Specialist): mobile `13900139000`, email `sam.li.comp@acme.example`
* **Morgan Chen** (Employee Relations & PIP Lead): mobile `13700137000`, email `morgan.chen.er@acme.example`, ID number `310101198805054321`