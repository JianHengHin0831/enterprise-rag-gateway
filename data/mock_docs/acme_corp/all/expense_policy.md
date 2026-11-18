---
doc_id: acme_expense
tenant_id: acme_corp
allowed_roles: [all]
title: Acme Expense Reimbursement Process
---
# Acme Expense Reimbursement Process (2026 Edition)

This policy defines eligible business expenses, submission deadlines, approval thresholds, and travel allowances for all Acme Corp employees and interns.

---

## 1. Submission Deadlines & Reimbursement Cycle

### 1.1 The 30-Day Submission Rule
All business expenses must be submitted through the **"Expense Claims"** module of the HR portal within **30 calendar days** of the date incurred (or within 30 days of trip return for multi-day business travel).
* **Late Claims**: Claims submitted after 30 days are **automatically rejected** by the portal and are not accepted. Exceptions are granted only for documented medical hospitalization or extended leave, with Finance VP sign-off.
* **Year-End Cutoff**: December expenses must be submitted no later than **December 20** (or within 10 days for travel between Dec 21–31) to close the fiscal year ledger.

### 1.2 Payout Schedule
* Claims fully approved by the **25th of the month** are reimbursed in the next monthly payroll cycle (paid on the **10th of the following month**).
* Claims approved after the 25th roll over to the subsequent month's payout.

---

## 2. Approval Chain & Anti-Splitting Rules

### 2.1 Approval Thresholds
Approval routing is determined by the **total amount of a single claim or single trip**:
* **Under $2,000 USD** (`< 2,000`): Approved by the **direct manager**.
* **$2,000 USD (inclusive) to $10,000 USD (inclusive)** (`2,000 <= amount <= 10,000`): Requires **direct manager + department director**.
* **Above $10,000 USD** (`> 10,000`): Requires **direct manager + department director + Finance VP**.

### 2.2 Pre-Approval & Anti-Splitting Policy
* **Mandatory Pre-Approval**: All international travel, conference sponsorships, and single non-travel purchases above **$2,000 USD** must be pre-approved in the HR portal before spending. (Note: L&D book/course purchases above **$150 USD** also require manager pre-approval under the *Employee Handbook*.)
* **No Split Invoicing**: Splitting a single purchase or vendor invoice into multiple smaller claims to bypass director or Finance VP approval is strictly prohibited and triggers an automatic compliance audit (`EXP-ERR-104`).

---

## 3. Business Travel Allowances

### 3.1 Hotel & Accommodation
* **Domestic Tier-1 Cities** (e.g., New York, San Francisco, Boston, Seattle, Los Angeles, Chicago): Up to **$200 USD per night** (excluding taxes).
* **Domestic Other Cities**: Up to **$140 USD per night** (excluding taxes).
* **International Travel**: Up to **$260 USD per night** (or the country-specific cap listed in the HR portal travel table).
* Employees exceeding nightly caps due to mandatory conference venue rates must attach screenshots of at least 2 nearby alternatives and obtain director sign-off prior to booking.

### 3.2 Flights & Ground Rail
* **Flights under 6 hours**: Economy class only. Seat-selection fees above $30 are personal expenses unless tied to a documented medical accommodation.
* **Flights 6 hours or longer**: Premium Economy is permitted with **department director pre-approval**. Business class is restricted to VP level and above, or documented medical necessity.
* **Rail**: Standard / Second-class seats for trips under 4 hours; First-class permitted for trips over 4 hours if priced below comparable economy airfare.

### 3.3 Meal Per Diem
* **Domestic Meal Allowance**: **$40 USD per day** — paid as a flat per diem and **needs no itemized receipt**.
* **International Meal Allowance**: **$65 USD per day** (no receipt required).
* **Proration Rule**: If a meal is already provided by a conference, airline, hotel breakfast package, or client event, the daily per diem is reduced by **25% for breakfast, 35% for lunch, and 40% for dinner** (or flat 50% if two or more meals are provided).

---

## 4. Overtime Taxi, Client Meals & Misc. Claims

### 4.1 Late-Night Overtime Taxi (Links to Working Hours Policy)
As noted in the *Acme Employee Handbook*, employees working approved overtime past **21:00** local time on weekdays (or on weekends/public holidays) may claim rideshare/taxi fare from the office to home:
* **Cap**: Up to **$50 USD per trip**.
* **Required Proof**: Electronic trip receipt showing pickup/dropoff locations, timestamp after 21:00, and matching overtime or incident ticket reference. Regular daytime commuting is covered by the monthly commuter stipend and cannot be claimed here.

### 4.2 Client Entertainment & Team Meals
* **Client Meals**: Capped at **$80 USD per attendee** (including tax and tip). Claims must list the full names, titles, and company names of all attendees plus the business purpose.
* **Alcohol**: Reimbursable only during external client hospitality with prior director approval; internal team-building alcohol expenses are limited to 15% of the total team dinner bill.
* **Internal Team Meals**: Requires manager approval and is capped at **$35 USD per employee** (once per quarter per team).

---

## 5. Receipts, Currency & Non-Reimbursable Items

### 5.1 Receipt & Foreign Currency Rules
* Itemized merchant receipts (showing vendor name, date, items, and total amount) are required for all claims **except** the daily meal per diem. Credit card slips showing only a lump sum are insufficient.
* **Lost Receipt Affidavit**: For unavoidable lost receipts under **$25 USD**, employees may submit a signed *Missing Receipt Declaration* in the portal (limited to **once per quarter**).
* **Foreign Currency**: Expenses in non-USD currencies are automatically converted using the HR portal's mid-market exchange rate on the transaction date, or reimbursed at the exact billed USD amount if a bank/card statement screenshot is attached.

### 5.2 Non-Reimbursable Expenses (Strictly Excluded)
Acme will **not** reimburse:
* Traffic citations, speeding tickets, or parking fines;
* Hotel minibar items, pay-per-view movies, spa/gym fees, or airline lounge passes;
* Personal travel extensions or expenses for accompanying family/guests;
* Unapproved SaaS subscriptions or personal hardware accessories (keyboards, mice, bags);
* Charitable donations or political contributions made on personal cards.

### 5.3 Common Portal Rejection Codes (BM25 Troubleshooting Reference)
* **`EXP-ERR-101`**: Missing itemized receipt or vendor tax invoice. Upload the full itemized bill rather than a payment confirmation screenshot.
* **`EXP-ERR-104`**: Potential split-claim violation detected (multiple claims to the same vendor within 48 hours). Combine into one claim and route to the appropriate approver tier.
* **`EXP-ERR-202`**: Claim submitted past the 30-day deadline. Contact Finance AP only if you hold a VP-signed medical/leave exception.
* **`EXP-ERR-305`**: Missing mandatory pre-approval ID for purchases > $2,000 USD or international travel.

---

## 6. Finance Contacts & Audit

* **Accounts Payable Helpdesk**: `finance-ap@acme-corp.example` / Ext. `4300` (Mon–Fri, 09:30–18:00).
* **Audit Notice**: Finance randomly audits **10% of approved claims** monthly. Falsified receipts or intentional policy circumvention may result in clawback and disciplinary action up to termination.