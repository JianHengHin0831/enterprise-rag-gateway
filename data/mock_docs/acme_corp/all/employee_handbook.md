---
doc_id: acme_handbook
tenant_id: acme_corp
allowed_roles: [all]
title: Acme Employee Handbook (2026 edition)
---
# Acme Employee Handbook (2026 Edition)

Welcome to Acme Corp. This handbook outlines company-wide workplace policies, leave entitlements, IT security standards, and employee benefits. For role-specific engineering standards, refer to the Engineering Architecture Guide; for salary bands and bonus formulas, contact HR or refer to the restricted HR Compensation Policy.

---

## 1. Working Hours & Hybrid Work Policy

### 1.1 Standard & Flexible Hours
Acme uses flexible working hours. Standard full-time employees work **8 hours per day** (excluding a 1-hour lunch break), totaling 40 hours per week.
* **Core Hours**: **10:00–16:00** local time. All employees must be reachable on Slack and available for collaboration during core hours.
* **Flex Window**: Employees may start between 08:00 and 10:00 and finish between 17:00 and 19:00 accordingly.
* **Wednesday "No-Meeting Day"**: Internal recurring meetings are not scheduled on Wednesdays so teams have uninterrupted deep-work time. Exceptions require VP approval or must be urgent production incident bridges.

### 1.2 Hybrid & Remote Work
* **Full-Time Employees (Post-Probation)**: Eligible for up to **2 remote workdays per week** (Tuesday and Thursday are recommended team in-office anchor days, though teams may adjust with manager approval).
* **Probation & Interns**: Employees during their probation period and interns are expected to work **onsite 5 days a week**, unless prior written approval is granted by their manager and HR.
* **Work From Anywhere (WFA)**: Employees with over 1 year of tenure may apply for up to **10 business days** of temporary remote work outside their home city per calendar year, subject to IT security clearance and tax-compliance review.

### 1.3 Overtime & Compensatory Time Off (TOIL)
Overtime must be pre-approved by a direct manager in the HR portal. Approved weekend or public-holiday work for non-exempt staff is compensated either at statutory overtime rates or as **1:1.5 Time Off in Lieu (TOIL)**, which must be used within **90 days** of accrual.

---

## 2. Leave & Time-Off Policies

### 2.1 Annual Leave
* **Full-Time Employees**: Receive **10 days** of paid annual leave after completing their first year of service, plus **1 additional day** for each subsequent year, capped at a maximum of **15 days** per year. During the first year (post-probation), annual leave accrues pro-rata at 0.83 days per month.
* **Carryover Rule**: Up to **5 unused annual leave days** may be carried over into the next calendar year, but carried-over days expire automatically on **March 31** if unused. Unused leave cannot be cashed out except upon involuntary separation or as required by local law.
* **Interns**: Interns do not receive paid annual leave, but may request **1 day of unpaid personal leave per month** with at least 48 hours' notice to their mentor.

### 2.2 Sick Leave
* Sick leave requests must be submitted in the HR portal **before 10:00 on the same day** (or as soon as reasonably possible in medical emergencies).
* **Medical Certificate Requirement**: Sick leave lasting **longer than 2 consecutive days** requires an official doctor's note uploaded to the HR portal within 3 business days of return.
* **Paid Sick Allowance**: Full-time employees are entitled to **5 days of 100% paid sick leave** per calendar year. Additional certified sick leave is paid at 70% of base salary up to 30 days.

### 2.3 Other Leave Types
* **Marriage Leave**: **3 consecutive business days** of paid leave for full-time employees (must be taken within 6 months of marriage registration).
* **Parental Leave**: Primary caregivers receive **16 weeks** of paid parental leave; secondary caregivers receive **4 weeks** of paid leave, usable within 12 months of birth or adoption.
* **Bereavement Leave**: Up to **5 paid days** for immediate family members (spouse, parents, children, siblings).

---

## 3. Onboarding, Probation & Workplace Conduct

### 3.1 Probation Period
All full-time new hires undergo a **3-month probation period**. Managers conduct a formal midpoint check-in at Day 45 and a final confirmation review at Day 80. Probation may be extended once, by up to **3 additional months**, only with HR Business Partner sign-off.

### 3.2 Badges & Physical Security
* Access badges must be worn visibly inside all Acme offices and never lent to colleagues or visitors.
* Lost badges must be reported immediately to Facilities (`facilities@acme-corp.example`). The first replacement badge is free; subsequent replacements incur a **$25 administrative fee**.
* Visitors must be registered at reception 24 hours in advance and escorted at all times inside secured zones.

### 3.3 Acceptable Use & AI Confidentiality
Employees must never upload Acme proprietary source code, customer PII, or unreleased financial figures to unapproved public generative AI websites. Only the internal **Acme Enterprise AI Gateway** (which enforces data retention zero-logging and role-based access control) is approved for internal document queries.

---

## 4. Office Network, VPN & IT Troubleshooting

### 4.1 Wi-Fi Networks
* **Corporate Network (`ACME-Office`)**: For company-managed laptops and authorized work devices only. The Wi-Fi password rotates **every quarter** (Jan 1, Apr 1, Jul 1, Oct 1) and is posted in the **"IT Notices"** section of the HR portal home page.
* **Guest Network (`ACME-Guest`)**: Intended for visitors and personal phones. It requires **no password** (click-through captive portal), is isolated from internal servers, and is bandwidth-limited to **10 Mbps**.

### 4.2 VPN & Remote Access
Remote access to internal dashboards, Git repositories, and staging environments requires **Acme-VPN (WireGuard)** authenticated via corporate Okta SSO + Hardware/App MFA.
* Idle VPN sessions automatically disconnect after **4 hours**.
* Split-tunneling is enabled for video conferencing but disabled for all `*.internal.acme` traffic.

### 4.3 Common IT Error Codes (Self-Service Quick Reference)
Before opening an IT ticket, check the error code shown on your client:
* **`ERR-VPN-401`**: Okta MFA token expired or system clock drift exceeds 60 seconds. Sync your OS time and re-authenticate via the Okta Verify app.
* **`ERR-VPN-429`**: Too many concurrent session attempts within 5 minutes. Wait 10 minutes before retrying, or switch to the backup gateway (`vpn-west.acme-corp.example`).
* **`ERR-WIFI-204`**: Device certificate expired following quarterly rotation. Re-fetch the latest Wi-Fi profile from `"IT Notices"` on the HR portal.
* **`ERR-SSO-503`**: Identity provider maintenance window or upstream timeout. Check `#it-status` or contact the IT Helpdesk.

### 4.4 Password & Device Policy
* Corporate passwords must be at least **14 characters** long and managed via the company password manager (1Password Enterprise).
* Company laptops must have full-disk encryption (FileVault / BitLocker) enabled and auto-lock after **5 minutes** of inactivity.

---

## 5. Employee Benefits & Allowances

### 5.1 Commuter & Wellness Stipend
* **Commuter Allowance**: Full-time employees receive **$80/month** toward public transit or office parking; interns receive **$40/month**. (Note: Taxi reimbursements for late-night overtime after 21:00 are claimed separately under the *Expense Policy*.)
* **Annual Health Checkup**: Full-time employees are eligible for up to **$300/year** for an annual preventive health screening at approved partner clinics.

### 5.2 Learning & Development (L&D) Budget
Post-probation full-time employees receive an annual **$500 L&D budget** for technical books, online courses, or conference tickets relevant to their role.
* Pre-approval from your manager is required for single purchases above **$150**.
* **Exclusions**: The L&D budget does not cover hardware accessories, failed certification exam retake fees, or non-work hobby classes.

---

## 6. Internal Support Contacts

| Issue Type | Contact Channel | SLA / Availability |
| :--- | :--- | :--- |
| IT & VPN Support | `it-helpdesk@acme-corp.example` / Ext. `4400` | Mon–Fri, 09:00–18:30 (Urgent P1: 24/7) |
| HR & Leave Queries | `hr-support@acme-corp.example` / Ext. `4200` | Mon–Fri, 09:30–18:00 |
| Facilities & Badges | `facilities@acme-corp.example` / Ext. `4100` | Mon–Fri, 09:00–18:00 |
| Security Incidents | `security-soc@acme-corp.example` / Hotline `+1-555-0199` | 24/7 Immediate Response |

> **Scope Note**: Detailed travel/meal reimbursement caps and receipt submission rules are documented in the *Acme Expense Policy*. Compensation bands, equity vesting, and annual bonus formulas are restricted to HR leadership and are not covered in this handbook.