# Customer Churn & Retention Analysis

A churn prediction project simulating how a credit card company identifies cardmembers likely to 
stop using their card, using SQL, PySpark (Databricks), and machine learning.

---

## Business Problem

Customers often reduce card usage gradually before cancelling entirely, and by the time they 
cancel, it's usually too late to intervene. The goal of this project is to identify at-risk 
customers early, using behavioral signals like declining spend, reduced engagement, and support 
complaints, so retention efforts can be targeted before churn happens.

---

## Dataset

A realistic, relational, synthetic dataset simulating 12 months of customer card activity.

| Table | Rows | Description |
|---|---|---|
| `customers` | 3,000 | Demographics, card type, signup date, churn status |
| `monthly_activity` | 33,102 | Monthly spend, transaction count, rewards redemption, support calls, late payments |
| `churn_labels` | 3,000 | Clean churn target table (is_churned, churn_month) |
| `support_tickets` | 6,130 | Ticket-level detail (reason, resolved flag) |

**Churn rate: 17.93%** — set to reflect realistic card/subscription churn benchmarks.

A gradual decline pattern was built into the data for churned customers: spend, transaction count, 
and rewards engagement decline over the 3-4 months before churn, while support calls and late 
payments increase slightly — mirroring real early-warning churn signals rather than a random drop-off.

*Note: This is synthetic data designed to mimic realistic customer behavior. It is used to 
demonstrate methodology and approach, not to represent real-world churn statistics.*

---

## Tech Stack

- **SQL** (Databricks SQL)
- **PySpark** (Databricks) — feature engineering
- **Python** (pandas, scikit-learn) — model training and evaluation
- **Power BI** — dashboard *(in progress — see below)*

---

## Approach

### 1. SQL Analysis
Built 8 investigative SQL queries (`sql/churn_queries.sql`) covering:
- Month-over-month spend change per customer using `LAG()`
- Flagging customers with 3 consecutive months of declining spend
- RFM (Recency, Frequency, Monetary) segmentation using `NTILE()` quartiles
- A combined at-risk cohort query using declining spend, no rewards redemption, and support 
  calls together (multi-signal risk segmentation)
- Support ticket reason breakdown, comparing churned vs. active customers
- Average spend trajectory comparison (churned vs. active) over the months leading up to churn
- Card type churn rate comparison
- Top spenders per card type using `DENSE_RANK()`

### 2. Feature Engineering (PySpark)
Built customer-level features from each customer's own historical monthly activity:
- Average spend and transaction count over the last 6 months
- Total support calls, rewards redemptions, and late payments over the last 6 months
- **Spend trend ratio**: average spend in the most recent 3 months vs. the prior 3 months 
  (values below 1.0 indicate declining spend) — this operationalizes the SQL "declining spend" 
  signal as a numeric feature for the model
- Transaction count trend ratio, calculated the same way

A random train/test split was used at the customer level, since each customer's features are 
already derived only from their own historical data (not from other customers or future 
information), so there is no time-based leakage risk here — unlike the fraud detection project, 
which required a strict time-based split.

### 3. Modeling
Trained and compared two classification models:
- **Logistic Regression** — with `class_weight='balanced'` to account for the 17.93% churn rate
- **Random Forest** — with `class_weight='balanced'`, using 200 trees

Both models were evaluated using precision, recall, and F1-score. Feature importance was reviewed 
using the Random Forest's built-in importance scores to identify which behaviors most strongly 
signal churn risk.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | *(add your numbers)* | | | |
| Random Forest | *(add your numbers)* | | | |

*(Fill in from your model_comparison_metrics.csv output)*

**Top features driving churn predictions:** *(add your top features from feature_importance.csv)*

---

## Power BI Dashboard 

A dashboard is being built to present the SQL and model results in a business-facing format, 
including:
- Customer spend trajectory: churned vs. active customers over time
- Churn rate by card type
- RFM segment distribution
- Model comparison (accuracy/precision/recall/F1 by model)



---

## What I'd Do With More Time
- Add SHAP for individual-customer-level explanation of churn risk drivers
- Test additional models (e.g., XGBoost) for comparison
- Incorporate a time-based validation window to simulate predicting churn ahead of time in production
- Validate the approach against real (non-synthetic) customer data

---

## Repository Structure
```
customer-churn-retention/
├── README.md
├── sql/
│   └── churn_queries.sql
├── pyspark/
│   └── feature_engineering.py
├── modeling/
│   └── model_training.py
```
