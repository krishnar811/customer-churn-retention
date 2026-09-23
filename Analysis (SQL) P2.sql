-- Databricks notebook source
USE CATALOG customer_churn_retention;
USE SCHEMA dataset;


-- COMMAND ----------


SELECT 'customers' AS table_name, COUNT(*) AS row_count FROM customers
UNION ALL
SELECT 'monthly_activity', COUNT(*) FROM monthly_activity
UNION ALL
SELECT 'churn_labels', COUNT(*) FROM churn_labels
UNION ALL
SELECT 'support_tickets', COUNT(*) FROM support_tickets;



-- COMMAND ----------


SELECT 
    is_churned, 
    COUNT(*) AS cnt,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM customers
GROUP BY is_churned;



-- COMMAND ----------

SELECT * FROM monthly_activity LIMIT 5;


-- COMMAND ----------

DESCRIBE monthly_activity;

-- COMMAND ----------

SELECT
    customer_id,
    activity_month,
    total_spend,
    LAG(total_spend) OVER (PARTITION BY customer_id ORDER BY activity_month) AS prev_month_spend,
    ROUND(total_spend - LAG(total_spend) OVER (PARTITION BY customer_id ORDER BY activity_month), 2) AS mom_change,
    ROUND(100.0 * (total_spend - LAG(total_spend) OVER (PARTITION BY customer_id ORDER BY activity_month))
          / NULLIF(LAG(total_spend) OVER (PARTITION BY customer_id ORDER BY activity_month), 0), 2) AS mom_pct_change
FROM monthly_activity
ORDER BY customer_id, activity_month;

-- COMMAND ----------

WITH spend_with_lag AS (
    SELECT
        customer_id,
        activity_month,
        total_spend,
        LAG(total_spend, 1) OVER (PARTITION BY customer_id ORDER BY activity_month) AS spend_m1,
        LAG(total_spend, 2) OVER (PARTITION BY customer_id ORDER BY activity_month) AS spend_m2
    FROM monthly_activity
)
SELECT
    customer_id,
    activity_month,
    spend_m2 AS spend_2_months_ago,
    spend_m1 AS spend_1_month_ago,
    total_spend AS spend_this_month,
    CASE
        WHEN spend_m2 > spend_m1 AND spend_m1 > total_spend THEN 1
        ELSE 0
    END AS declining_3_months_flag
FROM spend_with_lag
WHERE spend_m2 IS NOT NULL
ORDER BY customer_id, activity_month;

-- COMMAND ----------

WITH customer_summary AS (
    SELECT
        customer_id,
        MAX(activity_month) AS last_active_month,
        SUM(transaction_count) AS total_transactions,
        SUM(total_spend) AS total_spend
    FROM monthly_activity
    GROUP BY customer_id
),
rfm_base AS (
    SELECT
        customer_id,
        DATEDIFF(month, last_active_month, '2025-12-01') AS recency_months,
        total_transactions AS frequency,
        total_spend AS monetary
    FROM customer_summary
)
SELECT
    customer_id,
    recency_months,
    frequency,
    monetary,
    NTILE(4) OVER (ORDER BY recency_months DESC) AS recency_score,     
    NTILE(4) OVER (ORDER BY frequency ASC) AS frequency_score,         
    NTILE(4) OVER (ORDER BY monetary ASC) AS monetary_score           
FROM rfm_base
ORDER BY monetary DESC;

-- COMMAND ----------

WITH latest_3_months AS (
    SELECT
        customer_id,
        activity_month,
        total_spend,
        rewards_redeemed_flag,
        customer_support_calls,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY activity_month DESC) AS month_rank
    FROM monthly_activity
),
aggregated AS (
    SELECT
        customer_id,
        SUM(CASE WHEN month_rank <= 3 THEN total_spend ELSE 0 END) AS spend_last_3_months,
        SUM(CASE WHEN month_rank <= 3 THEN rewards_redeemed_flag ELSE 0 END) AS rewards_redeemed_last_3_months,
        SUM(CASE WHEN month_rank <= 3 THEN customer_support_calls ELSE 0 END) AS support_calls_last_3_months
    FROM latest_3_months
    GROUP BY customer_id
)
SELECT
    customer_id,
    spend_last_3_months,
    rewards_redeemed_last_3_months,
    support_calls_last_3_months,
    CASE
        WHEN spend_last_3_months < 500 AND rewards_redeemed_last_3_months = 0 AND support_calls_last_3_months >= 1
        THEN 'HIGH_RISK'
        WHEN spend_last_3_months < 1000 AND rewards_redeemed_last_3_months = 0
        THEN 'MEDIUM_RISK'
        ELSE 'LOW_RISK'
    END AS risk_segment
FROM aggregated
ORDER BY spend_last_3_months ASC;

-- COMMAND ----------

SELECT
    cl.is_churned,
    st.reason,
    COUNT(*) AS ticket_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY cl.is_churned), 2) AS pct_of_tickets
FROM support_tickets st
JOIN churn_labels cl ON st.customer_id = cl.customer_id
GROUP BY cl.is_churned, st.reason
ORDER BY cl.is_churned DESC, ticket_count DESC;

-- COMMAND ----------

WITH customer_month_rank AS (
    SELECT
        ma.customer_id,
        cl.is_churned,
        ma.total_spend,
        ROW_NUMBER() OVER (PARTITION BY ma.customer_id ORDER BY ma.activity_month DESC) AS months_before_end
    FROM monthly_activity ma
    JOIN churn_labels cl ON ma.customer_id = cl.customer_id
)
SELECT
    is_churned,
    months_before_end,
    ROUND(AVG(total_spend), 2) AS avg_spend
FROM customer_month_rank
WHERE months_before_end <= 6
GROUP BY is_churned, months_before_end
ORDER BY is_churned, months_before_end;

-- COMMAND ----------

SELECT
    c.card_type,
    COUNT(*) AS total_customers,
    SUM(c.is_churned) AS churned_customers,
    ROUND(100.0 * SUM(c.is_churned) / COUNT(*), 2) AS churn_rate_pct
FROM customers c
GROUP BY c.card_type
ORDER BY churn_rate_pct DESC;

-- COMMAND ----------

WITH customer_total_spend AS (
    SELECT
        c.customer_id,
        c.card_type,
        SUM(ma.total_spend) AS lifetime_spend
    FROM customers c
    JOIN monthly_activity ma ON c.customer_id = ma.customer_id
    GROUP BY c.customer_id, c.card_type
),
ranked AS (
    SELECT
        *,
        DENSE_RANK() OVER (PARTITION BY card_type ORDER BY lifetime_spend DESC) AS spend_rank
    FROM customer_total_spend
)
SELECT customer_id, card_type, lifetime_spend
FROM ranked
WHERE spend_rank = 3   -- 3rd highest spender per card type
ORDER BY card_type;

-- COMMAND ----------

