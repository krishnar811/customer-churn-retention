# Databricks notebook source

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

spark.sql("USE CATALOG customer_churn_retention")
spark.sql("USE SCHEMA dataset")

# COMMAND ----------

monthly_activity = spark.table("monthly_activity")
customers        = spark.table("customers")
churn_labels     = spark.table("churn_labels")
 
monthly_activity = monthly_activity.withColumn(
    "activity_month", F.to_date("activity_month")
)

# COMMAND ----------

window_by_customer = Window.partitionBy("customer_id").orderBy(F.desc("activity_month"))
 
monthly_activity = monthly_activity.withColumn(
    "month_rank", F.row_number().over(window_by_customer)
)

# COMMAND ----------

recent_activity = monthly_activity.filter(F.col("month_rank") <= 6)

# COMMAND ----------

agg_features = recent_activity.groupBy("customer_id").agg(
    F.avg("total_spend").alias("avg_spend_last_6m"),
    F.sum("total_spend").alias("total_spend_last_6m"),
    F.avg("transaction_count").alias("avg_txn_count_last_6m"),
    F.sum("customer_support_calls").alias("total_support_calls_last_6m"),
    F.sum("rewards_redeemed_flag").alias("rewards_redemptions_last_6m"),
    F.sum("late_payment_flag").alias("late_payments_last_6m"),
    F.count("activity_month").alias("months_of_data_available")
)

# COMMAND ----------

recent_3m = (monthly_activity.filter(F.col("month_rank") <= 3)
             .groupBy("customer_id")
             .agg(F.avg("total_spend").alias("avg_spend_recent_3m"),
                  F.avg("transaction_count").alias("avg_txn_recent_3m")))
 
prior_3m = (monthly_activity.filter((F.col("month_rank") > 3) & (F.col("month_rank") <= 6))
            .groupBy("customer_id")
            .agg(F.avg("total_spend").alias("avg_spend_prior_3m"),
                 F.avg("transaction_count").alias("avg_txn_prior_3m")))
 
trend_features = recent_3m.join(prior_3m, on="customer_id", how="left")
 
trend_features = trend_features.withColumn(
    "spend_trend_ratio",
    F.when(F.col("avg_spend_prior_3m") > 0,
           F.col("avg_spend_recent_3m") / F.col("avg_spend_prior_3m")
    ).otherwise(1.0)   # 1.0 = no change; <1 means declining spend
).withColumn(
    "txn_trend_ratio",
    F.when(F.col("avg_txn_prior_3m") > 0,
           F.col("avg_txn_recent_3m") / F.col("avg_txn_prior_3m")
    ).otherwise(1.0)
)

# COMMAND ----------

features_df = (agg_features
               .join(trend_features.select("customer_id", "spend_trend_ratio", "txn_trend_ratio",
                                            "avg_spend_recent_3m"),
                     on="customer_id", how="left")
               .join(customers.select("customer_id", "age", "card_type", "home_state"),
                     on="customer_id", how="left")
               .join(churn_labels.select("customer_id", "is_churned"),
                     on="customer_id", how="left"))

# COMMAND ----------

features_df = features_df.fillna({
    "spend_trend_ratio": 1.0,
    "txn_trend_ratio": 1.0,
    "avg_spend_recent_3m": 0.0
})

# COMMAND ----------

features_df = features_df.withColumn("rand_val", F.rand(seed=42))
features_df = features_df.withColumn(
    "dataset_split",
    F.when(F.col("rand_val") <= 0.8, "train").otherwise("test")
).drop("rand_val")
 
print("Split sizes:")
features_df.groupBy("dataset_split").count().show()
 
print("\nChurn rate by split (should be similar):")
features_df.groupBy("dataset_split", "is_churned").count().show()

# COMMAND ----------

features_df.write.format("delta").mode("overwrite").saveAsTable("churn_features")
 
print(f"\nFeature table created: {features_df.count()} rows")
display(features_df.limit(10))

# COMMAND ----------

print("\n=== Feature sanity check: avg values by churn status ===")
display(features_df.groupBy("is_churned").agg(
    F.round(F.avg("spend_trend_ratio"), 3).alias("avg_spend_trend_ratio"),
    F.round(F.avg("txn_trend_ratio"), 3).alias("avg_txn_trend_ratio"),
    F.round(F.avg("total_support_calls_last_6m"), 2).alias("avg_support_calls"),
    F.round(F.avg("rewards_redemptions_last_6m"), 2).alias("avg_rewards_redemptions"),
    F.round(F.avg("late_payments_last_6m"), 2).alias("avg_late_payments")
))

# COMMAND ----------

