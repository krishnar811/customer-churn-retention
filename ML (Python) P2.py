# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd
import numpy as np

# COMMAND ----------

spark.sql("USE CATALOG customer_churn_retention")
spark.sql("USE SCHEMA dataset")

# COMMAND ----------

features_pd = spark.table("churn_features").toPandas()
 
train_df = features_pd[features_pd['dataset_split'] == 'train'].copy()
test_df  = features_pd[features_pd['dataset_split'] == 'test'].copy()
 
print(f"Train: {len(train_df)} customers, churn rate: {train_df['is_churned'].mean()*100:.2f}%")
print(f"Test:  {len(test_df)} customers, churn rate: {test_df['is_churned'].mean()*100:.2f}%")

# COMMAND ----------

categorical_cols = ['card_type']
numeric_cols = [
    'avg_spend_last_6m', 'total_spend_last_6m', 'avg_txn_count_last_6m',
    'total_support_calls_last_6m', 'rewards_redemptions_last_6m',
    'late_payments_last_6m', 'spend_trend_ratio', 'txn_trend_ratio',
    'avg_spend_recent_3m', 'age'
]
 
train_encoded = pd.get_dummies(train_df[categorical_cols], drop_first=True)
test_encoded = pd.get_dummies(test_df[categorical_cols], drop_first=True)
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
 
X_train = pd.concat([train_df[numeric_cols].reset_index(drop=True),
                      train_encoded.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_df[numeric_cols].reset_index(drop=True),
                     test_encoded.reset_index(drop=True)], axis=1)
 
y_train = train_df['is_churned'].reset_index(drop=True)
y_test = test_df['is_churned'].reset_index(drop=True)

# COMMAND ----------

X_train['spend_trend_ratio'] = X_train['spend_trend_ratio'].clip(upper=5)
X_test['spend_trend_ratio'] = X_test['spend_trend_ratio'].clip(upper=5)
X_train['txn_trend_ratio'] = X_train['txn_trend_ratio'].clip(upper=5)
X_test['txn_trend_ratio'] = X_test['txn_trend_ratio'].clip(upper=5)
 
print(f"\nFinal feature count: {X_train.shape[1]}")

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
 
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
 
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)   
 
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
 
def evaluate(name, y_true, y_pred):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_true, y_pred, target_names=['Retained', 'Churned'], digits=3))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"                 Predicted Retained  Predicted Churned")
    print(f"Actual Retained  {cm[0][0]:>17}  {cm[0][1]:>17}")
    print(f"Actual Churned   {cm[1][0]:>17}  {cm[1][1]:>17}")
    return cm
 
lr_cm = evaluate("Logistic Regression", y_test, y_pred_lr)
rf_cm = evaluate("Random Forest", y_test, y_pred_rf)

# COMMAND ----------

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
 
print("\n=== Top Features Driving Churn Predictions (Random Forest) ===")
print(importance_df.head(10).to_string(index=False))

# COMMAND ----------

def get_metrics(name, y_true, y_pred):
    return {
        'model': name,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
        'f1_score': round(f1_score(y_true, y_pred), 4)
    }
 
comparison_df = pd.DataFrame([
    get_metrics('Logistic Regression', y_test, y_pred_lr),
    get_metrics('Random Forest', y_test, y_pred_rf)
])
print("\n=== Model Comparison ===")
print(comparison_df.to_string(index=False))

# COMMAND ----------

def assign_risk_tier(prob):
    if prob >= 0.6:
        return "HIGH_RISK - Immediate retention offer"
    elif prob >= 0.3:
        return "MEDIUM_RISK - Monitor + light engagement"
    else:
        return "LOW_RISK - No action needed"
 
action_df = pd.DataFrame({
    'customer_id': test_df['customer_id'].values,
    'actual_churned': y_test.values,
    'churn_probability': y_proba_rf,
    'risk_tier': [assign_risk_tier(p) for p in y_proba_rf]
})
 
print("\n=== Risk Tier Distribution ===")
print(action_df['risk_tier'].value_counts())
print("\n=== Actual churn rate within each tier (validates the tiering) ===")
print(action_df.groupby('risk_tier')['actual_churned'].mean())

# COMMAND ----------

spark.sql("CREATE VOLUME IF NOT EXISTS customer_churn_retention.dataset.models")

comparison_df.to_csv('/Volumes/customer_churn_retention/dataset/models/model_comparison_metrics.csv', index=False)
importance_df.to_csv('/Volumes/customer_churn_retention/dataset/models/feature_importance.csv', index=False)
action_df.to_csv('/Volumes/customer_churn_retention/dataset/models/churn_risk_predictions.csv', index=False)
 


# COMMAND ----------

