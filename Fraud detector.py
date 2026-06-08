# ==========================================
# HEALTH INSURANCE FRAUD DETECTION MODEL
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_excel("Medical_Insurance_Claim_Fraud_Detection_Dataset.xlsx")

print("Dataset Loaded Successfully")
print(df.shape)

# mean & max values of numeric columns
print(df.describe())

# ==========================================
# REMOVE EMPTY COLUMNS
# ==========================================

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nShape after removing empty columns:")
print(df.shape)

# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# DROP CLAIM ID
# ==========================================

df = df.drop("claim_id", axis=1)

# ==========================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================

encoder = LabelEncoder()

categorical_columns = [
    "patient_gender",
    "hospital_type",
    "treatment_category",
    "diagnosis_code"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("fraud_label", axis=1)

y = df["fraud_label"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# FRAUD DISTRIBUTION
# ==========================================

print("\nFraud Distribution:")
print(y.value_counts())

print("\nFraud Percentage:")
print(y.value_counts(normalize=True) * 100)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================================
# SMOTE BALANCING
# ==========================================

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# ==========================================
# MODEL TRAINING
# ==========================================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ==========================================
# FRAUD PROBABILITY PREDICTION
# ==========================================

y_prob = model.predict_proba(X_test)[:, 1]

# Lower threshold to catch more frauds

y_pred = (y_prob >= 0.30).astype(int)

print("\nSample Fraud Probabilities:")
print(y_prob[:20])

# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features")

print(importance)

# ==========================================
# SAVE PREDICTIONS
# ==========================================

results = X_test.copy()

results["Actual"] = y_test.values

results["Fraud_Probability"] = y_prob

results["Predicted"] = y_pred

results.to_csv(
    "fraud_predictions.csv",
    index=False
)

print("\nPredictions Saved Successfully")

print("\nFile Created: fraud_predictions.csv")

# ==========================================
# MISSED FRAUD CASES (FALSE NEGATIVES)
# ==========================================

missed_frauds = results[
    (results["Actual"] == 1) &
    (results["Predicted"] == 0)
]

print("\nMissed Fraud Cases:")
print(missed_frauds)

print("\nTotal Missed Frauds:")
print(len(missed_frauds))

missed_frauds.to_csv(
    "missed_fraud_cases.csv",
    index=False
)

print("\nFile Created: missed_fraud_cases.csv")



# ==========================================
# DETECTED FRAUD CASES (TRUE POSITIVES)
# ==========================================

detected_frauds = results[
    (results["Actual"] == 1) &
    (results["Predicted"] == 1)
]

print("\nTotal Detected Frauds:")
print(len(detected_frauds))

detected_frauds.to_csv(
    "detected_fraud_cases.csv",
    index=False
)

print("\nFile Created: detected_fraud_cases.csv")
# ==========================================
# HIGH RISK CLAIMS
# ==========================================

high_risk = results[
    results["Fraud_Probability"] >= 0.70
]

print("\nHigh Risk Claims Found:")
print(len(high_risk))

high_risk.to_csv(
    "high_risk_claims.csv",

    index=False
)

print("\nFile Created: high_risk_claims.csv")