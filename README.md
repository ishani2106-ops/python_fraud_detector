# Medical Insurance Claim Fraud Detection Dataset

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Rows](https://img.shields.io/badge/Records-10%2C000-blue)
![Features](https://img.shields.io/badge/Features-16-green)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-orange)

A structured dataset designed for **detecting fraudulent medical insurance claims** using machine learning. It contains 10,000 records with 16 features and a binary fraud label, suitable for classification, anomaly detection, and risk analysis tasks.

---

##  Dataset Overview

| Property | Details |
|---|---|
| **File** | `Medical_Insurance_Claim_Fraud_Detection_Dataset.csv` |
| **Records** | 10,000 |
| **Features** | 15 input features + 1 target label |
| **Task** | Binary Classification (Fraud / Not Fraud) |
| **Fraud Rate** | ~15% (1,502 fraudulent claims out of 10,000) |
| **Format** | CSV |

---

##  Feature Description

| Column | Type | Description |
|---|---|---|
| `claim_id` | String | Unique identifier for each insurance claim |
| `patient_age` | Integer | Age of the patient (18–89 years) |
| `patient_gender` | Categorical | Gender of the patient (`Male`, `Female`) |
| `hospital_type` | Categorical | Type of hospital (`Private`, `Government`) |
| `treatment_category` | Categorical | Nature of treatment (`Emergency`, `Surgery`, `Therapy`, `Consultation`) |
| `diagnosis_code` | String | Medical diagnosis code associated with the claim |
| `claim_amount` | Float | Total amount claimed (₹5,046 – ₹4,99,950) |
| `approved_amount` | Float | Amount approved by the insurance provider |
| `hospital_stay_days` | Integer | Duration of hospital stay in days |
| `previous_claims_count` | Integer | Number of claims filed by the patient previously |
| `policy_tenure_years` | Integer | Number of years the insurance policy has been active |
| `claim_submission_delay_days` | Integer | Days between treatment and claim submission |
| `high_risk_procedure_flag` | Binary | 1 if the procedure is flagged as high risk, else 0 |
| `document_mismatch_flag` | Binary | 1 if submitted documents have inconsistencies, else 0 |
| `anomaly_score` | Float | Computed anomaly score between 0.0 and 1.0 |
| `fraud_label` | Binary | **Target variable** — 1 = Fraudulent, 0 = Legitimate |

---

##  Key Statistics

| Feature | Min | Mean | Max |
|---|---|---|---|
| `patient_age` | 18 | 53.5 | 89 |
| `claim_amount` | ₹5,046 | ₹2,53,039 | ₹4,99,950 |
| `hospital_stay_days` | 1 | 15.0 | 29 |
| `anomaly_score` | 0.00 | 0.50 | 1.00 |

### Class Distribution

| Label | Count | Percentage |
|---|---|---|
| 0 — Legitimate | 8,498 | 85.0% |
| 1 — Fraudulent | 1,502 | 15.0% |

>  **Note:** The dataset is imbalanced. Consider using techniques like SMOTE, class weighting, or stratified sampling when training models.

---

##  Potential Use Cases

- **Fraud Detection Models** — Train supervised classifiers (Logistic Regression, Random Forest, XGBoost, Neural Networks)
- **Anomaly Detection** — Use `anomaly_score` and flag features for unsupervised analysis
- **Risk Scoring** — Build risk scoring pipelines for insurance claim review
- **Feature Engineering Practice** — Explore interaction features between claim amount, delay days, and document mismatch
- **Imbalanced Classification Research** — Benchmark oversampling/undersampling strategies

---

##  Getting Started

### Load the Dataset

```python
import pandas as pd

df = pd.read_csv("Medical_Insurance_Claim_Fraud_Detection_Dataset.csv")
print(df.shape)       # (10000, 16)
print(df.head())
```

### Quick EDA

```python
# Class distribution
print(df['fraud_label'].value_counts(normalize=True))

# Correlation with fraud
print(df.corr(numeric_only=True)['fraud_label'].sort_values(ascending=False))

# Summary stats
print(df.describe())
```

### Example: Training a Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Encode categoricals
df_enc = df.copy()
for col in ['patient_gender', 'hospital_type', 'treatment_category', 'diagnosis_code', 'claim_id']:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

X = df_enc.drop('fraud_label', axis=1)
y = df_enc['fraud_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
```

---

##  Repository Structure

```
├── Medical_Insurance_Claim_Fraud_Detection_Dataset.csv   # Main dataset
├── README.md                                              # Project documentation
└── notebooks/                                             # (optional) EDA & model notebooks
```

---

##  License

This dataset is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it for personal, academic, or commercial purposes with attribution.

---

##  Acknowledgements

This dataset was synthetically generated for educational and research purposes in the domain of healthcare fraud analytics and insurance risk modeling.

---

##  If you find this dataset useful, please consider starring the repository!
