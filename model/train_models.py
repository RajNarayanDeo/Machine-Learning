"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier
  4. Gaussian Naive Bayes
  5. Random Forest (Ensemble)

For each model it computes: Accuracy, AUC, Precision, Recall, F1, MCC
It saves:
  - Trained model objects (.pkl) in the model/ folder
  - The fitted StandardScaler (scaler.pkl)
  - metrics.json (comparison table data, used by README + app)
  - test_data.csv (holdout test split, saved to project root, used for Streamlit upload)

Run with:  python model/train_models.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
# Breast Cancer Wisconsin (Diagnostic) dataset:
#   - 569 instances (>= 500 required)
#   - 30 numeric features (>= 12 required)
#   - Binary classification target: malignant (0) / benign (1)
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
TARGET_COL = "target"

print(f"Dataset shape: {df.shape}")
print(f"Class balance:\n{df[TARGET_COL].value_counts()}\n")

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features (helps Logistic Regression / KNN converge & perform well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Save test data (used for the Streamlit "upload CSV" feature)
# ---------------------------------------------------------------------------
test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values
project_root = Path(__file__).resolve().parent.parent
test_df.to_csv(project_root / "test_data.csv", index=False)
print(f"Saved test_data.csv with shape {test_df.shape}")

# ---------------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
}

# ---------------------------------------------------------------------------
# 5. Train, evaluate, and save each model
# ---------------------------------------------------------------------------
model_dir = Path(__file__).resolve().parent
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Some models expose predict_proba for AUC; fall back to decision_function if needed
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results[name] = metrics
    print(f"{name}: {metrics}")

    # Save model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(model_dir / f"{fname}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save the scaler (needed at inference time in the Streamlit app)
with open(model_dir / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names (so the app can validate uploaded CSVs)
with open(model_dir / "feature_names.json", "w") as f:
    json.dump(list(X.columns), f)

# Save metrics comparison table
with open(model_dir / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll models, scaler, and metrics saved to the model/ folder.")
