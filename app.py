"""
Streamlit App - Breast Cancer Classification Model Comparison
----------------------------------------------------------------
Features:
  a. CSV upload of test data
  b. Model selection dropdown (5 trained classification models)
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report

Run locally with: streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Classification Model Comparison",
    page_icon="🩺",
    layout="wide",
)

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

TARGET_COL = "target"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_scaler():
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(filename):
    with open(MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_names():
    with open(MODEL_DIR / "feature_names.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_precomputed_metrics():
    with open(MODEL_DIR / "metrics.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🩺 Breast Cancer Classification - Model Comparison App")
st.markdown(
    """
This app demonstrates **5 classification models** trained on the
Breast Cancer Wisconsin (Diagnostic) dataset (30 features, 569 instances,
binary classification: malignant vs benign).

Upload the provided `test_data.csv` (or any CSV with the same 30 feature
columns + a `target` column) to see live predictions and evaluation metrics.
"""
)

feature_names = load_feature_names()

# ---------------------------------------------------------------------------
# (a) Dataset upload
# ---------------------------------------------------------------------------
st.header("1. Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload test_data.csv (must contain the 30 feature columns + 'target' column)",
    type=["csv"],
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded yet — using the bundled sample test_data.csv for preview.")
    sample_path = Path(__file__).resolve().parent / "test_data.csv"
    test_df = pd.read_csv(sample_path) if sample_path.exists() else None

if test_df is not None:
    missing_cols = [c for c in feature_names if c not in test_df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()
    if TARGET_COL not in test_df.columns:
        st.error(f"Uploaded CSV must contain a '{TARGET_COL}' column with true labels.")
        st.stop()

    with st.expander("Preview uploaded data"):
        st.dataframe(test_df.head(10))

    X_test = test_df[feature_names]
    y_test = test_df[TARGET_COL]

    # ---------------------------------------------------------------------
    # (b) Model selection dropdown
    # ---------------------------------------------------------------------
    st.header("2. Select a Model")
    model_choice = st.selectbox("Choose a classification model:", list(MODEL_FILES.keys()))

    scaler = load_scaler()
    model = load_model(MODEL_FILES[model_choice])

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)

    # ---------------------------------------------------------------------
    # (c) Evaluation metrics on the uploaded data
    # ---------------------------------------------------------------------
    st.header("3. Evaluation Metrics")

    live_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, live_metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    with st.expander("See training-time metrics for ALL 5 models (comparison table)"):
        all_metrics = load_precomputed_metrics()
        st.dataframe(pd.DataFrame(all_metrics).T)

    # ---------------------------------------------------------------------
    # (d) Confusion matrix + classification report
    # ---------------------------------------------------------------------
    st.header("4. Confusion Matrix & Classification Report")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with c2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))

else:
    st.warning("Please upload a CSV file to continue, or add test_data.csv to the repo root.")

st.markdown("---")
st.caption(
    "Built for BITS Pilani WILP M.Tech (AIML/DSE) - Machine Learning Assignment 2."
)
