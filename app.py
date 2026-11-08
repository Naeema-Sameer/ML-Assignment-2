
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")

st.title("🛡️ PhishGuard AI")
st.subheader("Machine Learning Based Phishing Website Detection")
st.write("Evaluate phishing detection models using labelled test data.")

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "feature_columns.json", "r") as file:
    feature_columns = json.load(file)

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

uploaded_file = st.file_uploader(
    "Upload labelled test dataset (CSV)",
    type=["csv"]
)

selected_model = st.selectbox(
    "Select a machine learning model",
    list(model_files.keys())
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "label" not in data.columns:
        st.error("The uploaded CSV must contain a 'label' column.")
    elif not set(feature_columns).issubset(data.columns):
        st.error("The uploaded CSV is missing required input features.")
    else:
        data = data.dropna(subset=["label"]).copy()

        X_test = data[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        y_test = (data["label"] == 0).astype(int)

        model_path = MODEL_DIR / model_files[selected_model]
        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        try:
            scores = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            scores = model.decision_function(X_test)

        st.success(f"Evaluation completed using {selected_model}")

        auc = roc_auc_score(y_test, scores)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision_score(y_test, predictions, zero_division=0):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall_score(y_test, predictions, zero_division=0):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, predictions, zero_division=0):.4f}")
        col6.metric("MCC", f"{matthews_corrcoef(y_test, predictions):.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual Legitimate", "Actual Phishing"],
                columns=["Predicted Legitimate", "Predicted Phishing"]
            )
        )

        st.subheader("Classification Report")
        report = classification_report(
            y_test, predictions,
            target_names=["Legitimate", "Phishing"],
            output_dict=True,
            zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Upload the labelled test CSV to evaluate a selected model.")
    st.write("The application loads previously trained models. It does not retrain them.")
