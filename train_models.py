
import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# 1. Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. Load dataset
print("Loading dataset...")
df = pd.read_csv("dataset.csv")

TARGET = "label"

if TARGET not in df.columns:
    raise ValueError("Target column 'label' not found.")

# Remove records with missing target values
df = df.dropna(subset=[TARGET]).copy()

# 3. Select numerical input features
X = df.select_dtypes(include=np.number).drop(
    columns=[TARGET], errors="ignore"
)

# Encode phishing as 1 and legitimate as 0
y = (df[TARGET] == 0).astype(int)

# Remove columns with no variation
X = X.loc[:, X.nunique(dropna=False) > 1]

if X.shape[1] < 12:
    raise ValueError(
        f"Only {X.shape[1]} usable numerical features found. "
        "Inspect the dataset before proceeding."
    )

if y.nunique() != 2:
    raise ValueError("This project requires exactly two target classes.")

print("Total usable records:", len(df))
print("Numerical input features:", X.shape[1])
print("Target distribution:")
print(y.value_counts())

# 4. Use a stratified sample for manageable training
MAX_RECORDS = 20000

if len(X) > MAX_RECORDS:
    X, _, y, _ = train_test_split(
        X, y,
        train_size=MAX_RECORDS,
        stratify=y,
        random_state=42
    )

print("\nRecords selected:", len(X))

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Save feature names for the application
with open("models/feature_columns.json", "w") as f:
    json.dump(X.columns.tolist(), f, indent=4)

# Save test data for Streamlit evaluation
test_data = X_test.copy()
test_data[TARGET] = y_test
test_data.to_csv("results/test_data.csv", index=False)

# 6. Define five machine learning models
model_definitions = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "SVM": LinearSVC(
        max_iter=5000, random_state=42
    )
}

results = []

# 7. Train and evaluate each model
for name, classifier in model_definitions.items():

    print("\n" + "=" * 55)
    print("Training:", name)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    start_time = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = pipeline.predict(X_test)

    # Obtain scores for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions),
        "Training Time (seconds)": training_time
    }

    results.append(metrics)

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, os.path.join("models", filename))

    # Save confusion matrix and classification report
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    with open(
        os.path.join("results", filename.replace(".joblib", ".json")),
        "w"
    ) as f:
        json.dump({
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist(),
            "classification_report": report
        }, f, indent=4)

    print("Accuracy:", round(metrics["Accuracy"], 4))
    print("AUC:", round(metrics["AUC"], 4))
    print("F1:", round(metrics["F1"], 4))
    print("MCC:", round(metrics["MCC"], 4))
    print("Training time:", round(training_time, 2), "seconds")

# 8. Save model comparison
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nAll five models trained and saved successfully!")
