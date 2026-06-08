"""
evaluate_ml_model.py
--------------------
Evaluates the trained XGBoost ML model on the cleaned dataset
and plots a confusion matrix.

Input : emails/cleaned_ealvaradob_emails.csv
        models/phishing_xgboost_model.pkl
        models/vectorizer.pkl
"""

import os
import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

INPUT_CSV       = "emails/cleaned_ealvaradob_emails.csv"
MODEL_PATH      = "models/phishing_xgboost_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def main():
    # ── Check files ───────────────────────────────────────────────────────────
    for path in [INPUT_CSV, MODEL_PATH, VECTORIZER_PATH]:
        if not os.path.exists(path):
            print(f"❌  File not found: {path}")
            sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"📂  Loading dataset from '{INPUT_CSV}' ...")
    df = pd.read_csv(INPUT_CSV).dropna(subset=["email_text"])

    print("🔍  Loading trained ML model and vectorizer ...")
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X = df["email_text"]
    y = df["label"].astype(int)

    # ── Predict ───────────────────────────────────────────────────────────────
    print(f"⚙️   Running predictions on {len(df)} emails ...\n")
    X_vec  = vectorizer.transform(X)
    y_pred = model.predict(X_vec)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy  = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall    = recall_score(y, y_pred)
    f1        = f1_score(y, y_pred)

    print("────────────────────────────────────────")
    print("  ML MODEL (XGBoost) EVALUATION")
    print("────────────────────────────────────────")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("────────────────────────────────────────\n")
    print("📋  Classification Report:")
    print(classification_report(y, y_pred, target_names=["legit", "phishing"]))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Phishing"],
        yticklabels=["Legit", "Phishing"]
    )
    plt.title("Confusion Matrix — ML Model (XGBoost)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix_ml.png", dpi=150)
    print("📊  Confusion matrix saved → 'confusion_matrix_ml.png'")
    plt.show()


if __name__ == "__main__":
    main()
