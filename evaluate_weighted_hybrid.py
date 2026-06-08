"""
evaluate_weighted_hybrid.py
---------------------------
Evaluates the Weighted Hybrid Strategy (80% ML + 20% Rule-Based)
on the balanced dataset and plots a confusion matrix.

  final_score = (0.8 × ML_prob) + (0.2 × rule_binary)
  Threshold   = 0.4

Input : emails/balanced_ealvaradob_emails.csv
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
from rule_based import score_email

INPUT_CSV       = "emails/balanced_ealvaradob_emails.csv"
MODEL_PATH      = "models/phishing_xgboost_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

ML_WEIGHT   = 0.8
RULE_WEIGHT = 0.2
THRESHOLD   = 0.4


def main():
    for path in [INPUT_CSV, MODEL_PATH, VECTORIZER_PATH]:
        if not os.path.exists(path):
            print(f"❌  File not found: {path}")
            sys.exit(1)

    print(f"📂  Loading dataset from '{INPUT_CSV}' ...")
    df = pd.read_csv(INPUT_CSV)
    df = df[df["email_text"].notna()]

    print("🔍  Loading trained ML model and vectorizer ...")
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print(f"⚙️   Generating weighted hybrid predictions on {len(df)} emails ...\n")

    y_true       = df["label"].astype(int).tolist()
    hybrid_preds = []

    for text in df["email_text"]:
        ml_proba    = model.predict_proba(vectorizer.transform([text]))[0][1]
        _, rule_pred, _ = score_email(text)
        rule_binary = 1.0 if rule_pred == "phishing" else 0.0

        final_score = (ML_WEIGHT * ml_proba) + (RULE_WEIGHT * rule_binary)
        hybrid_preds.append(1 if final_score >= THRESHOLD else 0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy  = accuracy_score(y_true, hybrid_preds)
    precision = precision_score(y_true, hybrid_preds)
    recall    = recall_score(y_true, hybrid_preds)
    f1        = f1_score(y_true, hybrid_preds)

    print("────────────────────────────────────────────")
    print("  HYBRID MODEL EVALUATION  (Weighted 80/20)")
    print("────────────────────────────────────────────")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("────────────────────────────────────────────\n")
    print("📋  Classification Report:")
    print(classification_report(y_true, hybrid_preds, target_names=["legit", "phishing"]))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm     = confusion_matrix(y_true, hybrid_preds)
    labels = ["Legit", "Phishing"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Weighted Hybrid (80% ML + 20% Rules)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix_weighted.png", dpi=150)
    print("📊  Confusion matrix saved → 'confusion_matrix_weighted.png'")
    plt.show()


if __name__ == "__main__":
    main()
