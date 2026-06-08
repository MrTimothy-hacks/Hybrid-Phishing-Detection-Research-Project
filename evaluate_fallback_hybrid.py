"""
evaluate_fallback_hybrid.py
---------------------------
Evaluates the Fallback Hybrid Strategy on the balanced dataset
and plots a confusion matrix.

  ML prob >= 0.6  →  phishing  (ML trusted)
  ML prob <= 0.4  →  legit     (ML trusted)
  0.4 < prob < 0.6  →  rule-based engine decides

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

HIGH_CONF = 0.6
LOW_CONF  = 0.4


def main():
    for path in [INPUT_CSV, MODEL_PATH, VECTORIZER_PATH]:
        if not os.path.exists(path):
            print(f"❌  File not found: {path}")
            sys.exit(1)

    print(f"📂  Loading dataset from '{INPUT_CSV}' ...")
    data = pd.read_csv(INPUT_CSV).dropna(subset=["email_text"])

    print("🔍  Loading trained ML model and vectorizer ...")
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    x = data["email_text"]
    y = data["label"].astype(int)

    print(f"⚙️   Generating fallback hybrid predictions on {len(data)} emails ...\n")

    predictions  = []
    fallback_count = 0

    for text in x:
        ml_proba = model.predict_proba(vectorizer.transform([text]))[0][1]

        if ml_proba >= HIGH_CONF:
            predictions.append(1)
        elif ml_proba <= LOW_CONF:
            predictions.append(0)
        else:
            _, rule_pred, _ = score_email(text)
            predictions.append(1 if rule_pred == "phishing" else 0)
            fallback_count += 1

    print(f"ℹ️   Fallback (rule-based) triggered for {fallback_count} emails "
          f"({fallback_count/len(data)*100:.1f}%)\n")

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy  = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall    = recall_score(y, predictions)
    f1        = f1_score(y, predictions)

    print("────────────────────────────────────────────")
    print("  HYBRID MODEL EVALUATION  (Fallback Strategy)")
    print("────────────────────────────────────────────")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("────────────────────────────────────────────\n")
    print("📋  Classification Report:")
    print(classification_report(y, predictions, target_names=["legit", "phishing"]))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Phishing"],
        yticklabels=["Legit", "Phishing"]
    )
    plt.title("Confusion Matrix — Fallback Hybrid Strategy")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix_fallback.png", dpi=150)
    print("📊  Confusion matrix saved → 'confusion_matrix_fallback.png'")
    plt.show()


if __name__ == "__main__":
    main()
