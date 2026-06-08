"""
evaluate_rule_based.py
----------------------
Evaluates the standalone rule-based engine against the cleaned dataset.

Input : emails/cleaned_ealvaradob_emails.csv
"""

import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from rule_based import score_email

INPUT_CSV = "emails/cleaned_ealvaradob_emails.csv"


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌  File not found: {INPUT_CSV}. Run preprocess.py first.")
        return

    print(f"\n📂  Loading dataset from '{INPUT_CSV}' ...")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["email_text", "label"])
    df = df[df["email_text"].apply(lambda x: isinstance(x, str))]

    print(f"📊  Evaluating {len(df)} emails ...\n")

    y_true = df["label"].astype(int).tolist()
    y_pred = []

    for text in df["email_text"]:
        _, prediction, _ = score_email(text)
        y_pred.append(1 if prediction == "phishing" else 0)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print("────────────────────────────────────────")
    print("  RULE-BASED MODEL EVALUATION")
    print("────────────────────────────────────────")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("────────────────────────────────────────\n")
    print("📋  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["legit", "phishing"]))


if __name__ == "__main__":
    main()
