"""
train_model.py
--------------
Trains an XGBoost classifier on the balanced phishing email dataset.

Input  : emails/balanced_ealvaradob_emails.csv
Output : models/phishing_xgboost_model.pkl
         models/vectorizer.pkl
"""

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from xgboost import XGBClassifier

# ─── Paths ────────────────────────────────────────────────────────────────────

BALANCED_CSV    = "emails/balanced_ealvaradob_emails.csv"
MODEL_PATH      = "models/phishing_xgboost_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def main():
    print("\n══════════════════════════════════════════════════════════════")
    print("  TRAINING PHISHING DETECTION MODEL — XGBoost (Balanced Data)")
    print("══════════════════════════════════════════════════════════════\n")

    if not os.path.exists(BALANCED_CSV):
        print(f"❌  File not found: {BALANCED_CSV}")
        print("    Run balance_with_smote.py first.")
        return

    # ── Load Data ─────────────────────────────────────────────────────────────
    print(f"📂  Loading dataset from '{BALANCED_CSV}' ...")
    df = pd.read_csv(BALANCED_CSV)
    df.dropna(subset=["email_text", "label"], inplace=True)

    X_text = df["email_text"]
    y      = df["label"].astype(int)

    print(f"📊  Total samples : {len(df)}")
    print(f"📊  Distribution  :\n{y.value_counts()}\n")

    # ── TF-IDF Vectorisation ──────────────────────────────────────────────────
    print("🔢  Vectorising with TF-IDF (ngram_range=(1,3), max_features=5000) ...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=5000
    )
    X = vectorizer.fit_transform(X_text)

    # ── Train / Test Split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📐  Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}\n")

    # ── Model Training ────────────────────────────────────────────────────────
    print("🤖  Training XGBoost classifier ...")
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print("\n────────────────────────────────────────")
    print("  EVALUATION RESULTS (on Test Set 20%)")
    print("────────────────────────────────────────")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("────────────────────────────────────────\n")
    print("📋  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))

    # ── Save Model & Vectorizer ───────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"💾  Model saved      → '{MODEL_PATH}'")
    print(f"💾  Vectorizer saved → '{VECTORIZER_PATH}'")
    print("\n✅  Training complete!\n")


if __name__ == "__main__":
    main()
