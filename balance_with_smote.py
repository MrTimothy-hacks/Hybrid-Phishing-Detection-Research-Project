"""
balance_with_smote.py
---------------------
Balances the cleaned email dataset using SMOTE
(Synthetic Minority Oversampling Technique).

Input  : emails/cleaned_ealvaradob_emails.csv
Output : emails/balanced_ealvaradob_emails.csv
         models/vectorizer.pkl   (TF-IDF vectorizer saved for reuse)
"""

import os
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Paths ────────────────────────────────────────────────────────────────────

INPUT_CSV       = "emails/cleaned_ealvaradob_emails.csv"
OUTPUT_CSV      = "emails/balanced_ealvaradob_emails.csv"
VECTORIZER_PATH = "models/vectorizer.pkl"


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌  File not found: {INPUT_CSV}")
        print("    Run preprocess.py first.")
        return

    print(f"📂  Loading cleaned dataset from '{INPUT_CSV}' ...")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["email_text", "label"])

    print(f"📊  Original label distribution:\n{df['label'].value_counts()}\n")

    # ── Step 1: TF-IDF Vectorisation ──────────────────────────────────────────
    print("🔢  Vectorising with TF-IDF (max_features=5000) ...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["email_text"])
    y = df["label"].astype(int)

    # ── Step 2: Apply SMOTE ───────────────────────────────────────────────────
    print("⚖️   Applying SMOTE ...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # ── Step 3: Reconstruct DataFrame ─────────────────────────────────────────
    print("🔄  Reconstructing DataFrame from resampled vectors ...")
    email_texts_resampled = vectorizer.inverse_transform(X_resampled)
    text_data = [" ".join(tokens) for tokens in email_texts_resampled]

    df_balanced = pd.DataFrame({
        "email_text": text_data,
        "label":      y_resampled
    })

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"💾  Vectorizer saved  → '{VECTORIZER_PATH}'")

    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"💾  Balanced CSV saved → '{OUTPUT_CSV}'")

    print(f"\n📊  New label distribution:\n{df_balanced['label'].value_counts()}")
    print(f"\n✅  Total samples after SMOTE: {len(df_balanced)}")


if __name__ == "__main__":
    main()
