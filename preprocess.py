"""
preprocess.py
-------------
Cleans and prepares raw email data for training and evaluation.

Input  : emails/raw_emails.csv          (columns: Email Text, Email Type)
Output : emails/cleaned_ealvaradob_emails.csv
"""

import pandas as pd
import re
import os

INPUT_CSV  = "emails/raw_emails.csv"
OUTPUT_CSV = "emails/cleaned_ealvaradob_emails.csv"


def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)       # Remove HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)      # Keep letters only
    text = " ".join(text.lower().split())        # Lowercase + clean whitespace
    return text


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌  Input file not found: {INPUT_CSV}")
        return

    print(f"📂  Loading dataset from '{INPUT_CSV}' ...")
    df = pd.read_csv(INPUT_CSV)

    # Rename columns to standard names
    df = df.rename(columns={"Email Text": "email_text", "Email Type": "label"})

    # Convert labels: "Phishing Email" → 1, "Safe Email" → 0
    df["label"] = df["label"].map({"Phishing Email": 1, "Safe Email": 0})

    print(f"📊  Raw rows: {len(df)}")
    print(f"📊  Label distribution:\n{df['label'].value_counts()}\n")

    # Drop nulls and empty rows
    df = df.dropna(subset=["email_text", "label"])
    df = df[df["email_text"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

    # Clean text
    print("🧹  Cleaning email text ...")
    df["email_text"] = df["email_text"].apply(preprocess_text)

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["email_text"])
    print(f"🗑️   Duplicates removed: {before - len(df)}")

    df["label"] = df["label"].astype(int)

    print(f"\n✅  Clean rows: {len(df)}")
    print(f"📊  Final label distribution:\n{df['label'].value_counts()}\n")

    os.makedirs("emails", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾  Saved to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()
