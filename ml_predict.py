"""
ml_predict.py
-------------
Command-line tool for phishing detection using the trained ML model only.

Usage:
    python ml_predict.py
    (then paste email text and press Ctrl+D / Ctrl+Z to submit)
"""

import os
import sys
import joblib

MODEL_PATH      = "models/phishing_xgboost_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print(f"❌  Model not found at '{MODEL_PATH}'. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(VECTORIZER_PATH):
        print(f"❌  Vectorizer not found at '{VECTORIZER_PATH}'. Run train_model.py first.")
        sys.exit(1)
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(email_text: str, model, vectorizer):
    vec   = vectorizer.transform([email_text])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return label, proba


def main():
    model, vectorizer = load_artifacts()

    print("\n📧  ML-Based Phishing Detector")
    print("    Paste email text, then press Ctrl+D (Unix) / Ctrl+Z (Windows) to submit.")
    print("    Type 'exit' on a blank line to quit.\n")

    while True:
        print("─" * 50)
        lines = []
        try:
            while True:
                line = input()
                if line.strip().lower() == "exit":
                    print("👋  Exiting.")
                    sys.exit(0)
                lines.append(line)
        except EOFError:
            pass

        email_text = "\n".join(lines).strip()
        if not email_text:
            print("⚠️   No text received. Try again.\n")
            continue

        label, proba = predict(email_text, model, vectorizer)
        result = "🚨 PHISHING" if label == 1 else "✅ LEGIT"

        print("\n─── ML PHISHING DETECTION RESULT ───")
        print(f"  Prediction  : {result}")
        print(f"  Confidence  → Phishing: {proba[1]*100:.2f}%  |  Legit: {proba[0]*100:.2f}%")
        print("─────────────────────────────────────\n")

        again = input("Analyse another email? [y/N]: ").strip().lower()
        if again != "y":
            print("👋  Exiting.")
            break


if __name__ == "__main__":
    main()
