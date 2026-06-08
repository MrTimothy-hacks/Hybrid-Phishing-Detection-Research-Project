"""
hybrid_weighted_predict.py
--------------------------
Hybrid phishing detector — Weighted Strategy.

  final_score = (0.8 × ML_probability) + (0.2 × rule_binary)

  If final_score >= 0.4  →  phishing
  Otherwise              →  legit

Usage:
    python hybrid_weighted_predict.py
"""

import os
import sys
import joblib
from rule_based import score_email

MODEL_PATH      = "models/phishing_xgboost_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

ML_WEIGHT   = 0.8
RULE_WEIGHT = 0.2
THRESHOLD   = 0.4


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print(f"❌  Model not found at '{MODEL_PATH}'. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(VECTORIZER_PATH):
        print(f"❌  Vectorizer not found at '{VECTORIZER_PATH}'.")
        sys.exit(1)
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def predict_weighted(text: str, model, vectorizer):
    """
    Returns
    -------
    ml_proba    : float   ML phishing probability
    rule_score  : int     Rule-based score (0–10)
    rule_pred   : str     "phishing" or "legit"
    rule_reasons: list
    final_score : float   Weighted combined score
    final_pred  : str     "phishing" or "legit"
    """
    # ML component
    ml_proba = model.predict_proba(vectorizer.transform([text]))[0][1]

    # Rule-based component
    rule_score, rule_pred, rule_reasons = score_email(text)
    rule_binary = 1.0 if rule_pred == "phishing" else 0.0

    # Combine
    final_score = (ML_WEIGHT * ml_proba) + (RULE_WEIGHT * rule_binary)
    final_pred  = "phishing" if final_score >= THRESHOLD else "legit"

    return ml_proba, rule_score, rule_pred, rule_reasons, final_score, final_pred


def main():
    model, vectorizer = load_artifacts()

    print("\n📧  Hybrid Phishing Detector — Weighted Strategy (80% ML + 20% Rules)")
    print("    Paste email text, then Ctrl+D (Unix) / Ctrl+Z (Windows) to submit.\n")

    try:
        input_text = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\n🔵  Input cancelled.")
        sys.exit(0)

    if not input_text:
        print("⚠️   No input received.")
        sys.exit(1)

    ml_proba, rule_score, rule_pred, rule_reasons, final_score, final_pred = \
        predict_weighted(input_text, model, vectorizer)

    result = "🚨 PHISHING" if final_pred == "phishing" else "✅ LEGIT"

    print("\n──── HYBRID PREDICTION RESULT (Weighted) ────")
    print(f"  🤖 ML probability (phishing) : {ml_proba:.4f}")
    print(f"  🔺 Rule-based score          : {rule_score}/10  →  {rule_pred.upper()}")
    print(f"  📊 Final weighted score      : {final_score:.4f}")
    print(f"  🔷 Final prediction          : {result}")
    print("──────────────────────────────────────────────")

    if rule_reasons:
        print("\n⚠️  Rule-Based Triggers:")
        for r in rule_reasons:
            print(f"    • {r}")
    else:
        print("\n✅  Rule-Based Triggers: None detected")


if __name__ == "__main__":
    main()
