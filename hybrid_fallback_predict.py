"""
hybrid_fallback_predict.py
--------------------------
Hybrid phishing detector — Fallback Strategy.

  If ML confidence >= 0.6  →  trust ML  (phishing)
  If ML confidence <= 0.4  →  trust ML  (legit)
  If 0.4 < ML confidence < 0.6  →  defer to rule-based engine

Usage:
    python hybrid_fallback_predict.py
"""

import os
import sys
import joblib
from rule_based import score_email

MODEL_PATH      = "models/phishing_xgboost_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

HIGH_CONF = 0.6
LOW_CONF  = 0.4


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print(f"❌  Model not found at '{MODEL_PATH}'. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(VECTORIZER_PATH):
        print(f"❌  Vectorizer not found at '{VECTORIZER_PATH}'.")
        sys.exit(1)
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def predict_fallback(text: str, model, vectorizer):
    """
    Returns
    -------
    ml_proba      : float   ML phishing probability
    final_pred    : str     "phishing" or "legit"
    fallback_used : bool    Whether rule-based fallback was triggered
    rule_score    : int|None
    rule_pred     : str|None
    rule_reasons  : list
    """
    ml_proba = model.predict_proba(vectorizer.transform([text]))[0][1]

    if ml_proba >= HIGH_CONF:
        return ml_proba, "phishing", False, None, None, []
    elif ml_proba <= LOW_CONF:
        return ml_proba, "legit",    False, None, None, []
    else:
        # Ambiguous — defer to rule-based
        rule_score, rule_pred, rule_reasons = score_email(text)
        return ml_proba, rule_pred, True, rule_score, rule_pred, rule_reasons


def main():
    model, vectorizer = load_artifacts()

    print("\n📧  Hybrid Phishing Detector — Fallback Strategy")
    print("    Paste email text, then Ctrl+D (Unix) / Ctrl+Z (Windows) to submit.\n")

    try:
        input_text = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\n🔵  Input cancelled.")
        sys.exit(0)

    if not input_text:
        print("⚠️   No input received.")
        sys.exit(1)

    ml_proba, final_pred, fallback_used, rule_score, rule_pred, rule_reasons = \
        predict_fallback(input_text, model, vectorizer)

    result = "🚨 PHISHING" if final_pred == "phishing" else "✅ LEGIT"

    print("\n──── HYBRID PREDICTION RESULT (Fallback) ────")
    print(f"  🤖 ML probability (phishing) : {ml_proba:.4f}")

    if fallback_used:
        print(f"  🔺 Rule-based score          : {rule_score}/10  →  {rule_pred.upper()}")
        print(f"     (Fallback triggered — ML confidence was ambiguous)")
    else:
        print(f"  🔒 Rule-based system         : Not used (ML was confident)")

    print(f"  🔷 Final prediction          : {result}")
    print("─────────────────────────────────────────────")

    if fallback_used and rule_reasons:
        print("\n⚠️  Rule-Based Triggers:")
        for r in rule_reasons:
            print(f"    • {r}")
    elif fallback_used:
        print("\n✅  Rule-Based Triggers: None detected")


if __name__ == "__main__":
    main()
