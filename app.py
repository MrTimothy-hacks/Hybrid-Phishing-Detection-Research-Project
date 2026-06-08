"""
app.py
------
Flask web application for the Hybrid Phishing Email Detection System.

Run with:
    python app.py

Then open your browser at:
    http://127.0.0.1:5000
"""

import os
import sys
import joblib
from flask import Flask, render_template, request, jsonify
from rule_based import score_email

app = Flask(__name__)

MODEL_PATH      = "models/phishing_xgboost_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# ── Load model & vectorizer once at startup ───────────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("❌  Model or vectorizer not found. Run train_model.py first.")
    sys.exit(1)

model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("✅  Model and vectorizer loaded successfully.")

ML_WEIGHT   = 0.8
RULE_WEIGHT = 0.2
THRESHOLD   = 0.4


def analyse_email(text: str) -> dict:
    """Run weighted hybrid detection and return a results dict."""
    # ML component
    ml_proba = model.predict_proba(vectorizer.transform([text]))[0][1]

    # Rule-based component
    rule_score, rule_pred, rule_reasons = score_email(text)
    rule_binary = 1.0 if rule_pred == "phishing" else 0.0

    # Weighted combination
    final_score = (ML_WEIGHT * ml_proba) + (RULE_WEIGHT * rule_binary)
    is_phishing = final_score >= THRESHOLD

    return {
        "verdict":       "phishing" if is_phishing else "safe",
        "final_score":   round(float(final_score) * 100, 1),
        "ml_confidence": round(float(ml_proba) * 100, 1),
        "rule_score":    int(rule_score),
        "rule_pred":     rule_pred,
        "reasons":       rule_reasons,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.get_json()
    text = (data or {}).get("email_text", "").strip()

    if not text:
        return jsonify({"error": "No email text provided."}), 400

    result = analyse_email(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
