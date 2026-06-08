"""
rule_based.py
-------------
Rule-Based Phishing Detection Engine.
Uses boolean logic and regular expressions to score emails
based on known phishing heuristics.

Scoring: 0–10. Threshold >= 2 → phishing.
"""

import re
import sys

# ─── Rule Definitions ────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    "suspended", "verify", "account update", "click below", "security alert",
    "limited access", "unusual activity", "update your info", "unauthorized login",
    "confirm password", "verify identity", "urgent attention required", "billing failure",
    "reset password", "your account has been compromised", "unrecognized device",
    "account locked", "login attempt", "action required", "unauthorized access",
    "immediate action required", "verify billing info", "click to resolve"
]

SUSPICIOUS_DOMAINS = [
    ".xyz", ".ru", ".tk", ".top", ".ml", ".ga", ".cn", ".work", ".support",
    "bit.ly", "tinyurl.com", "t.co", "shorturl.at", "goo.gl"
]

BRAND_SPOOF_PATTERNS = [
    ("paypal",    re.compile(r"paypal.*(account|verify|login|secure|update|payment)", re.IGNORECASE)),
    ("amazon",    re.compile(r"amazon.*(payment|invoice|verify|order|shipment)", re.IGNORECASE)),
    ("apple",     re.compile(r"apple.*(id|account|support|locked)", re.IGNORECASE)),
    ("bank",      re.compile(r"bank.*(account|verify|transaction|suspended|transfer)", re.IGNORECASE)),
    ("microsoft", re.compile(r"microsoft.*(account|security|verify|activity|login)", re.IGNORECASE)),
    ("facebook",  re.compile(r"facebook.*(account|verify|suspend|login|locked)", re.IGNORECASE)),
    ("google",    re.compile(r"google.*(account|verify|security|reset)", re.IGNORECASE)),
    ("netflix",   re.compile(r"netflix.*(payment|verify|login|cancelled|reactivate)", re.IGNORECASE)),
    ("zoom",      re.compile(r"zoom.*(meeting|cancelled|expired|reactivate)", re.IGNORECASE)),
]

SUSPICIOUS_LINK_PATTERNS = [
    re.compile(r"https?://[^ ]+/(login|verify|secure|update|reset|confirm)([ $])", re.IGNORECASE),
    re.compile(r"https?://(bit\.ly|tinyurl\.com|t\.co|shorturl\.at|goo\.gl)/[\w]+", re.IGNORECASE),
    re.compile(r"https?://[^ ]+\.(cn|ml|ga|ru|tk|xyz)/[\w]+", re.IGNORECASE),
    re.compile(r'href=["\']http.+?["\'].*?>.*?<', re.IGNORECASE),
]

GENERIC_GREETINGS = [
    "dear user", "dear customer", "valued customer", "hello user"
]

OBFUSCATED_EMAIL_PATTERNS = [
    re.compile(r"\[at\]"),
    re.compile(r"\[dot\]"),
    re.compile(r"\s+at\s+", re.IGNORECASE),
    re.compile(r"\s+dot\s+", re.IGNORECASE),
]

HTML_TRICKS = [
    re.compile(r"<form",               re.IGNORECASE),
    re.compile(r"display\s*:\s*none",  re.IGNORECASE),
    re.compile(r"<script",             re.IGNORECASE),
    re.compile(r"base64,",             re.IGNORECASE),
]

URGENT_PHRASES = [
    "your account will be closed", "your card has been charged", "final notice",
    "payment failed", "invoice attached", "click to pay", "update billing",
    "immediate attention required", "unauthorized transaction", "confirm your account"
]

SENDER_SPOOF_PATTERN = re.compile(
    r"from:.*?(support|admin|security).+?@(?!gmail\.com|yahoo\.com|outlook\.com)",
    re.IGNORECASE
)


# ─── Scoring Function ─────────────────────────────────────────────────────────

def score_email(text: str):
    """
    Score an email string for phishing indicators.

    Returns
    -------
    final_score : int   (0–10)
    prediction  : str   ("phishing" or "legit")
    reasons     : list  (human-readable triggers)
    """
    if not isinstance(text, str):
        return 0, "legit", ["Invalid email text"]

    score = 0
    reasons = []

    # 1. Suspicious keywords
    for kw in SUSPICIOUS_KEYWORDS:
        if kw.lower() in text.lower():
            score += 1
            reasons.append(f"Keyword detected: '{kw}'")

    # 2. Suspicious domains
    for domain in SUSPICIOUS_DOMAINS:
        if domain in text:
            score += 1
            reasons.append(f"Suspicious domain detected: '{domain}'")

    # 3. Suspicious link patterns
    for pattern in SUSPICIOUS_LINK_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            score += len(matches)
            reasons.append(
                f"Suspicious link pattern matched {len(matches)} time(s): '{pattern.pattern}'"
            )

    # 4. Brand spoofing
    for brand, pattern in BRAND_SPOOF_PATTERNS:
        if pattern.search(text) and brand not in text.lower():
            score += 2
            reasons.append(f"Possible spoofing of brand '{brand}' with untrusted domain")

    # 5. Generic greetings
    for greet in GENERIC_GREETINGS:
        if greet in text.lower():
            score += 1
            reasons.append(f"Generic greeting detected: '{greet}'")

    # 6. Obfuscated email addresses
    for pattern in OBFUSCATED_EMAIL_PATTERNS:
        if pattern.search(text):
            score += 1
            reasons.append("Obfuscated email address detected (e.g. '[at]', '[dot]')")
            break  # count once per email

    # 7. HTML tricks
    for pattern in HTML_TRICKS:
        if pattern.search(text):
            score += 1
            reasons.append(f"Suspicious HTML pattern detected: '{pattern.pattern}'")

    # 8. Urgency phrases
    for phrase in URGENT_PHRASES:
        if phrase in text.lower():
            score += 1
            reasons.append(f"Urgency phrase detected: '{phrase}'")

    # 9. Sender spoofing
    if SENDER_SPOOF_PATTERN.search(text):
        score += 1
        reasons.append("Suspicious sender address pattern detected")

    final_score = min(score, 10)
    prediction = "phishing" if final_score >= 2 else "legit"
    return final_score, prediction, reasons


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n📧 Paste your email text below (Ctrl+D / Ctrl+Z to submit):\n")
    input_text = sys.stdin.read()

    score, prediction, reasons = score_email(input_text)

    print("\n──── RULE-BASED PHISHING DETECTION ────")
    print(f"🔺 Score      : {score}/10")
    print(f"🔷 Prediction : {prediction.upper()}\n")
    print("Reasons:")
    if reasons:
        for reason in reasons:
            print(f"  • {reason}")
    else:
        print("  • No phishing indicators detected.")
