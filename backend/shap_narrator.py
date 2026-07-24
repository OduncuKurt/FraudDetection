"""
shap_narrator.py  — v3 (English)
---------------------------------
Takes SHAP values + real feature values and generates automatic
English-language explanation bullets for each fraud decision.

Example output:
  "V12 (value: +2.84, contribution: 31.4%, effect: strongly increases ↑):
   V12 in this fraud type typically trends high — consistent pattern here.
   Strongly supports the FRAUD decision."
"""

from typing import Optional

# PCA components that carry strong signals per fraud type
# (Cross-referenced from training results + SHAP analysis)
CLUSTER_SIGNATURES = {
    "fraud_type_0": {
        "positive": ["V14", "V4", "V3", "V11", "V2"],
        "negative": ["V12", "V10", "V17", "V16"],
        "pattern": "card cloning / large-scale e-commerce fraud",
        "typical_amount_range": (50, 2000),
    },
    "fraud_type_1": {
        "positive": ["V3", "V7", "V4", "V10", "V2"],
        "negative": ["V17", "V1", "V16", "V14"],
        "pattern": "account takeover (ATO)",
        "typical_amount_range": (20, 500),
    },
    "fraud_type_2": {
        "positive": ["V7", "V3", "V10", "V14"],
        "negative": ["V1", "V5", "V12"],
        "pattern": "card activity probing (micro-transaction testing)",
        "typical_amount_range": (0.01, 15),
    },
    "fraud_type_3": {
        "positive": ["V14", "V12", "V3", "V4"],
        "negative": ["V17", "V10", "V1", "V16"],
        "pattern": "money laundering / unknown fraud type (Zero-Shot)",
        "typical_amount_range": (5, 1000),
    },
}


def _pct(val: float, total: float) -> str:
    if total < 1e-9:
        return "—"
    return f"{abs(val)/total*100:.1f}%"


def narrate_shap(
    shap_values: dict,
    feature_values: dict,          # scaled feature values {V12: +2.84, ...}
    fraud_type: str,
    amount: float,
    time_sec: float,
    fl_probability: float,
    fzsl_fraud_prob: float,
    confidence: float,
    similarity_scores: dict,
    top_k: int = 6,
) -> list:
    """
    Generate automatic English explanation bullets from SHAP + feature values.
    Each bullet: "Feature X (value: Y, contribution: Z%): Explanation..."
    """
    bullets = []
    sig = CLUSTER_SIGNATURES.get(fraud_type, {})

    # Total absolute SHAP (for contribution % calculation)
    total_abs = sum(abs(v) for v in shap_values.values()) or 1e-9

    # Top-k most influential features
    sorted_shap = sorted(shap_values.items(), key=lambda x: -abs(x[1]))[:top_k]

    # ── Transaction summary ───────────────────────────────────────────
    hour = int((time_sec % 86400) / 3600) if time_sec > 0 else -1
    time_str = f"{hour:02d}:xx" if hour >= 0 else "?"

    bullets.append(
        f"📋 Transaction summary: ${amount:.2f} at {time_str}. "
        f"FL probability: {fl_probability*100:.1f}%, FZSL fraud score: {fzsl_fraud_prob*100:.1f}%, "
        f"overall confidence: {confidence*100:.1f}%."
    )

    # ── SHAP-based feature explanations ──────────────────────────────
    for feature, shap_val in sorted_shap:
        if abs(shap_val) < 0.02:
            continue

        raw_val = feature_values.get(feature, None)
        pct_contrib = _pct(shap_val, total_abs)
        direction_word = "increases ↑" if shap_val > 0 else "decreases ↓"
        decision_effect = "supports the FRAUD decision" if shap_val > 0 else "partially counteracts the FRAUD decision"

        # Effect magnitude
        strength = abs(shap_val) / total_abs
        if strength > 0.25:
            mag = "strongly"
        elif strength > 0.15:
            mag = "significantly"
        elif strength > 0.08:
            mag = "moderately"
        else:
            mag = "slightly"

        # Feature-specific explanation
        if feature == "Amount":
            raw_str = f"${amount:.2f}"
            typical_lo, typical_hi = sig.get("typical_amount_range", (0, 9999))
            if amount < typical_lo:
                context = (f"This amount is below the typical range for this fraud type "
                           f"(${typical_lo}–${typical_hi}) — behavioral pattern dominates over amount.")
            elif amount > typical_hi:
                context = (f"This amount exceeds the typical range (${typical_lo}–${typical_hi}) "
                           f"— high amount reinforces the fraud decision.")
            else:
                context = (f"Amount is within the typical fraud range (${typical_lo}–${typical_hi}) "
                           f"— used as supporting evidence by the model.")
        elif feature == "Time":
            raw_str = time_str
            if hour >= 0 and (hour < 5 or hour > 22):
                context = f"Transaction at {hour:02d}:xx — late-night hours are common for this fraud type."
            else:
                context = "Business hours transaction — time factor has limited impact."
        else:
            # V-feature
            raw_str = f"{raw_val:+.4f}" if raw_val is not None else "?"
            known_pos = feature in sig.get("positive", [])
            known_neg = feature in sig.get("negative", [])

            if known_pos and shap_val > 0:
                context = (f"{feature} typically trends high (positive) in this fraud type. "
                           f"Value here: {raw_str} — matches the known pattern, {decision_effect}.")
            elif known_neg and shap_val < 0:
                context = (f"{feature} typically trends low (negative) in this fraud type. "
                           f"Value here: {raw_str} — matches the known pattern, {decision_effect}.")
            elif known_pos and shap_val < 0:
                context = (f"{feature} expected high in this fraud type, but observed {raw_str} here. "
                           f"Counterintuitive — yet still {decision_effect}.")
            elif known_neg and shap_val > 0:
                context = (f"{feature} expected low in this fraud type, but observed {raw_str} here. "
                           f"Unexpected deviation — model was surprised but still {decision_effect}.")
            else:
                if shap_val > 0:
                    context = (f"{feature} value of {raw_str} contains an unexpected deviation "
                               f"that {decision_effect}.")
                else:
                    context = (f"{feature} value of {raw_str} slightly reduces fraud probability, "
                               f"but other features dominate.")

        bullets.append(
            f"{'📊' if shap_val > 0 else '🔵'} **{feature}** "
            f"(value: {raw_str}, contribution: {pct_contrib}, effect: {mag} {direction_word}): "
            f"{context}"
        )

    # ── FZSL similarity explanation ───────────────────────────────────
    if similarity_scores:
        sorted_sim = sorted(similarity_scores.items(), key=lambda x: -x[1])
        top_class, top_score = sorted_sim[0]
        normal_score = similarity_scores.get("normal", 0)
        type_labels = {
            "fraud_type_0": "card cloning (Cluster-0)",
            "fraud_type_1": "account takeover (Cluster-1)",
            "fraud_type_2": "card probing (Cluster-2)",
            "fraud_type_3": "money laundering / Zero-Shot",
        }
        bullets.append(
            f"🟣 **FZSL Classification**: Highest similarity to "
            f"'{type_labels.get(top_class, top_class)}' prototype "
            f"({top_score:+.4f}). Normal transaction similarity: {normal_score:+.4f}. "
            f"Gap: {top_score - normal_score:+.4f} — larger gap = stronger fraud signal."
        )

    # ── Final decision summary ────────────────────────────────────────
    top_drivers = [f for f, v in sorted_shap[:3] if v > 0]
    if top_drivers:
        bullets.append(
            f"✅ **Decision summary**: Model flagged this transaction as FRAUD. "
            f"Key drivers: {', '.join(top_drivers)}. "
            f"Confidence: {confidence*100:.1f}%. "
            f"{'Automatic block recommended.' if confidence > 0.9 else 'Manual review recommended.'}"
        )

    return bullets


def get_risk_level(fl_probability: float, confidence: float, fraud_type: str) -> str:
    if fraud_type == "fraud_type_3":
        return "CRITICAL — NEW TYPE"
    if confidence > 0.90 or fl_probability > 0.90:
        return "CRITICAL"
    if confidence > 0.75 or fl_probability > 0.70:
        return "HIGH"
    return "MEDIUM"


def build_human_explanation(
    shap_values: Optional[dict],
    feature_values: Optional[dict],
    fraud_type: str,
    amount: float,
    time_sec: float,
    fl_probability: float,
    fzsl_fraud_prob: float,
    confidence: float,
    similarity_scores: dict,
) -> dict:
    """
    Full explanation object returned to the dashboard.
    shap_values + feature_values available → real SHAP-based explanation.
    Otherwise → metric-based explanation (fallback).
    """
    risk_level = get_risk_level(fl_probability, confidence, fraud_type)

    verdict_map = {
        "CRITICAL — NEW TYPE": "⚠️ FRAUD — Previously Unseen Pattern (Zero-Shot)!",
        "CRITICAL":            "FRAUD — Critical Risk",
        "HIGH":                "FRAUD — High Risk",
        "MEDIUM":              "FRAUD — Medium Risk (Manual Review Recommended)",
    }
    verdict = verdict_map.get(risk_level, "FRAUD")

    has_shap = bool(shap_values and len(shap_values) > 0)
    has_feat = bool(feature_values and len(feature_values) > 0)

    if has_shap and has_feat:
        bullets = narrate_shap(
            shap_values=shap_values,
            feature_values=feature_values,
            fraud_type=fraud_type,
            amount=amount,
            time_sec=time_sec,
            fl_probability=fl_probability,
            fzsl_fraud_prob=fzsl_fraud_prob,
            confidence=confidence,
            similarity_scores=similarity_scores,
        )
        source = "shap_based"
    else:
        bullets = _metric_bullets(
            amount=amount, time_sec=time_sec, fraud_type=fraud_type,
            fl_probability=fl_probability, fzsl_fraud_prob=fzsl_fraud_prob,
            confidence=confidence, similarity_scores=similarity_scores,
        )
        source = "metric_based"

    return {
        "verdict": verdict,
        "risk_level": risk_level,
        "bullets": bullets,
        "source": source,
        "shap_used": has_shap and has_feat,
    }


def _metric_bullets(amount, time_sec, fraud_type, fl_probability,
                     fzsl_fraud_prob, confidence, similarity_scores):
    """Fallback: metric-based bullets when SHAP is unavailable."""
    bullets = []
    hour = int((time_sec % 86400) / 3600) if time_sec > 0 else -1
    sig = CLUSTER_SIGNATURES.get(fraud_type, {})

    bullets.append(
        f"📋 Transaction summary: ${amount:.2f} at {hour:02d}:xx. "
        f"FL: {fl_probability*100:.1f}% | FZSL: {fzsl_fraud_prob*100:.1f}% | Confidence: {confidence*100:.1f}%. "
        "(SHAP values computing in background — detailed explanation will follow.)"
    )

    lo, hi = sig.get("typical_amount_range", (0, 9999))
    if amount < lo:
        bullets.append(f"💰 Amount ${amount:.2f} is below typical range for this fraud type "
                        f"(${lo}–${hi}) — behavioral pattern is the primary signal.")
    elif amount > hi:
        bullets.append(f"💰 High amount (${amount:.2f}): above typical range (${lo}–${hi}).")
    else:
        bullets.append(f"💰 Amount (${amount:.2f}) falls within typical fraud range (${lo}–${hi}).")

    if hour >= 0 and (hour < 5 or hour > 22):
        bullets.append(f"🕐 Late-night transaction at {hour:02d}:xx — this time window is common for fraud.")

    if similarity_scores:
        sorted_sim = sorted(similarity_scores.items(), key=lambda x: -x[1])
        top_class, top_score = sorted_sim[0]
        normal_score = similarity_scores.get("normal", 0)
        type_labels = {
            "fraud_type_0": "card cloning",
            "fraud_type_1": "account takeover",
            "fraud_type_2": "card probing",
            "fraud_type_3": "money laundering (ZSL)"
        }
        bullets.append(
            f"🟣 FZSL: '{type_labels.get(top_class, top_class)}' similarity {top_score:+.4f}, "
            f"normal score {normal_score:+.4f}. Gap: {top_score-normal_score:+.4f}."
        )

    bullets.append(
        f"✅ Decision: FRAUD with {confidence*100:.1f}% confidence. "
        f"{'Automatic block recommended.' if confidence > 0.9 else 'Manual review recommended.'}"
    )
    return bullets
