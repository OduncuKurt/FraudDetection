"""
shap_narrator.py  — v2
-----------------------
SHAP değerlerini + gerçek feature değerlerini alarak
her özellik için otomatik Türkçe açıklama üretir.

Örnek çıktı:
  "V12 özelliği (değer: +2.84, fraud katkısı: %31.4):
   Bu işlemdeki V12, bilinen fraud örüntüsünde genellikle yüksek seyreder —
   bu işlemde de yüksek. Modelin FRAUD kararını güçlü biçimde destekliyor."
"""

from typing import Optional

# Fraud tiplerinin hangi PCA bileşenlerinde güçlü sinyal verdiği
# (Eğitim sonuçları + SHAP analizi çapraz referansı)
CLUSTER_SIGNATURES = {
    "fraud_type_0": {
        "positive": ["V14", "V4", "V3", "V11", "V2"],
        "negative": ["V12", "V10", "V17", "V16"],
        "pattern": "kart kopyalama / büyük e-ticaret fraud",
        "typical_amount_range": (50, 2000),
    },
    "fraud_type_1": {
        "positive": ["V3", "V7", "V4", "V10", "V2"],
        "negative": ["V17", "V1", "V16", "V14"],
        "pattern": "hesap ele geçirme (account takeover)",
        "typical_amount_range": (20, 500),
    },
    "fraud_type_2": {
        "positive": ["V7", "V3", "V10", "V14"],
        "negative": ["V1", "V5", "V12"],
        "pattern": "kart aktiflik testi (micro-transaction probing)",
        "typical_amount_range": (0.01, 15),
    },
    "fraud_type_3": {
        "positive": ["V14", "V12", "V3", "V4"],
        "negative": ["V17", "V10", "V1", "V16"],
        "pattern": "para aklama / bilinmeyen fraud tipi (Zero-Shot)",
        "typical_amount_range": (5, 1000),
    },
}


def _pct(val: float, total: float) -> str:
    if total < 1e-9:
        return "—"
    return f"%{abs(val)/total*100:.1f}"


def narrate_shap(
    shap_values: dict,
    feature_values: dict,          # ham özellik değerleri (scaled) {V12: +2.84, ...}
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
    SHAP + feature değerlerinden otomatik Türkçe açıklama üretir.
    Her bullet:  "Özellik X (değer: Y, katkı: %Z): Açıklama..."
    """
    bullets = []
    sig = CLUSTER_SIGNATURES.get(fraud_type, {})

    # Toplam mutlak SHAP (katkı yüzdesi hesabı için)
    total_abs = sum(abs(v) for v in shap_values.values()) or 1e-9

    # En etkili top_k özellik
    sorted_shap = sorted(shap_values.items(), key=lambda x: -abs(x[1]))[:top_k]

    # ── İşlem özeti ──────────────────────────────────────────────────
    hour = int((time_sec % 86400) / 3600) if time_sec > 0 else -1
    time_str = f"{hour:02d}:xx" if hour >= 0 else "?"

    bullets.append(
        f"📋 İşlem özeti: ${amount:.2f} tutarında, saat {time_str}'de gerçekleşen işlem. "
        f"FL olasılığı %{fl_probability*100:.1f}, FZSL fraud skoru %{fzsl_fraud_prob*100:.1f}, "
        f"genel güven %{confidence*100:.1f}."
    )

    # ── SHAP-bazlı özellik açıklamaları ──────────────────────────────
    for feature, shap_val in sorted_shap:
        if abs(shap_val) < 0.02:
            continue

        raw_val = feature_values.get(feature, None)
        pct_contrib = _pct(shap_val, total_abs)
        direction_word = "artırıyor ↑" if shap_val > 0 else "azaltıyor ↓"
        decision_effect = "FRAUD kararını destekliyor" if shap_val > 0 else "FRAUD kararını frenliyor"

        # Etki büyüklüğü
        strength = abs(shap_val) / total_abs
        if strength > 0.25:
            mag = "en güçlü"
        elif strength > 0.15:
            mag = "güçlü"
        elif strength > 0.08:
            mag = "orta"
        else:
            mag = "hafif"

        # Feature-özel açıklama
        if feature == "Amount":
            raw_str = f"${amount:.2f}"
            typical_lo, typical_hi = sig.get("typical_amount_range", (0, 9999))
            if amount < typical_lo:
                context = (f"Bu tutar bu fraud tipinin tipik aralığının (${typical_lo}–${typical_hi}) "
                           "altında — tutar tek başına belirleyici değil, davranış örüntüsü baskın.")
            elif amount > typical_hi:
                context = (f"Bu tutar bu fraud tipinin tipik aralığının (${typical_lo}–${typical_hi}) "
                           "üzerinde — yüksek tutar kararı güçlendiriyor.")
            else:
                context = (f"Bu tutar tipik aralık içinde (${typical_lo}–${typical_hi}) — "
                           "model tutarı destekleyici kanıt olarak kullanıyor.")
        elif feature == "Time":
            raw_str = time_str
            if hour >= 0 and (hour < 5 or hour > 22):
                context = f"Gece {hour:02d}:xx'de gerçekleşen işlem — bu saat dilimi bu fraud tipinde yaygın."
            else:
                context = "İş saatlerinde gerçekleşen işlem — zaman faktörü katkısı sınırlı."
        else:
            # V-özelliği
            raw_str = f"{raw_val:+.4f}" if raw_val is not None else "?"
            known_pos = feature in sig.get("positive", [])
            known_neg = feature in sig.get("negative", [])

            if known_pos and shap_val > 0:
                context = (f"{feature} bu fraud tipinde genellikle yüksek (pozitif) seyreder. "
                           f"Bu işlemde {raw_str} — örüntüyle tam uyumlu, {decision_effect}.")
            elif known_neg and shap_val < 0:
                context = (f"{feature} bu fraud tipinde genellikle düşük (negatif) seyreder. "
                           f"Bu işlemde {raw_str} — örüntüyle tam uyumlu, {decision_effect}.")
            elif known_pos and shap_val < 0:
                context = (f"{feature} bu fraud tipinde normalde yüksek beklenir ama bu işlemde {raw_str}. "
                           f"Beklenenin tersine davranıyor — {decision_effect}.")
            elif known_neg and shap_val > 0:
                context = (f"{feature} bu fraud tipinde normalde düşük beklenir ama bu işlemde {raw_str}. "
                           f"Beklenenin tersine davranıyor — modelin kararını şaşırttı ama yine de {decision_effect}.")
            else:
                if shap_val > 0:
                    context = (f"Bu işlemdeki {feature} değeri {raw_str}, "
                               f"modelin fraud kararını destekleyen beklenmedik bir sapma içeriyor.")
                else:
                    context = (f"Bu işlemdeki {feature} değeri {raw_str}, "
                               f"kararı hafifçe frenliyor ancak diğer özellikler baskın geliyor.")

        bullets.append(
            f"{'📊' if shap_val > 0 else '🔵'} **{feature}** "
            f"(değer: {raw_str}, katkı: {pct_contrib}, etki: {mag} {direction_word}): "
            f"{context}"
        )

    # ── FZSL benzerlik açıklaması ─────────────────────────────────────
    if similarity_scores:
        sorted_sim = sorted(similarity_scores.items(), key=lambda x: -x[1])
        top_class, top_score = sorted_sim[0]
        normal_score = similarity_scores.get("normal", 0)
        type_labels = {
            "fraud_type_0": "kart kopyalama (Cluster-0)",
            "fraud_type_1": "hesap ele geçirme (Cluster-1)",
            "fraud_type_2": "kart test işlemi (Cluster-2)",
            "fraud_type_3": "para aklama / Zero-Shot",
        }
        bullets.append(
            f"🟣 **FZSL Sınıflandırma**: En yüksek benzerlik "
            f"'{type_labels.get(top_class, top_class)}' prototipiyle "
            f"({top_score:+.4f}). Normal işlem benzerliği: {normal_score:+.4f}. "
            f"Fark: {top_score - normal_score:+.4f} — ne kadar büyükse o kadar güçlü fraud sinyali."
        )

    # ── Final karar ───────────────────────────────────────────────────
    top_drivers = [f for f, v in sorted_shap[:3] if v > 0]
    if top_drivers:
        bullets.append(
            f"✅ **Karar özeti**: Model bu işlemi FRAUD olarak işaretledi. "
            f"En belirleyici özellikler: {', '.join(top_drivers)}. "
            f"Güven: %{confidence*100:.1f}. "
            f"{'Otomatik engelleme önerilir.' if confidence > 0.9 else 'Manuel inceleme önerilir.'}"
        )

    return bullets


def get_risk_level(fl_probability: float, confidence: float, fraud_type: str) -> str:
    if fraud_type == "fraud_type_3":
        return "KRİTİK — YENİ TİP"
    if confidence > 0.90 or fl_probability > 0.90:
        return "KRİTİK"
    if confidence > 0.75 or fl_probability > 0.70:
        return "YÜKSEK"
    return "ORTA"


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
    Dashboard'a döndürülecek tam açıklama objesi.
    shap_values + feature_values varsa → gerçek SHAP bazlı açıklama.
    yoksa → metrik bazlı açıklama (fallback).
    """
    risk_level = get_risk_level(fl_probability, confidence, fraud_type)

    verdict_map = {
        "KRİTİK — YENİ TİP": "⚠️ FRAUD — Eğitimde Görülmemiş Yeni Tip!",
        "KRİTİK":             "FRAUD — Kritik Risk",
        "YÜKSEK":             "FRAUD — Yüksek Risk",
        "ORTA":               "FRAUD — Orta Risk (Manuel İnceleme Önerilir)",
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
    """SHAP olmadan sadece model çıktısı + tutar + zaman."""
    bullets = []
    hour = int((time_sec % 86400) / 3600) if time_sec > 0 else -1
    sig = CLUSTER_SIGNATURES.get(fraud_type, {})

    bullets.append(
        f"📋 İşlem özeti: ${amount:.2f} tutar, saat {hour:02d}:xx. "
        f"FL %{fl_probability*100:.1f} | FZSL %{fzsl_fraud_prob*100:.1f} | Güven %{confidence*100:.1f}. "
        "(SHAP arka planda hesaplanıyor, tamamlandığında detaylı açıklama gelecek.)"
    )

    lo, hi = sig.get("typical_amount_range", (0, 9999))
    if amount < lo:
        bullets.append(f"💰 Tutar ${amount:.2f} bu fraud tipinin tipik aralığı (${lo}–${hi}) altında — "
                        "tutar değil davranış örüntüsü belirleyici.")
    elif amount > hi:
        bullets.append(f"💰 Yüksek tutar (${amount:.2f}): tipik aralığın (${lo}–${hi}) üzerinde.")
    else:
        bullets.append(f"💰 Tutar (${amount:.2f}) tipik fraud aralığı içinde (${lo}–${hi}).")

    if hour >= 0 and (hour < 5 or hour > 22):
        bullets.append(f"🕐 Gece {hour:02d}:xx işlemi — bu saat dilimi fraud'larda yaygın.")

    if similarity_scores:
        sorted_sim = sorted(similarity_scores.items(), key=lambda x: -x[1])
        top_class, top_score = sorted_sim[0]
        normal_score = similarity_scores.get("normal", 0)
        type_labels = {"fraud_type_0":"kart kopyalama","fraud_type_1":"hesap ele geçirme",
                       "fraud_type_2":"kart test","fraud_type_3":"para aklama (ZSL)"}
        bullets.append(
            f"🟣 FZSL: '{type_labels.get(top_class,top_class)}' benzerliği {top_score:+.4f}, "
            f"normal skoru {normal_score:+.4f}. Fark: {top_score-normal_score:+.4f}."
        )

    bullets.append(
        f"✅ Karar: %{confidence*100:.1f} güvenle FRAUD. "
        f"{'Otomatik engelleme önerilir.' if confidence>0.9 else 'Manuel inceleme önerilir.'}"
    )
    return bullets
