"""
explainer.py
------------
İşlem verilerinden son kullanıcı için anlaşılır Türkçe risk açıklaması üretir.
V1-V28 değerleri doğrudan yorumlanamaz (PCA-anonymized), bu yüzden:
  - Amount ve Time (gerçek, yorumlanabilir)
  - Model çıktıları (FL prob, FZSL sim scores, confidence)
  - Fraud tipi küme bilgisi
üzerinden doğal dil açıklaması oluşturulur.
"""

def human_explanation(
    amount: float,
    time_sec: float,
    fraud_type: str,
    fl_probability: float,
    fzsl_fraud_prob: float,
    confidence: float,
    similarity_scores: dict,
    shap_values: dict = None,
) -> dict:
    """
    Son kullanıcıya gösterilecek Türkçe risk açıklaması üretir.
    Teknik SHAP değerleri ayrı tutulur.

    Döner:
        {
          "verdict":    "FRAUD — Yüksek Risk",
          "risk_level": "KRİTİK",  # DÜŞÜK / ORTA / YÜKSEK / KRİTİK
          "bullets":    ["...", "...", ...],   # Madde madde Türkçe gerekçe
          "technical":  {"top_shap_feature": "V14", ...}  # Teknik özet
        }
    """
    bullets = []
    technical = {}

    # ── 1. İşlem tutarı yorumu ──────────────────────────────────────────
    if amount < 1.0:
        bullets.append(
            f"İşlem tutarı çok düşük (${amount:.2f}): Bu büyük fraud öncesi "
            "çalıntı kartın aktif olup olmadığını test eden tipik bir 'keşif hareketi'."
        )
    elif amount < 10.0:
        bullets.append(
            f"Küçük tutarlı işlem (${amount:.2f}): Kart test işlemi olabilir. "
            "Kısa sürede ardışık küçük ödemeler genellikle büyük fraud öncesi görülür."
        )
    elif amount > 500:
        bullets.append(
            f"Olağandışı yüksek tutar (${amount:.2f}): "
            "Büyük tutarlı ani alışveriş, kart kopyalama veya hesap ele geçirme ile uyumlu."
        )
    elif amount > 200:
        bullets.append(
            f"Normalin üzerinde tutar (${amount:.2f}): "
            "Kart sahibinin tipik harcama aralığı dışında görünüyor."
        )
    else:
        bullets.append(
            f"İşlem tutarı (${amount:.2f}) tek başına şüphe uyandırmıyor — "
            "ancak gizli davranış özellikleri (V1-V28) güçlü fraud sinyali veriyor."
        )

    # ── 2. Zaman yorumu ──────────────────────────────────────────────────
    # Time saniye cinsinden, günün saatine çevir (yaklaşık)
    hour = int((time_sec % 86400) / 3600) if time_sec > 0 else -1
    if hour >= 0:
        if 0 <= hour < 5:
            bullets.append(
                f"Gece yarısı işlemi ({hour:02d}:xx): "
                "Gece 00-05 arası fraudların önemli bir kısmı bu saatte gerçekleşir."
            )
        elif 5 <= hour < 8:
            bullets.append(
                f"Erken sabah işlemi ({hour:02d}:xx): "
                "İş saatleri öncesi otomatik fraud girişimleriyle örtüşüyor."
            )

    # ── 3. Model kararı yorumu ───────────────────────────────────────────
    if fl_probability > 0.90:
        bullets.append(
            f"Federated Learning modeli %{fl_probability*100:.0f} kesinlikte fraud işaret ediyor. "
            f"Bu model, 4 farklı bankanın verisiyle birlikte eğitildi."
        )
    elif fl_probability > 0.70:
        bullets.append(
            f"FL modeli %{fl_probability*100:.0f} olasılıkla fraud diyor "
            "(eşiğin üzerinde, kural devreye girdi)."
        )

    if fzsl_fraud_prob > 0.80:
        bullets.append(
            f"FZSL sınıflandırıcısı %{fzsl_fraud_prob*100:.0f} fraud olasılığı "
            "hesaplıyor (bilinen fraud davranış prototipleriyle yüksek benzerlik)."
        )

    # ── 4. FZSL benzerlik yorumu (hangi fraud tipine en çok benziyor) ───
    if similarity_scores:
        sorted_sims = sorted(similarity_scores.items(), key=lambda x: -x[1])
        top_class, top_score = sorted_sims[0]
        normal_score = similarity_scores.get("normal", 0)

        if top_class != "normal" and top_score > 0.3:
            type_names = {
                "fraud_type_0": "kart kopyalama (Cluster-0)",
                "fraud_type_1": "hesap ele geçirme (Cluster-1)",
                "fraud_type_2": "kart test işlemi (Cluster-2)",
                "fraud_type_3": "para aklama / bilinmeyen tip (Cluster-3)",
            }
            bullets.append(
                f"FZSL benzerlik analizi: Bu işlem, '{type_names.get(top_class, top_class)}' "
                f"fraud örüntüsüne {top_score:.2f} puan benzerlik gösteriyor. "
                f"Normal işlem benzerliği ise yalnızca {normal_score:.2f}."
            )
        elif top_class != "normal" and top_score > 0:
            bullets.append(
                f"FZSL: İşlem bilinen fraud tiplerine düşük benzerlik gösteriyor "
                f"ama normal skor ({normal_score:.2f}) altında — yine de alarm verildi."
            )

    # ── 5. Fraud tipine göre ek açıklama ────────────────────────────────
    type_extra = {
        "fraud_type_0": (
            "Küme analizi: Bu işlem, V14-V4-V12 gizli özelliklerinde 'Cluster-0' fraud grubuyla "
            "örtüşüyor. Bu gruptaki işlemler genellikle kart bilgisi çalınıp farklı lokasyondan "
            "kullanım veya büyük e-ticaret alışverişleri şeklinde görünür. "
            f"Küme ortalaması $172 — bu işlemin tutarı bu ortalamanın "
            f"{'altında' if amount < 172 else 'üzerinde'} olsa da davranış özellikleri eşleşiyor."
        ),
        "fraud_type_1": (
            "Küme analizi: 'Cluster-1' — hesap sahipliği değişimi örüntüsü. "
            "V3-V17-V7 özellikleri kartın normal kullanıcısından farklı birinin "
            "kullandığına işaret ediyor. Kart sahibinin geçmiş alışveriş davranışından "
            "belirgin sapma."
        ),
        "fraud_type_2": (
            "Küme analizi: 'Cluster-2' — kart keşif işlemi. "
            f"${amount:.2f} tutarındaki bu işlem kart aktiflik testi için tipik. "
            "V7-V3-V1 gizli özellikleri çok küçük tutarlı hızlı ardışık işlem örüntüsüyle uyumlu."
        ),
        "fraud_type_3": (
            "⚠️ FZSL Zero-Shot Tespiti: Bu fraud tipi modelin eğitim setinde HİÇ yoktu. "
            "Metin tabanlı sınıf prototipleriyle yapılan Zero-Shot karşılaştırması "
            "para aklama şemasıyla uyumluluk gösteriyor. "
            "Gizli özellikler (V14-V17-V12) normalden yapısal sapma içeriyor "
            "ancak işlem tutarı normal görünebilir — bu para aklamaya özgü bir taktik."
        ),
    }
    if fraud_type in type_extra:
        bullets.append(type_extra[fraud_type])

    # ── 6. SHAP teknik özet (grafik için) ───────────────────────────────
    if shap_values:
        top3 = sorted(shap_values.items(), key=lambda x: -abs(x[1]))[:3]
        technical["top_shap_features"] = [
            {"feature": f, "value": round(v, 4), "direction": "fraud↑" if v > 0 else "fraud↓"}
            for f, v in top3
        ]
        technical["shap_note"] = (
            "SHAP değerleri FL modelinin binary kararını açıklar. "
            "V1-V28, bankanın PCA ile anonim hale getirdiği orijinal özelliklerin "
            "matematiksel bileşimleridir. Pozitif SHAP = fraud ihtimalini artıran özellik."
        )

    # ── Risk seviyesi ────────────────────────────────────────────────────
    if confidence > 0.90 or fl_probability > 0.90:
        risk_level = "KRİTİK"
        verdict = "FRAUD — Kritik Risk"
    elif confidence > 0.75 or fl_probability > 0.75:
        risk_level = "YÜKSEK"
        verdict = "FRAUD — Yüksek Risk"
    else:
        risk_level = "ORTA"
        verdict = "FRAUD — Orta Risk (Manuel İnceleme Önerilir)"

    if fraud_type == "fraud_type_3":
        risk_level = "KRİTİK — YENİ TİP"
        verdict = "⚠️ FRAUD — Eğitimde Görülmemiş Yeni Tip!"

    return {
        "verdict": verdict,
        "risk_level": risk_level,
        "bullets": bullets,
        "technical": technical,
    }
