# Seen fraud types: fraud_type_0, fraud_type_1, fraud_type_2
# Unseen fraud type (zero-shot target): fraud_type_3
# These natural-language descriptions are embedded with SBERT and used
# as class prototypes for zero-shot classification.

FRAUD_CLASS_DESCRIPTIONS = {
    "fraud_type_0": (
        "This is a high-value fraudulent transaction characterized by unusually large "
        "purchase amounts. The transaction exhibits strong anomaly signals in multiple "
        "latent dimensions, suggesting card cloning or large-scale online fraud where "
        "stolen card details are used for expensive purchases in a very short time window."
    ),
    "fraud_type_1": (
        "This is a medium-scale fraudulent transaction with moderate amounts and distinct "
        "behavioral deviations from the cardholder's normal spending pattern. It resembles "
        "account takeover fraud where an attacker gains access to a legitimate account and "
        "makes purchases that deviate subtly from typical user behavior."
    ),
    "fraud_type_2": (
        "This is a micro-transaction fraud pattern involving very small purchase amounts, "
        "often used to test whether a stolen card is active before making larger fraudulent "
        "purchases. These card-probing transactions occur in rapid succession and show "
        "concentrated irregularities in latent transaction features."
    ),
    "fraud_type_3": (
        "This is a sophisticated money laundering fraud pattern where moderate transaction "
        "amounts are used to disguise illegally obtained funds as legitimate purchases. "
        "The transactions appear superficially normal in terms of amount but show strong "
        "structural anomalies in latent space, particularly in time-based behavioral features "
        "and transaction velocity patterns, consistent with transaction laundering schemes."
    ),
    "normal": (
        "This is a completely legitimate credit card transaction made by the authorized "
        "cardholder. The transaction amount, timing, and all behavioral features are "
        "consistent with the cardholder's historical spending habits. No anomaly signals "
        "are present and the transaction follows expected patterns for genuine purchases."
    ),
}

# Hangi fraud tipleri eğitimde görülüyor (seen), hangisi sıfır-atış testiyle değerlendiriliyor (unseen)
SEEN_CLASSES = ["normal", "fraud_type_0", "fraud_type_1", "fraud_type_2"]
UNSEEN_CLASS = "fraud_type_3"

# Sınıf indeksleri (eğitim sırasında kullanılır)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SEEN_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}