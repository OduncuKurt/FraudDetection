"""
inference.py
------------
Tek bir islem icin tahmin API'si.

Kullanim:
    from src.inference import FraudAnalyzer
    analyzer = FraudAnalyzer()          # egitilmis sistemi yukler
    result   = analyzer.analyze(row)    # dict ya da list kabul eder
    analyzer.print_report(result)
"""

import numpy as np
from src.system import FraudDetectionSystem, CHECKPOINT_PATH
from src.fzsl.class_descriptions import FRAUD_CLASS_DESCRIPTIONS


class FraudAnalyzer:
    """Egitilmis FraudDetectionSystem uzerinden kolay tahmin arayuzu."""

    def __init__(self, checkpoint: str = CHECKPOINT_PATH, device: str = "cpu"):
        self.system = FraudDetectionSystem.load(checkpoint, device=device)
        self.device = device

    def analyze(self, transaction, explain: bool = False, top_k: int = 8) -> dict:
        """
        Tek bir islem analiz eder.

        Parametreler:
            transaction : dict {ozellik_adi: deger}  ya da  list/array (30 eleman)
            explain     : SHAP aciklamasi eklensin mi? (yavas, opsiyonel)
            top_k       : Kac ozellik gosterilsin

        Donus: dict
            is_fraud, fl_probability, fzsl_fraud_probability,
            fraud_type, fraud_type_description, confidence,
            similarity_scores, [shap_explanation]
        """
        # Girdi normalize et
        if isinstance(transaction, dict):
            feat_arr = np.array(
                [transaction[f] for f in self.system.feature_names],
                dtype=np.float32,
            ).reshape(1, -1)
        else:
            feat_arr = np.array(transaction, dtype=np.float32).reshape(1, -1)

        # StandardScaler uygula
        feat_scaled = self.system.scaler.transform(feat_arr)

        # Tahmin
        result = self.system.predict(feat_scaled, device=self.device)[0]

        # SHAP aciklamasi (istege bagli)
        if explain:
            result["shap_explanation"] = self.system.explain(
                feat_scaled, sample_idx=0, top_k=top_k, device=self.device
            )

        return result

    def analyze_batch(self, transactions, explain: bool = False) -> list:
        """
        Birden fazla islemi toplu analiz eder.

        transactions: list[dict] ya da 2D array [N, 30]
        """
        if isinstance(transactions[0], dict):
            arr = np.array([
                [t[f] for f in self.system.feature_names]
                for t in transactions
            ], dtype=np.float32)
        else:
            arr = np.array(transactions, dtype=np.float32)

        scaled = self.system.scaler.transform(arr)
        results = self.system.predict(scaled, device=self.device)

        if explain:
            for i, res in enumerate(results):
                res["shap_explanation"] = self.system.explain(
                    scaled, sample_idx=i, top_k=8, device=self.device
                )
        return results

    @staticmethod
    def print_report(result: dict):
        """Tahmin sonucunu okunakli sekilde yazdirir."""
        verdict = "FRAUD" if result["is_fraud"] else "NORMAL"
        emoji   = "X" if result["is_fraud"] else "OK"

        print(f"\n{'='*55}")
        print(f"  [{emoji}]  VERDICT : {verdict}")
        print(f"{'='*55}")
        print(f"  FL Binary Prob   : {result['fl_probability']:.4f}")
        print(f"  FZSL Fraud Prob  : {result['fzsl_fraud_probability']:.4f}")
        print(f"  Fraud Type       : {result['fraud_type']}")
        print(f"  Confidence       : {result['confidence']:.4f}")
        print(f"\n  Aciklama:")
        desc = result.get("fraud_type_description", "")
        # Satir sar
        words, line = desc.split(), ""
        for w in words:
            if len(line) + len(w) > 60:
                print(f"    {line}")
                line = w + " "
            else:
                line += w + " "
        if line:
            print(f"    {line}")

        print(f"\n  Similarity Skorlari:")
        for cls, score in sorted(
            result["similarity_scores"].items(), key=lambda x: -x[1]
        ):
            bar = "#" * max(0, int((score + 0.5) * 20))
            print(f"    {cls:<20} {score:>7.4f}  {bar}")

        if "shap_explanation" in result:
            print(f"\n  SHAP - En Etkili Ozellikler:")
            for feat, val in sorted(
                result["shap_explanation"].items(), key=lambda x: -abs(x[1])
            ):
                sign = "+" if val >= 0 else "-"
                print(f"    {feat:<12} {sign}{abs(val):.6f}")

        print(f"{'='*55}\n")
