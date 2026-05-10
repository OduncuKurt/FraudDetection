"""
demo.py
-------
Unified Fraud Detection System - Tam Demo

Kullanim:
    python demo.py --mode train     # Sistemi egit ve kaydet
    python demo.py --mode predict   # Egitilmis sistemi yukle, ornekler goster
    python demo.py --mode all       # Egit + goster

    python demo.py --csv data/creditcard.csv --mode train
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def check_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"\n[HATA] {csv_path} bulunamadi!")
        print("  Lutfen indirin: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  Indirilen dosyayi 'data/' klasorune koyun.\n")
        sys.exit(1)


def mode_train(csv_path, fl_rounds, fzsl_epochs):
    from src.system import FraudDetectionSystem
    check_data(csv_path)

    print("\n" + "=" * 60)
    print("  EGITIM MODU")
    print("=" * 60)

    system = FraudDetectionSystem()
    system.train(
        csv_path=csv_path,
        fl_rounds=fl_rounds,
        fzsl_epochs=fzsl_epochs,
    )
    system.save()
    print("\n[OK] Sistem kaydedildi: checkpoints/fraud_system.pkl")


def mode_predict(csv_path, n_samples=5):
    from src.system import CHECKPOINT_PATH
    from src.inference import FraudAnalyzer
    check_data(csv_path)

    if not os.path.exists(CHECKPOINT_PATH):
        print("[HATA] Egitilmis model bulunamadi. Once 'python demo.py --mode train' calistirin.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  TAHMIN MODU")
    print("=" * 60)

    analyzer = FraudAnalyzer()
    df = pd.read_csv(csv_path)

    # Kac fraud ve normal ornek gosterecegiz
    n_each = n_samples // 2

    fraud_rows  = df[df["Class"] == 1].sample(n=min(n_each, (df["Class"]==1).sum()),
                                               random_state=42)
    normal_rows = df[df["Class"] == 0].sample(n=min(n_each, (df["Class"]==0).sum()),
                                               random_state=42)
    sample_df   = pd.concat([fraud_rows, normal_rows]).sample(frac=1, random_state=0)

    feat_cols = [c for c in df.columns if c != "Class"]
    true_labels = sample_df["Class"].values

    print(f"\n{n_samples} ornek islem analiz ediliyor...\n")

    correct = 0
    for i, (_, row) in enumerate(sample_df.iterrows()):
        feat = {f: row[f] for f in feat_cols}
        result = analyzer.analyze(feat, explain=False)

        pred_fraud = result["is_fraud"]
        true_fraud = bool(true_labels[i])
        match = pred_fraud == true_fraud
        correct += int(match)

        status = "DOGRU" if match else "YANLIS"
        print(f"  Ornek {i+1:2d} | Gercek: {'FRAUD' if true_fraud else 'NORMAL':<6} "
              f"| Tahmin: {'FRAUD' if pred_fraud else 'NORMAL':<6} "
              f"| Tip: {result['fraud_type']:<20} "
              f"| FL_prob={result['fl_probability']:.3f} "
              f"| [{status}]")

    print(f"\n  Dogruluk: {correct}/{n_samples} ({100*correct/n_samples:.1f}%)")

    # Bir fraud ornegi icin detayli rapor
    fraud_sample = sample_df[sample_df["Class"] == 1].iloc[0]
    feat = {f: fraud_sample[f] for f in feat_cols}
    print("\n" + "=" * 60)
    print("  Detayli Rapor (Fraud Ornegi):")
    result = analyzer.analyze(feat, explain=False)
    FraudAnalyzer.print_report(result)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Fraud Detection System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["train", "predict", "all"],
                        default="all", help="Calistirma modu (varsayilan: all)")
    parser.add_argument("--csv", default="data/creditcard.csv",
                        help="creditcard.csv dosya yolu")
    parser.add_argument("--fl-rounds", type=int, default=5,
                        help="Federated Learning tur sayisi")
    parser.add_argument("--fzsl-epochs", type=int, default=30,
                        help="FZSL egitim epoch sayisi")
    parser.add_argument("--samples", type=int, default=6,
                        help="Predict modunda gosterilecek ornek sayisi")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        mode_train(args.csv, args.fl_rounds, args.fzsl_epochs)

    if args.mode in ("predict", "all"):
        mode_predict(args.csv, n_samples=args.samples)


if __name__ == "__main__":
    main()