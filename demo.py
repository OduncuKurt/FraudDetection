"""
demo.py
-------
FraudDetection projesini hizli calistirmak icin yardimci script.

Kullanim:
    python demo.py --mode centralized   # Merkezi model egit
    python demo.py --mode fl            # Federated Learning egit
    python demo.py --mode fzsl          # Zero-Shot Learning egit (ana konu)
    python demo.py --mode all           # Hepsini sirali calistir
"""

import argparse
import subprocess
import sys


MODES = {
    "centralized": {
        "module": "src.train_centralized",
        "desc": "Merkezi (Centralized) MLP egitimi",
    },
    "fl": {
        "module": "src.train_fl",
        "desc": "Federated Learning (FedAvg) egitimi",
    },
    "fzsl": {
        "module": "src.fzsl.train_fzsl",
        "desc": "Federated Zero-Shot Learning egitimi",
    },
}


def run_module(module: str, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, "-m", module],
        check=False,
    )
    if result.returncode != 0:
        print(f"\n[HATA] {module} basarisiz oldu (exit code {result.returncode})")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="FraudDetection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()) + ["all"],
        default="fzsl",
        help="Hangi modeli calistirmak istiyorsunuz? (varsayilan: fzsl)",
    )
    args = parser.parse_args()

    # Veri dosyasinin varligini kontrol et
    import os
    if not os.path.exists("data/creditcard.csv"):
        print("\n[UYARI] data/creditcard.csv bulunamadi!")
        print("  Lutfen asagidaki adresten indirin:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  Indirdikten sonra 'data/' klasorune koyun.\n")
        sys.exit(1)

    if args.mode == "all":
        for name, cfg in MODES.items():
            ok = run_module(cfg["module"], cfg["desc"])
            if not ok:
                print(f"[DURDURULDU] {name} basarisiz, devam edilmiyor.")
                break
    else:
        cfg = MODES[args.mode]
        run_module(cfg["module"], cfg["desc"])


if __name__ == "__main__":
    main()