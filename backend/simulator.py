import pandas as pd
import time
import requests
import sys

API_URL = "http://localhost:8000/analyze"
CSV_PATH = "data/creditcard.csv"
DELAY_SECONDS = 0.4 # Saniyede 2.5 islem

def main():
    print("=" * 60)
    print("[BANK] FraudDetection Banka Simulatoru Baslatiliyor...")
    print("=" * 60)

    try:
        # Sadece test amacli bir surec oldugu icin once df'i tamamen okuyoruz
        # Buyuk veri ise chunk size ile de okunabilir ama demo icin yeterli
        df = pd.read_csv(CSV_PATH)
        feature_cols = [c for c in df.columns if c != "Class"]
    except FileNotFoundError:
        print(f"[HATA] {CSV_PATH} bulunamadı.")
        sys.exit(1)

    print(f"[BİLGİ] {len(df)} adet işlem kuyruğa alındı.")
    print(f"[BİLGİ] İşlemler saniyede ~{1/DELAY_SECONDS:.1f} hızında API'ye gönderiliyor...\n")

    for index, row in df.iterrows():
        # Transaction modeline uygun dictionary olustur
        payload = row[feature_cols].to_dict()

        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                is_fraud = result.get("is_fraud", False)
                fraud_type = result.get("fraud_type", "normal")
                confidence = result.get("confidence", 0.0)
                message = result.get("message", "")

                # Formatli cikti
                amount = payload.get("Amount", 0.0)
                time_val = payload.get("Time", 0.0)
                
                base_info = f"Islem #{index} | Zaman: {time_val} | Tutar: ${amount:.2f}"
                
                if is_fraud:
                    if fraud_type == "UNKNOWN_NEW_FRAUD":
                        print(f"\n[!] [NEW FRAUD ALERT] {base_info}")
                        print(f"   => {message}")
                        print(f"   => Confidence: {confidence:.4f}\n")
                    else:
                        print(f"\n[!] [ALERT] {base_info}")
                        print(f"   => Tip: {fraud_type} | Confidence: {confidence:.4f}\n")
                else:
                    # Normal islemler icin tek satir log (hizli akis hissi)
                    sys.stdout.write(f"\r[OK] {base_info} - Normal")
                    sys.stdout.flush()

            else:
                print(f"\n[HATA] API'den {response.status_code} dondu. {response.text}")
        except requests.exceptions.ConnectionError:
            print("\n[HATA] API'ye baglanilamadi. 'uvicorn backend.main:app' calisiyor mu?")
            break

        time.sleep(DELAY_SECONDS)

if __name__ == "__main__":
    main()
