from src.fzsl.fraud_subtypes import (
    load_creditcard_dataset,
    generate_fraud_subtypes,
    summarize_fraud_subtypes,
    build_seen_unseen_datasets,
)

print("test_fraud_subtypes.py başladı")


def main():
    print("main() içine girdi")

    csv_path = "data/creditcard.csv"

    df = load_creditcard_dataset(csv_path)
    print("Dataset yüklendi:", df.shape)

    df_labeled, kmeans, scaler = generate_fraud_subtypes(
        df,
        n_clusters=4,
        random_state=42
    )
    print("Fraud subtype üretildi")

    print("\nFraud subtype counts:")
    print(df_labeled["fraud_subtype"].value_counts())

    summary_df = summarize_fraud_subtypes(df_labeled)
    print("\nFraud subtype summary:")
    print(summary_df)

    train_df, test_df, df_marked = build_seen_unseen_datasets(
        df_labeled,
        unseen_subtype="fraud_type_3"
    )
    print("Seen/unseen ayrımı yapıldı")

    print("\nFZSL split counts:")
    print(df_marked["fzsl_split"].value_counts())

    print("\nTrain split subtype counts:")
    print(train_df["fraud_subtype"].value_counts())

    print("\nTest split subtype counts:")
    print(test_df["fraud_subtype"].value_counts())


if __name__ == "__main__":
    main()