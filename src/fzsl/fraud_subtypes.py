import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_creditcard_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def generate_fraud_subtypes(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42
):
    """
    Sadece fraud kayıtları üzerinde clustering yapar.
    Sonra tüm veri setine fraud_subtype kolonu ekler.

    normal kayıtlar için fraud_subtype = 'normal'
    fraud kayıtlar için fraud_subtype = 'fraud_type_0', ...
    """

    df = df.copy()

    fraud_df = df[df["Class"] == 1].copy()
    normal_df = df[df["Class"] == 0].copy()

    # Clustering için feature'lar
    X_fraud = fraud_df.drop(columns=["Class"])

    scaler = StandardScaler()
    X_fraud_scaled = scaler.fit_transform(X_fraud)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_fraud_scaled)

    fraud_df["fraud_cluster"] = cluster_labels
    fraud_df["fraud_subtype"] = fraud_df["fraud_cluster"].apply(lambda x: f"fraud_type_{x}")

    normal_df["fraud_cluster"] = -1
    normal_df["fraud_subtype"] = "normal"

    df_labeled = pd.concat([normal_df, fraud_df], axis=0).sort_index()

    return df_labeled, kmeans, scaler


def summarize_fraud_subtypes(df_labeled: pd.DataFrame):
    """
    Her fraud subtype için temel istatistikleri çıkarır.
    """
    fraud_only = df_labeled[df_labeled["Class"] == 1].copy()

    summary_rows = []

    for subtype in sorted(fraud_only["fraud_subtype"].unique()):
        subset = fraud_only[fraud_only["fraud_subtype"] == subtype]

        row = {
            "fraud_subtype": subtype,
            "count": len(subset),
            "amount_mean": subset["Amount"].mean(),
            "amount_std": subset["Amount"].std(),
            "time_mean": subset["Time"].mean(),
            "time_std": subset["Time"].std(),
        }

        # En uç ortalama feature'ları bulmak için V sütunlarının ortalaması
        v_cols = [c for c in subset.columns if c.startswith("V")]
        feature_means = subset[v_cols].mean().abs().sort_values(ascending=False)

        top_features = feature_means.head(5).index.tolist()
        row["top_latent_features"] = ", ".join(top_features)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def assign_seen_unseen(
    df_labeled: pd.DataFrame,
    unseen_subtype: str = "fraud_type_3"
):
    """
    Seen/unseen etiketlerini ekler.
    """
    df_labeled = df_labeled.copy()

    def map_seen_unseen(subtype):
        if subtype == "normal":
            return "normal"
        elif subtype == unseen_subtype:
            return "unseen"
        else:
            return "seen"

    df_labeled["fzsl_split"] = df_labeled["fraud_subtype"].apply(map_seen_unseen)
    return df_labeled


def build_seen_unseen_datasets(df_labeled: pd.DataFrame, unseen_subtype: str = "fraud_type_3"):
    """
    Eğitim ve test mantığı için:
    - train: normal + seen fraud
    - test: normal + seen fraud + unseen fraud
    """

    df_marked = assign_seen_unseen(df_labeled, unseen_subtype=unseen_subtype)

    train_df = df_marked[df_marked["fzsl_split"].isin(["normal", "seen"])].copy()
    test_df = df_marked.copy()

    return train_df, test_df, df_marked