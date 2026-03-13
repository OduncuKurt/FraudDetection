import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


'''def load_and_preprocess_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(csv_path)
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Scale all features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), scaler'''

def load_and_preprocess_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(csv_path)

    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()
    '''deneme'''
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), scaler, feature_names


def split_into_clients(X, y, num_clients=4):
    """
    IID / random split
    """
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    split_indices = np.array_split(indices, num_clients)

    clients = []
    for idx in split_indices:
        client_X = X[idx]
        client_y = y[idx]
        clients.append((client_X, client_y))

    return clients


def split_into_clients_noniid_amount(X, y, num_clients=4, amount_column_index=-1, shuffle_within_client=True):
    """
    Senaryo A: Amount-based non-IID split

    Mantık:
    - Veriyi Amount sütununa göre sıralar
    - Sıralı veriyi num_clients parçaya böler
    - Böylece her client farklı amount aralığı görür
    """

    sorted_indices = np.argsort(X[:, amount_column_index])
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]

    split_X = np.array_split(X_sorted, num_clients)
    split_y = np.array_split(y_sorted, num_clients)

    clients = []
    for client_X, client_y in zip(split_X, split_y):
        if shuffle_within_client:
            local_idx = np.arange(len(client_X))
            np.random.shuffle(local_idx)
            client_X = client_X[local_idx]
            client_y = client_y[local_idx]

        clients.append((client_X, client_y))

    return clients

def split_into_clients_noniid_label_skew(
    X,
    y,
    num_clients=4,
    fraud_ratios=None,
    shuffle_within_client=True
):
    """
    Senaryo B: Label-skew non-IID split

    Mantık:
    - Fraud örnekleri client'lara eşit olmayan oranlarda dağıtılır
    - Normal örnekler client boyutlarını dengelemek için tamamlanır

    Parametreler:
    - X: numpy array
    - y: numpy array
    - num_clients: client sayısı
    - fraud_ratios: fraud örneklerinin client'lara dağılım oranı
      Örn: [0.5, 0.25, 0.15, 0.10]
    - shuffle_within_client: her client içinde karıştırma

    Dönüş:
    - clients: [(client_X, client_y), ...]
    """

    if fraud_ratios is None:
        fraud_ratios = [0.5, 0.25, 0.15, 0.10]

    if len(fraud_ratios) != num_clients:
        raise ValueError("fraud_ratios uzunluğu num_clients ile aynı olmalı.")

    if not np.isclose(sum(fraud_ratios), 1.0):
        raise ValueError("fraud_ratios toplamı 1.0 olmalı.")

    fraud_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    np.random.shuffle(fraud_idx)
    np.random.shuffle(normal_idx)

    total_samples = len(y)
    target_client_size = total_samples // num_clients

    # Fraud örneklerini ratio'lara göre böl
    fraud_counts = [int(len(fraud_idx) * r) for r in fraud_ratios]

    # Yuvarlama farkını son client'a ekle
    fraud_counts[-1] = len(fraud_idx) - sum(fraud_counts[:-1])

    clients = []
    fraud_start = 0
    normal_start = 0

    for i in range(num_clients):
        fraud_count = fraud_counts[i]
        fraud_end = fraud_start + fraud_count

        client_fraud_idx = fraud_idx[fraud_start:fraud_end]

        # Client boyutunu yaklaşık eşit tutmak için geri kalanı normal ile doldur
        normal_count = target_client_size - len(client_fraud_idx)

        # Son client kalan tüm normal örnekleri alsın
        if i == num_clients - 1:
            client_normal_idx = normal_idx[normal_start:]
        else:
            normal_end = normal_start + normal_count
            client_normal_idx = normal_idx[normal_start:normal_end]
            normal_start = normal_end

        fraud_start = fraud_end

        client_idx = np.concatenate([client_fraud_idx, client_normal_idx])

        if shuffle_within_client:
            np.random.shuffle(client_idx)

        client_X = X[client_idx]
        client_y = y[client_idx]

        clients.append((client_X, client_y))

    return clients

def print_client_statistics(clients, amount_column_index=-1):
    for i, (client_X, client_y) in enumerate(clients, start=1):
        fraud_count = int((client_y == 1).sum())
        normal_count = int((client_y == 0).sum())
        fraud_ratio = fraud_count / len(client_y)

        amount_values = client_X[:, amount_column_index]

        print(f"\nClient {i}")
        print(f"Samples      : {len(client_y)}")
        print(f"Normal       : {normal_count}")
        print(f"Fraud        : {fraud_count}")
        print(f"Fraud Ratio  : {fraud_ratio:.6f}")
        print(f"Amount Mean  : {amount_values.mean():.4f}")
        print(f"Amount Min   : {amount_values.min():.4f}")
        print(f"Amount Max   : {amount_values.max():.4f}")