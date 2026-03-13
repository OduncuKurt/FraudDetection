# Threshold deneyi halen aktif. Unutma!

from src.data_prep import (
    load_and_preprocess_data,
    split_into_clients_noniid_label_skew,
    print_client_statistics
)
from src.model import MLPFraudDetector
from src.fl_utils import train_local, fedavg, evaluate_model

import torch
from sklearn.metrics import confusion_matrix


def evaluate_thresholds(model, X_test, y_test, thresholds=None, device="cpu"):
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.8, 0.9]

    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        metrics = evaluate_model(model, X_test, y_test, threshold=threshold, device=device)

        print(f"\n=== Threshold: {threshold} ===")
        print("Confusion Matrix:")
        print(cm)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


def main():
    csv_path = "data/creditcard.csv"
    num_clients = 4
    num_rounds = 5
    local_epochs = 2
    lr = 1e-3
    device = "cpu"

    print("Loading data...")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(csv_path)

    print("Splitting into clients...")
    #clients = split_into_clients(X_train, y_train, num_clients=num_clients)
    #clients = split_into_clients_noniid_amount(
    #    X_train,
    #    y_train,
    #    num_clients=num_clients,
    #    amount_column_index=-1
    #)
    clients = split_into_clients_noniid_label_skew(
        X_train,
        y_train,
        num_clients=num_clients,
        fraud_ratios=[0.5, 0.25, 0.15, 0.10]
)

    input_dim = X_train.shape[1]
    global_model = MLPFraudDetector(input_dim=input_dim)

    for rnd in range(num_rounds):
        print(f"\n--- Federated Round {rnd + 1}/{num_rounds} ---")

        local_states = []
        for client_id, (client_X, client_y) in enumerate(clients, start=1):
            print(f"Training client {client_id} on {len(client_X)} samples...")
            local_state = train_local(
                model=global_model,
                X=client_X,
                y=client_y,
                epochs=local_epochs,
                lr=lr,
                device=device
            )
            local_states.append(local_state)

        new_global_state = fedavg(local_states)
        global_model.load_state_dict(new_global_state)

        print("\nThreshold = 0.5 sonuçları:")
        preds_metrics = evaluate_model(global_model, X_test, y_test, threshold=0.5, device=device)

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = global_model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

        cm = confusion_matrix(y_test, preds)
        print("Confusion Matrix:")
        print(cm)

        print("Global model metrics:")
        for k, v in preds_metrics.items():
            print(f"{k}: {v:.4f}")

    print("\n==============================")
    print("Final global model threshold analysis")
    print("==============================")
    evaluate_thresholds(
        global_model,
        X_test,
        y_test,
        thresholds=[0.5, 0.7, 0.8, 0.9],
        device=device
    )

    print_client_statistics(clients, amount_column_index=-1)


if __name__ == "__main__":
    main()