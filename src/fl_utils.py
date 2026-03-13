import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix


def get_dataloader(X, y, batch_size=256, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_local(model, X, y, epochs=2, lr=1e-3, batch_size=256, device="cpu"):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    loader = get_dataloader(X, y, batch_size=batch_size, shuffle=True)

    # Class imbalance için pos_weight
    fraud_count = max((y == 1).sum(), 1)
    normal_count = max((y == 0).sum(), 1)
    pos_weight = torch.tensor([normal_count / fraud_count], dtype=torch.float32).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def fedavg(state_dicts):
    avg_state = copy.deepcopy(state_dicts[0])

    for key in avg_state.keys():
        for i in range(1, len(state_dicts)):
            avg_state[key] += state_dicts[i][key]
        avg_state[key] = avg_state[key] / len(state_dicts)

    return avg_state


def evaluate_model(model, X_test, y_test, device="cpu", threshold=0.8):
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs)
    }

    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)

    return metrics