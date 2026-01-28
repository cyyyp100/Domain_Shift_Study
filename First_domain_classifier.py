import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

"""# Reproductibilite + chemins"""

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# NOTE: on veut pouvoir entraîner sur une petite fraction (ex: 0.1 ou 0.2)
# pour éviter que le classifieur de domaines devienne trop performant.
TRAIN_FRACTION = 0.01   # ex: 0.1 = 10% train, 0.2 = 20% train

# Validation prise sur le TRAIN (pour early-stopping du MLP / Discriminator)
VAL_FRACTION_ON_TRAIN = 0.2

# Le test prend le reste (ex: 80% si TRAIN_FRACTION=0.2)
TEST_SIZE = 1.0 - TRAIN_FRACTION

STRATIFY = True
BALANCE_STRATEGY = "downsample"  # valeurs possibles: "none", "downsample"

# Chemins
EMBEDDINGS_ROOT = "embeddings"
OUTPUT_ROOT = "Résultats"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_FIELDS = [
    "run_id", "dataset_a", "dataset_b", "n_a", "n_b", "dim",
    "model", "accuracy", "roc_auc", "error_eps", "pad"
]

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def save_json(obj, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def append_csv_row(row_dict, output_dir: Path, filename: str, fieldnames):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in fieldnames})
    return path


def save_plot(fig, output_dir: Path, filename: str, dpi=200):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


"""# Chargement embeddings (npy)"""


def load_embedding_file(fp: Path) -> np.ndarray:
    x = np.load(fp)
    x = np.asarray(x)

    # Accepte (D,), (1,D), (D,1)
    if x.ndim == 2:
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[1] == 1:
            x = x[:, 0]
        else:
            raise ValueError(f"{fp} has shape {x.shape}, expected a single embedding.")
    elif x.ndim != 1:
        raise ValueError(f"{fp} has ndim={x.ndim}, expected 1D embedding.")

    return x.astype(np.float32)


def load_dataset_from_dir(root_dir: Path, label: int, pattern="*.npy"):
    files = sorted(root_dir.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {root_dir} with pattern {pattern}")

    X_list = []
    for fp in files:
        X_list.append(load_embedding_file(fp))

    # Verifie dimension coherente
    D = X_list[0].shape[0]
    for i, x in enumerate(X_list):
        if x.shape[0] != D:
            raise ValueError(f"Dim mismatch at {files[i]}: got {x.shape[0]}, expected {D}")

    X = np.stack(X_list, axis=0).astype(np.float32)   # (N, D)
    y = np.full((X.shape[0],), label, dtype=np.int64)  # (N,)
    return X, y, files


def compute_pad(acc: float):
    err = 1.0 - acc
    pad = 2.0 * (1.0 - 2.0 * err)
    pad = float(np.clip(pad, 0.0, 2.0))
    return float(err), float(pad)


def list_embedding_datasets(embeddings_root: Path):
    if not embeddings_root.exists():
        print(f"[WARN] Le dossier {embeddings_root} n'existe pas.")
        return []

    datasets = []
    for d in sorted(embeddings_root.iterdir()):
        if not d.is_dir():
            continue
        npy_files = list(d.glob("*.npy"))
        if len(npy_files) == 0:
            print(f"[WARN] Dossier vide (aucun .npy): {d.name}. Ignore.")
            continue
        datasets.append(d.name)
    return datasets


def apply_balance_strategy(X_a, y_a, X_b, y_b, strategy: str, seed: int):
    if strategy == "none":
        return X_a, y_a, X_b, y_b

    if strategy != "downsample":
        raise ValueError(f"BALANCE_STRATEGY invalide: {strategy}")

    n_a = X_a.shape[0]
    n_b = X_b.shape[0]
    n_min = min(n_a, n_b)

    rng = np.random.default_rng(seed)

    if n_a > n_min:
        idx_a = rng.choice(n_a, size=n_min, replace=False)
    else:
        idx_a = np.arange(n_a)

    if n_b > n_min:
        idx_b = rng.choice(n_b, size=n_min, replace=False)
    else:
        idx_b = np.arange(n_b)

    return X_a[idx_a], y_a[idx_a], X_b[idx_b], y_b[idx_b]


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DomainDiscriminator(nn.Sequential):
    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True, sigmoid=True):
        final_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() if sigmoid else nn.Identity()
        )
        super(DomainDiscriminator, self).__init__(
            nn.Linear(in_feature, hidden_size),
            nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
            nn.ReLU(),
            final_layer
        )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters()}]


def make_torch_loaders(X_train_s, y_train, X_test_s, y_test):
    strat = y_train if STRATIFY else None
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_s, y_train, test_size=VAL_FRACTION_ON_TRAIN, random_state=SEED, stratify=strat
        )
    except ValueError as e:
        print(f"[WARN] Split train/val impossible ({e}). On utilise train=val.")
        X_tr, y_tr = X_train_s, y_train
        X_val, y_val = X_train_s, y_train

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.float32)

    train_bs = max(1, min(256, len(X_tr_t)))
    val_bs = max(1, min(512, len(X_val_t)))
    test_bs = max(1, min(512, len(X_te_t)))

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=train_bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=val_bs, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=test_bs, shuffle=False)

    return train_loader, val_loader, test_loader


def eval_loader(model, loader, device: str, is_logits: bool):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            p = torch.sigmoid(out) if is_logits else out
            probs.append(p.detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy())
    probs = np.concatenate(probs).reshape(-1)
    ys = np.concatenate(ys).astype(int).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(ys, preds)
    auc = roc_auc_score(ys, probs)
    return acc, auc, probs, ys


def run_mlp(X_train_s, y_train, X_test_s, y_test, pair_dir: Path, run_root: Path,
            dataset_a: str, dataset_b: str, n_a: int, n_b: int, dim: int):
    train_loader, val_loader, test_loader = make_torch_loaders(X_train_s, y_train, X_test_s, y_test)

    model = MLP(dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience = 10
    pat = 0
    EPOCHS = 100

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        val_acc, val_auc, _, _ = eval_loader(model, val_loader, DEVICE, is_logits=True)
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | val_acc={val_acc:.4f} val_auc={val_auc:.4f} | best_val_auc={best_val_auc:.4f}")

        if pat >= patience:
            print("Early stopping.")
            break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    test_acc, test_auc, test_probs, test_ys = eval_loader(model, test_loader, DEVICE, is_logits=True)
    err, pad = compute_pad(test_acc)

    print(f"[MLP] Accuracy: {test_acc:.4f} | ROC-AUC: {test_auc:.4f} | PAD: {pad:.4f}")

    mlp_results = {
        "run_id": RUN_ID,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "model": "MLP",
        "seed": SEED,
        "test_size": TEST_SIZE,
        "val_fraction_on_train": VAL_FRACTION_ON_TRAIN,
        "epochs_max": EPOCHS,
        "best_val_auc": float(best_val_auc),
        "balance_strategy": BALANCE_STRATEGY,
        "n_a": int(n_a),
        "n_b": int(n_b),
        "embedding_dim": int(dim),
        "accuracy": float(test_acc),
        "roc_auc": float(test_auc),
        "error_eps": float(err),
        "pad": float(pad),
    }

    save_json(mlp_results, pair_dir, "metrics_mlp.json")

    fpr, tpr, thr = roc_curve(test_ys, test_probs)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - Domain classifier (MLP)")
    save_plot(fig, pair_dir, "roc_MLP.png")
    plt.close(fig)

    save_json({
        "run_id": RUN_ID,
        "model": "MLP",
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
    }, pair_dir, "roc_points_mlp.json")

    append_csv_row({
        "run_id": RUN_ID,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "n_a": int(n_a),
        "n_b": int(n_b),
        "dim": int(dim),
        "model": "MLP",
        "accuracy": float(test_acc),
        "roc_auc": float(test_auc),
        "error_eps": float(err),
        "pad": float(pad),
    }, run_root, "results_all_pairs.csv", CSV_FIELDS)


def run_domain_discriminator(X_train_s, y_train, X_test_s, y_test, pair_dir: Path, run_root: Path,
                             dataset_a: str, dataset_b: str, n_a: int, n_b: int, dim: int):
    train_loader, val_loader, test_loader = make_torch_loaders(X_train_s, y_train, X_test_s, y_test)

    use_bn = len(y_train) >= 2
    discriminator = DomainDiscriminator(in_feature=dim, hidden_size=256, batch_norm=use_bn, sigmoid=True).to(DEVICE)
    d_opt = torch.optim.AdamW(discriminator.parameters(), lr=1e-3, weight_decay=1e-4)
    d_crit = nn.BCELoss()

    best_val_auc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in discriminator.state_dict().items()}
    patience = 10
    pat = 0
    EPOCHS = 100

    print("Training Domain Discriminator...")
    for epoch in range(1, EPOCHS + 1):
        discriminator.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            d_opt.zero_grad()
            probs = discriminator(xb).view(-1)
            loss = d_crit(probs, yb)
            loss.backward()
            d_opt.step()

        val_acc, val_auc, _, _ = eval_loader(discriminator, val_loader, DEVICE, is_logits=False)
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in discriminator.state_dict().items()}
            pat = 0
        else:
            pat += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | val_acc={val_acc:.4f} val_auc={val_auc:.4f} | best_val_auc={best_val_auc:.4f}")

        if pat >= patience:
            print("Early stopping.")
            break

    discriminator.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    test_acc, test_auc, test_probs, test_ys = eval_loader(discriminator, test_loader, DEVICE, is_logits=False)
    err, pad = compute_pad(test_acc)

    print(f"[DomainDiscriminator] Accuracy: {test_acc:.4f} | ROC-AUC: {test_auc:.4f} | PAD: {pad:.4f}")

    disc_results = {
        "run_id": RUN_ID,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "model": "DomainDiscriminator",
        "seed": SEED,
        "test_size": TEST_SIZE,
        "val_fraction_on_train": VAL_FRACTION_ON_TRAIN,
        "epochs_max": EPOCHS,
        "best_val_auc": float(best_val_auc),
        "balance_strategy": BALANCE_STRATEGY,
        "n_a": int(n_a),
        "n_b": int(n_b),
        "embedding_dim": int(dim),
        "accuracy": float(test_acc),
        "roc_auc": float(test_auc),
        "error_eps": float(err),
        "pad": float(pad),
    }

    save_json(disc_results, pair_dir, "metrics_domain_discriminator.json")

    fpr, tpr, thr = roc_curve(test_ys, test_probs)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - Domain Discriminator")
    save_plot(fig, pair_dir, "roc_DomainDiscriminator.png")
    plt.close(fig)

    save_json({
        "run_id": RUN_ID,
        "model": "DomainDiscriminator",
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
    }, pair_dir, "roc_points_domain_discriminator.json")

    append_csv_row({
        "run_id": RUN_ID,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "n_a": int(n_a),
        "n_b": int(n_b),
        "dim": int(dim),
        "model": "DomainDiscriminator",
        "accuracy": float(test_acc),
        "roc_auc": float(test_auc),
        "error_eps": float(err),
        "pad": float(pad),
    }, run_root, "results_all_pairs.csv", CSV_FIELDS)


def run_pairwise_experiment(dataset_a: str, dataset_b: str, embeddings_root: Path, run_root: Path,
                            pair_index: int, total_pairs: int, skipped_pairs: List[Dict]):
    pair_name = f"{dataset_a}_vs_{dataset_b}"
    pair_dir = run_root / pair_name

    print(f"\nPair {pair_index}/{total_pairs}: {dataset_a} vs {dataset_b}")

    try:
        X_a, y_a, files_a = load_dataset_from_dir(embeddings_root / dataset_a, label=0)
        X_b, y_b, files_b = load_dataset_from_dir(embeddings_root / dataset_b, label=1)
    except FileNotFoundError as e:
        print(f"[WARN] {e}. Pair ignoree.")
        return
    except ValueError as e:
        print(f"[WARN] {pair_name} ignoree: {e}")
        skipped_pairs.append({
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "reason": str(e)
        })
        return

    dim_a = X_a.shape[1]
    dim_b = X_b.shape[1]
    if dim_a != dim_b:
        reason = f"Dim differente: {dataset_a}={dim_a}, {dataset_b}={dim_b}"
        print(f"[WARN] {pair_name} ignoree. {reason}")
        skipped_pairs.append({
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "reason": reason
        })
        return

    n_a_raw, n_b_raw = X_a.shape[0], X_b.shape[0]

    X_a, y_a, X_b, y_b = apply_balance_strategy(X_a, y_a, X_b, y_b, BALANCE_STRATEGY, SEED)

    n_a, n_b = X_a.shape[0], X_b.shape[0]
    if min(n_a, n_b) < 2:
        reason = "Pas assez d'exemples apres equilibrage"
        print(f"[WARN] {pair_name} ignoree. {reason}")
        skipped_pairs.append({
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "reason": reason
        })
        return

    X = np.vstack([X_a, X_b]).astype(np.float32)
    y = np.concatenate([y_a, y_b]).astype(np.int64)

    print(f"Tailles brutes: n_a={n_a_raw}, n_b={n_b_raw} | apres equilibrage: n_a={n_a}, n_b={n_b} | dim={dim_a}")

    strat = y if STRATIFY else None
    try:
        # On fixe explicitement la taille du TRAIN (petite) plutôt que de raisonner en 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=TRAIN_FRACTION,
            random_state=SEED,
            stratify=strat,
            shuffle=True,
        )
    except ValueError as e:
        print(f"[WARN] Split train/test impossible pour {pair_name}: {e}")
        skipped_pairs.append({
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "reason": f"Split train/test impossible: {e}"
        })
        return

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Train (fraction={TRAIN_FRACTION:.2f}):", X_train_s.shape, np.bincount(y_train))
    print(f"Test  (fraction={1.0-TRAIN_FRACTION:.2f}):", X_test_s.shape, np.bincount(y_test))

    """# Logistic Regression"""

    logreg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )
    logreg.fit(X_train_s, y_train)

    proba_test = logreg.predict_proba(X_test_s)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred_test)
    auc = roc_auc_score(y_test, proba_test)
    err, pad = compute_pad(acc)

    print(f"[LogReg] Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | PAD: {pad:.4f}")
    print("[LogReg] Probas (test) sample:", np.round(proba_test[:10], 4).tolist())

    logreg_results = {
        "run_id": RUN_ID,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "model": "LogReg",
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "test_fraction": 1.0 - TRAIN_FRACTION,
        "val_fraction_on_train": VAL_FRACTION_ON_TRAIN,
        "balance_strategy": BALANCE_STRATEGY,
        "n_a_raw": int(n_a_raw),
        "n_b_raw": int(n_b_raw),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "embedding_dim": int(dim_a),
        "train_counts": {"0": int((y_train == 0).sum()), "1": int((y_train == 1).sum())},
        "test_counts": {"0": int((y_test == 0).sum()), "1": int((y_test == 1).sum())},
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "error_eps": float(err),
        "pad": float(pad),
    }

    pair_dir.mkdir(parents=True, exist_ok=True)
    save_json(logreg_results, pair_dir, "metrics_logreg.json")

    save_json({
        "run_id": RUN_ID,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "proba_test": proba_test.tolist(),
        "y_test": y_test.tolist(),
    }, pair_dir, "proba_logreg.json")

    fpr, tpr, thr = roc_curve(y_test, proba_test)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - Domain classifier (LogReg)")
    save_plot(fig, pair_dir, "roc_logreg.png")
    plt.close(fig)

    save_json({
        "run_id": RUN_ID,
        "model": "LogReg",
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
    }, pair_dir, "roc_points_logreg.json")

    append_csv_row({
        "run_id": RUN_ID,
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "n_a": int(n_a),
        "n_b": int(n_b),
        "dim": int(dim_a),
        "model": "LogReg",
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "error_eps": float(err),
        "pad": float(pad),
    }, run_root, "results_all_pairs.csv", CSV_FIELDS)

    """# MLP + Domain Discriminator"""

    run_mlp(X_train_s, y_train, X_test_s, y_test, pair_dir, run_root, dataset_a, dataset_b, n_a, n_b, dim_a)
    run_domain_discriminator(X_train_s, y_train, X_test_s, y_test, pair_dir, run_root, dataset_a, dataset_b, n_a, n_b, dim_a)


def main():
    embeddings_root = Path(EMBEDDINGS_ROOT)
    run_root = Path(OUTPUT_ROOT) / RUN_ID
    run_root.mkdir(parents=True, exist_ok=True)

    print("device:", DEVICE)
    datasets = list_embedding_datasets(embeddings_root)

    if len(datasets) < 2:
        print("[WARN] Pas assez de datasets valides pour faire des paires.")
        return

    total_pairs = len(datasets) * (len(datasets) - 1) // 2
    skipped_pairs = []

    pair_index = 0
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            pair_index += 1
            run_pairwise_experiment(
                datasets[i], datasets[j], embeddings_root, run_root,
                pair_index, total_pairs, skipped_pairs
            )

    save_json({
        "run_id": RUN_ID,
        "skipped_pairs": skipped_pairs
    }, run_root, "skipped_pairs.json")

    print(f"\nSaved outputs to: {run_root.resolve()}")


if __name__ == "__main__":
    main()
