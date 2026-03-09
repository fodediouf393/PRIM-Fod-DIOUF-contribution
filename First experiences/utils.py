# utils.py

import math
import random
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)

from tqdm.auto import tqdm

# Initialisation globale de la BCE
criterion_bce = nn.BCEWithLogitsLoss()


def set_seed(seed: int):
    # Fixe toutes les graines aléatoires pour la reproductibilité
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coefficient_np(y_true: np.ndarray, y_pred_bin: np.ndarray, eps: float = 1e-7):
    # Dice coefficient non différentiable pour les métriques (numpy)
    assert y_true.shape == y_pred_bin.shape
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice


def mean_and_ci(values, alpha=0.95):
    # Calcule la moyenne et la demi-largeur d'un intervalle de confiance à 95%
    values = np.array(values, dtype=np.float64)
    n = len(values)
    mean = values.mean()
    if n > 1:
        std = values.std(ddof=1)
        z = 1.96
        half_width = z * std / math.sqrt(n)
    else:
        half_width = 0.0
    return mean, half_width


def dice_loss_from_logits(logits, targets, eps: float = 1e-7):
    # Dice loss différentiable (1 - Dice) à partir des logits
    probs = torch.sigmoid(logits)
    targets = targets.float()

    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dims)
    cardinality = (probs + targets).sum(dims)

    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    dice_loss = 1.0 - dice_score

    return dice_loss.mean()


def combined_loss(logits, targets):
    # Loss totale = BCEWithLogitsLoss + DiceLoss
    bce = criterion_bce(logits, targets)
    dsc = dice_loss_from_logits(logits, targets)
    return bce + dsc


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, desc="Eval"):
    # Évalue le modèle sur un DataLoader et renvoie toutes les métriques
    model.eval()

    all_probs = []
    all_targets = []
    running_loss = 0.0

    for imgs, masks, _ in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss = combined_loss(logits, masks)
        running_loss += loss.item() * imgs.size(0)

        probs = torch.sigmoid(logits)

        probs_flat = probs.detach().cpu().view(-1).numpy()
        targets_flat = masks.detach().cpu().view(-1).numpy()

        all_probs.append(probs_flat)
        all_targets.append(targets_flat)

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # AUC
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float("nan")

    # Binarisation
    preds_bin = (all_probs >= threshold).astype(np.uint8)
    y_true = all_targets.astype(np.uint8)

    # Dice
    dice = dice_coefficient_np(y_true, preds_bin)

    # MCC
    try:
        mcc = matthews_corrcoef(y_true, preds_bin)
    except Exception:
        mcc = float("nan")

    # Précision, rappel, accuracy
    precision = precision_score(y_true, preds_bin, zero_division=0)
    recall = recall_score(y_true, preds_bin, zero_division=0)
    accuracy = accuracy_score(y_true, preds_bin)

    avg_loss = running_loss / len(loader.dataset)

    metrics = {
        "loss": avg_loss,
        "auc": auc,
        "dice": dice,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }
    return metrics
