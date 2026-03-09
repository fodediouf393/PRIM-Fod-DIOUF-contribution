import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)

@torch.no_grad()
def compute_metrics_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    thr: float = 0.5,
    eps: float = 1e-6,
):
    """
    probs:   tensor of probabilities in [0,1], any shape
    targets: tensor same shape, values in {0,1}
    Metrics are computed pixel-wise on flattened arrays.

    Returns dict: dice, iou, mcc, precision, recall, accuracy, auc
    """
    probs_f = probs.detach().view(-1).cpu().numpy().astype(np.float32)
    targets_f = targets.detach().view(-1).cpu().numpy().astype(np.uint8)
    preds_f = (probs_f > thr).astype(np.uint8)

    inter = (preds_f * targets_f).sum()
    union = preds_f.sum() + targets_f.sum() - inter

    dice = (2.0 * inter + eps) / (preds_f.sum() + targets_f.sum() + eps)
    iou = (inter + eps) / (union + eps)

    mcc = matthews_corrcoef(targets_f, preds_f)
    precision = precision_score(targets_f, preds_f, zero_division=0)
    recall = recall_score(targets_f, preds_f, zero_division=0)
    accuracy = accuracy_score(targets_f, preds_f)

    # AUC only if both classes exist
    if len(np.unique(targets_f)) == 2:
        try:
            auc = roc_auc_score(targets_f, probs_f)
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")

    return {
        "dice": float(dice),
        "iou": float(iou),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auc": float(auc),
    }
# --- ADD BELOW to src/common/metrics.py ---
import torch
import torch.nn.functional as F


def _soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skel(img, iters: int):
    """
    img: tensor in [0,1], shape (B,1,H,W)
    """
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters - 1):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


@torch.no_grad()
def cldice_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    thr: float = 0.5,
    skel_iters: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    clDice metric computed from probabilities and targets.
    We binarize probs with thr then compute soft skeletons.

    probs:   (B,1,H,W) in [0,1]
    targets: (B,1,H,W) in {0,1}
    """
    pred = (probs > thr).float()
    gt = (targets > 0.5).float()

    skel_p = soft_skel(pred, skel_iters)
    skel_g = soft_skel(gt, skel_iters)

    tprec = (skel_p * gt).sum() / (skel_p.sum() + eps)
    tsens = (skel_g * pred).sum() / (skel_g.sum() + eps)

    cldice = (2.0 * tprec * tsens) / (tprec + tsens + eps)
    return float(cldice.item())