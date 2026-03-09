import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def _safe_auc(y_true, y_score):
    y_true = y_true.astype(np.int32)
    # AUC undefined if only one class present
    if y_true.max() == y_true.min():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

@torch.no_grad()
def compute_metrics_from_logits(logits, targets, thr=0.5):
    """
    logits: (B,1,H,W), targets: (B,1,H,W) in {0,1}
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y = targets.detach().cpu().numpy().reshape(-1).astype(np.uint8)
    pred = (probs >= thr).astype(np.uint8)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    eps = 1e-8
    dice = (2*tp + eps) / (2*tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    # MCC
    mcc_num = (tp * tn - fp * fn)
    mcc_den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + eps)
    mcc = float(mcc_num / (mcc_den + eps))

    auc = _safe_auc(y, probs)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "accuracy": float(accuracy),
    }
