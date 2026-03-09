import torch
import torch.nn.functional as F

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    den = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice.mean()

# --- clDice (soft skeleton) ---
def _soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)

def _soft_dilate(img):
    return F.max_pool2d(img, 3, 1, 1)

def _soft_open(img):
    return _soft_dilate(_soft_erode(img))

def soft_skel(img, iters=10):
    img1 = img
    skel = torch.zeros_like(img1)
    for _ in range(iters):
        opened = _soft_open(img1)
        delta = F.relu(img1 - opened)
        skel = skel + delta
        img1 = _soft_erode(img1)
    return skel

def cldice_loss_from_logits(logits, targets, iters=10, eps=1e-6):
    probs = torch.sigmoid(logits)
    skel_p = soft_skel(probs, iters)
    skel_t = soft_skel(targets, iters)

    tprec = (skel_p * targets).sum(dim=(1,2,3)) / (skel_p.sum(dim=(1,2,3)) + eps)
    tsens = (skel_t * probs).sum(dim=(1,2,3)) / (skel_t.sum(dim=(1,2,3)) + eps)
    cl_dice = (2 * tprec * tsens) / (tprec + tsens + eps)
    return 1 - cl_dice.mean()

def tversky_loss_from_logits(logits, targets, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky loss for binary segmentation.
    logits: (B,1,H,W)
    targets:(B,1,H,W) float in {0,1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (probs * targets).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - tversky.mean()


def combined_loss(
    logits,
    targets,
    w_bce=0.0,
    w_dice=1.0,
    w_cldice=0.0,
    w_tversky=0.0,
    tversky_alpha=0.7,
    tversky_beta=0.3,
    skel_iters=10,
):
    loss = 0.0
    if w_bce and w_bce > 0:
        loss = loss + w_bce * F.binary_cross_entropy_with_logits(logits, targets)
    if w_dice and w_dice > 0:
        loss = loss + w_dice * dice_loss_from_logits(logits, targets)
    if w_cldice and w_cldice > 0:
        loss = loss + w_cldice * cldice_loss_from_logits(logits, targets, iters=skel_iters)
    if w_tversky and w_tversky > 0:
        loss = loss + w_tversky * tversky_loss_from_logits(
            logits, targets, alpha=tversky_alpha, beta=tversky_beta
        )
    return loss
