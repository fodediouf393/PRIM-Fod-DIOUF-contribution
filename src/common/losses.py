import torch
import torch.nn.functional as F

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)

def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))

def soft_skel(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters - 1):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = torch.sum(probs * targets, dims)
    denom = torch.sum(probs + targets, dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

def cldice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, skel_iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    skel_pred = soft_skel(probs, iters=skel_iters)
    skel_true = soft_skel(targets, iters=skel_iters)

    tprec = (torch.sum(skel_pred * targets) + eps) / (torch.sum(skel_pred) + eps)
    tsens = (torch.sum(skel_true * probs) + eps) / (torch.sum(skel_true) + eps)
    cldice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return 1.0 - cldice

def combined_paper_loss(logits: torch.Tensor, targets: torch.Tensor, skel_iters: int = 10) -> torch.Tensor:
    return 0.8 * dice_loss_from_logits(logits, targets) + 0.2 * cldice_loss_from_logits(
        logits, targets, skel_iters=skel_iters
    )
