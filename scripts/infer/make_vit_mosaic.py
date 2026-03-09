#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml


# -------------------------
# PATHS
# -------------------------
PROJECT = Path("/home/infres/diouf-25/prim-project")
RUNS_ROOT = PROJECT / "experiments" / "runs"
OUT_DIR = PROJECT / "experiments" / "mosaics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3-channel full images dirs (400x400)
DIR_FULL = PROJECT / "data" / "OCTA(FULL)"
DIR_ILM  = PROJECT / "data" / "OCTA(ILM_OPL)"
DIR_OPL  = PROJECT / "data" / "OCTA(OPL_BM)"
DIR_GT   = PROJECT / "data" / "GT_Capillary"

NORM_YAML = PROJECT / "configs" / "norm_fullimg.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5
PAD_TO = 448

SAMPLE_IDS_DEFAULT = [247, 250, 272, 284, 300]

VIT_KEYS = ["swinunetr", "unetr", "transunet", "sswdual", "swinunet"]


# -------------------------
# utils
# -------------------------
def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img)

def find_image_file(folder: Path, sid: int) -> Path:
    candidates = [
        folder / f"{sid}.png",
        folder / f"{sid}.jpg",
        folder / f"{sid}.jpeg",
        folder / f"{sid:05d}.png",
        folder / f"{sid:05d}.jpg",
        folder / f"{sid+10000}.png",
        folder / f"{sid+10000}.jpg",
        folder / f"{sid+10000}.jpeg",
        folder / f"{sid+10000:05d}.png",
    ]
    for p in candidates:
        if p.exists():
            return p

    gl = list(folder.glob(f"*{sid}*"))
    if gl:
        return gl[0]

    raise FileNotFoundError(f"Cannot find file for id={sid} in {folder}")

def pad_to(x: torch.Tensor, target: int):
    """
    x: (C,H,W)
    """
    _, H, W = x.shape
    pad_h = max(0, target - H)
    pad_w = max(0, target - W)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    x = F.pad(x, (left, right, top, bottom))
    return x, (left, right, top, bottom)

def unpad(x: np.ndarray, pad):
    left, right, top, bottom = pad
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return x
    H, W = x.shape
    return x[top:H-bottom, left:W-right]

def load_norm_stats(path: Path) -> Dict:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    if "mean" in d and "std" in d:
        return {"mean": d["mean"], "std": d["std"]}
    if "channels" in d and "mean" in d["channels"]:
        return {"mean": d["channels"]["mean"], "std": d["channels"]["std"]}
    raise ValueError(f"Unrecognized norm yaml format: {path}")

def normalize_3ch(x: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device).view(3, 1, 1)
    std_t  = torch.tensor(std, device=x.device).view(3, 1, 1)
    return (x - mean_t) / (std_t + 1e-8)

def bin_to_white(pred: np.ndarray) -> Image.Image:
    img = (pred.astype(np.uint8) * 255)
    return Image.fromarray(img, mode="L")

def gt_to_green(mask: np.ndarray) -> Image.Image:
    g = (mask.astype(np.uint8) * 255)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[..., 1] = g
    return Image.fromarray(rgb, mode="RGB")

def gray_to_rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")

def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


# -------------------------
# Models
# -------------------------
def load_model(model_key: str):
    if model_key in ("swinunetr", "swinunet"):
        from src.architectures.VisualTransformers.SwinUnet.models.swin_unet import SwinUNet
        return SwinUNet(in_channels=3, n_classes=1, feature_size=48)

    if model_key == "unetr":
        from src.architectures.VisualTransformers.UNETR.models.unetr import UNETR2D
        return UNETR2D(in_channels=3, n_classes=1, img_size=448, patch_size=8)

    if model_key == "transunet":
        from src.architectures.VisualTransformers.TransUNet.models.transunet import TransUNet
        return TransUNet(in_channels=3, n_classes=1, img_size=448)

    if model_key == "sswdual":
        from src.architectures.VisualTransformers.SSW_Dual.models.ssw_dual import SSW_Dual
        return SSW_Dual(
            img_ch=3, output_ch=1,
            rate=48, layer_depth=4,
            kernel_size=9, extend_scope=3.0,
            window_size=7, dropout=0.0, repeat_n=1
        )

    raise KeyError(model_key)

def warmup_if_needed(model_key: str, model: torch.nn.Module, x_1bchw: torch.Tensor):
    """
    SSW_Dual: blocks are created lazily in forward -> warmup first.
    """
    if model_key == "sswdual":
        model.eval()
        _ = model(x_1bchw)

def load_state_dict_strict(model, ckpt_path: Path):
    """
    Loads checkpoint with weights_only=True when possible to remove warning.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        # older torch without weights_only
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)


# -------------------------
# Discover runs
# -------------------------
def detect_model_key(run_name: str) -> str:
    rn = run_name.lower()
    for k in sorted(VIT_KEYS, key=len, reverse=True):
        if rn.startswith(k):
            return k
    return "unknown"

def detect_loss(run_name: str) -> str:
    m = re.search(r"(loss[123])", run_name.lower())
    return m.group(1) if m else "loss?"

def discover_vit_runs(loss_filter: str) -> List[Tuple[str, str, str, Path]]:
    """
    returns list of (model_key, pretty_name, loss, best_ckpt)
    """
    pretty = {
        "swinunetr": "SwinUNETR",
        "swinunet": "SwinUNETR",
        "unetr": "UNETR",
        "transunet": "TransUNet",
        "sswdual": "SSW-Dual",
    }
    out = []
    for rd in sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()]):
        loss = detect_loss(rd.name)
        if loss_filter != "all" and loss != loss_filter:
            continue

        mk = detect_model_key(rd.name)
        if mk == "unknown":
            continue

        ck = rd / "best_model" / "best_model.pth"
        if not ck.exists():
            continue

        out.append((mk, pretty.get(mk, mk.upper()), loss, ck))
    return out


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def predict_one(model, x_3hw: torch.Tensor) -> np.ndarray:
    """
    x_3hw: (3,H,W) normalized & padded
    returns padded binary mask (H,W)
    """
    x = x_3hw.unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    logits = model(x)
    probs = torch.sigmoid(logits)
    pred = (probs >= THRESH).to(torch.uint8)[0, 0].detach().cpu().numpy()
    return pred


def make_mosaic_for_loss(loss_name: str, sample_ids: List[int]):
    norm = load_norm_stats(NORM_YAML)
    mean, std = norm["mean"], norm["std"]

    vit_runs = discover_vit_runs(loss_name)
    if not vit_runs:
        print(f"[WARN] No ViT runs found for {loss_name}")
        return

    # cell size from an ILM sample (assumed 400x400)
    sample0_ilm = load_gray(find_image_file(DIR_ILM, sample_ids[0]))
    H0, W0 = sample0_ilm.shape
    cell_w, cell_h = W0, H0

    # columns: ILM + models + GT
    n_models = len(vit_runs)
    n_cols = 1 + n_models + 1
    n_rows = len(sample_ids)

    header_h = 42
    pad = 4
    canvas_w = n_cols * cell_w + (n_cols + 1) * pad
    canvas_h = header_h + n_rows * cell_h + (n_rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(22)
    id_font = get_font(18)

    titles = ["ILM"] + [pn for _, pn, _, _ in vit_runs] + ["GT"]
    for j, t in enumerate(titles):
        x0 = pad + j * (cell_w + pad)
        draw.text((x0 + 8, 8), t, fill=(255, 255, 0), font=title_font)

    # Load models once for this mosaic
    models = []
    for mk, pn, loss, ck in vit_runs:
        m = load_model(mk).to(DEVICE)
        models.append((mk, pn, m, ck))

    # For each sample
    for i, sid in enumerate(tqdm(sample_ids, desc=f"Samples {loss_name}")):
        p_full = find_image_file(DIR_FULL, sid)
        p_ilm  = find_image_file(DIR_ILM, sid)
        p_opl  = find_image_file(DIR_OPL, sid)
        p_gt   = find_image_file(DIR_GT, sid)

        im_full = load_gray(p_full)
        im_ilm  = load_gray(p_ilm)
        im_opl  = load_gray(p_opl)
        gt      = load_gray(p_gt)
        gt_bin  = (gt >= 128).astype(np.uint8)

        # 3ch float in [0,1]
        x = np.stack([im_full, im_ilm, im_opl], axis=0).astype(np.float32) / 255.0
        x_t = torch.from_numpy(x)  # (3,H,W)

        # pad then normalize (as in dataset)
        x_pad, pad_tuple = pad_to(x_t, PAD_TO)
        x_pad = normalize_3ch(x_pad.to(DEVICE), mean, std).cpu()

        y0 = header_h + pad + i * (cell_h + pad)

        # col 0: ILM
        x0 = pad
        canvas.paste(gray_to_rgb(im_ilm), (x0, y0))

        # ID text in yellow like example
        stem = Path(p_ilm).stem
        try:
            stem_int = int(stem)
        except Exception:
            stem_int = sid

        if stem_int >= 10000:
            show_id = stem_int - 10000
            alt_id = stem_int
        else:
            show_id = sid
            alt_id = sid + 10000

        draw.text((x0 + 8, y0 + 8), f"ID  {show_id}", fill=(255, 255, 0), font=id_font)
        draw.text((x0 + 8, y0 + 30), f"{alt_id}", fill=(255, 255, 0), font=id_font)

        # predictions
        for j, (mk, pn, model, ckpt_path) in enumerate(models, start=1):
            if not hasattr(model, "_loaded_ckpt"):
                # warmup for lazy modules (SSW_Dual)
                warmup_if_needed(mk, model, x_pad.unsqueeze(0).to(DEVICE))
                load_state_dict_strict(model, ckpt_path)
                model.eval()
                model._loaded_ckpt = True

            pred_pad = predict_one(model, x_pad)
            pred = unpad(pred_pad, pad_tuple)
            pred_img = bin_to_white(pred)

            x_cell = pad + j * (cell_w + pad)
            canvas.paste(pred_img.convert("RGB"), (x_cell, y0))

        # last col: GT green
        x_last = pad + (n_cols - 1) * (cell_w + pad)
        canvas.paste(gt_to_green(gt_bin), (x_last, y0))

    # ✅ FIXED PATH BUILD
    out_path = OUT_DIR / f"mosaic_vit_{loss_name}_ids_{'_'.join(map(str, sample_ids))}.png"
    canvas.save(out_path)
    print(f"\n✅ Saved mosaic: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="loss2", choices=["loss1", "loss2", "loss3", "all"])
    ap.add_argument("--ids", default="247,250,272,284,300", help="comma-separated sample ids")
    args = ap.parse_args()

    sample_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
    if not sample_ids:
        sample_ids = SAMPLE_IDS_DEFAULT

    if args.loss == "all":
        for loss in ["loss1", "loss2", "loss3"]:
            make_mosaic_for_loss(loss, sample_ids)
    else:
        make_mosaic_for_loss(args.loss, sample_ids)


if __name__ == "__main__":
    main()
