#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml

# -------------------------
# PATHS (adapt if needed)
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

# sample IDs requested (these are the filenames stems usually)
SAMPLE_IDS = [247, 250, 272, 284, 300]

# ViT model keys (prefixes in run names)
VIT_KEYS = ["swinunetr", "unetr", "transunet", "sswdual", "swinunet"]


# -------------------------
# utils
# -------------------------
def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img)

def find_image_file(folder: Path, sid: int) -> Path:
    """
    Tries: sid.png, sid.jpg, 10000+sid.png, etc.
    """
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

    # fallback: try glob contains sid
    gl = list(folder.glob(f"*{sid}*"))
    if gl:
        return gl[0]
    raise FileNotFoundError(f"Cannot find file for id={sid} in {folder}")

def pad_to(x: torch.Tensor, target: int) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """
    x: (C,H,W) or (1,C,H,W)
    return padded and pad tuple (left,right,top,bottom)
    """
    if x.dim() == 3:
        C,H,W = x.shape
    else:
        _,C,H,W = x.shape
    pad_h = max(0, target - H)
    pad_w = max(0, target - W)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if x.dim() == 3:
        x = F.pad(x, (left, right, top, bottom))
    else:
        x = F.pad(x, (left, right, top, bottom))
    return x, (left, right, top, bottom)

def unpad(x: np.ndarray, pad: Tuple[int,int,int,int]) -> np.ndarray:
    left,right,top,bottom = pad
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return x
    H, W = x.shape
    return x[top:H-bottom, left:W-right]

def load_norm_stats(path: Path) -> Dict:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # expected keys like mean/std lists
    # support both:
    #  - {"mean":[...],"std":[...]}
    #  - {"channels":{"mean":[...],"std":[...]}}
    if "mean" in d and "std" in d:
        return {"mean": d["mean"], "std": d["std"]}
    if "channels" in d and "mean" in d["channels"]:
        return {"mean": d["channels"]["mean"], "std": d["channels"]["std"]}
    raise ValueError(f"Unrecognized norm yaml format: {path}")

def normalize_3ch(x: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    # x: (3,H,W) float in [0,1]
    mean_t = torch.tensor(mean, device=x.device).view(3,1,1)
    std_t  = torch.tensor(std, device=x.device).view(3,1,1)
    return (x - mean_t) / (std_t + 1e-8)

def bin_to_white(pred: np.ndarray) -> Image.Image:
    # pred: HxW {0,1}
    img = (pred.astype(np.uint8) * 255)
    return Image.fromarray(img, mode="L")

def gt_to_green(mask: np.ndarray) -> Image.Image:
    # mask: HxW {0,1}
    g = (mask.astype(np.uint8) * 255)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[...,1] = g
    return Image.fromarray(rgb, mode="RGB")

def gray_to_rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")

def get_font(size: int):
    # Use a default font; if you have a ttf you can set it here
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


# -------------------------
# Models (fixed to your known ViT classes)
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
    # For SSW_Dual lazy Swin blocks
    if model_key == "sswdual":
        model.eval()
        _ = model(x_1bchw)

def load_state_dict_strict(model, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)


# -------------------------
# Discover ViT loss2 runs
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

def discover_vit_loss2_runs() -> List[Tuple[str,str,Path,Path]]:
    """
    returns list of (model_key, pretty_name, run_dir, best_ckpt)
    only loss2
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
        if loss != "loss2":
            continue
        mk = detect_model_key(rd.name)
        if mk == "unknown":
            continue
        ck = rd / "best_model" / "best_model.pth"
        if not ck.exists():
            continue
        out.append((mk, pretty.get(mk, mk.upper()), rd, ck))
    return out


# -------------------------
# Inference for one sample
# -------------------------
@torch.no_grad()
def predict_one(model, x_3hw: torch.Tensor) -> np.ndarray:
    """
    x_3hw: float tensor normalized, padded to 448
    returns binary mask (H,W) after sigmoid+threshold, BEFORE unpad (still padded size)
    """
    x = x_3hw.unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    logits = model(x)
    probs = torch.sigmoid(logits)
    pred = (probs >= THRESH).to(torch.uint8)[0,0].detach().cpu().numpy()
    return pred


def main():
    norm = load_norm_stats(NORM_YAML)
    mean, std = norm["mean"], norm["std"]

    vit_runs = discover_vit_loss2_runs()
    if not vit_runs:
        raise RuntimeError("No ViT loss2 runs found in experiments/runs")

    print("Found ViT loss2 runs:")
    for mk, pn, rd, ck in vit_runs:
        print(" -", pn, "->", rd.name)

    # Load all models once
    models = []
    for mk, pn, rd, ck in vit_runs:
        m = load_model(mk).to(DEVICE)
        # warmup needs an input; we will warmup per-sample before strict load for sswdual
        models.append((mk, pn, m, ck))

    # Build mosaic canvas sizes
    # We'll use original image size (400x400 assumed)
    sample0_ilm = load_gray(find_image_file(DIR_ILM, SAMPLE_IDS[0]))
    H0, W0 = sample0_ilm.shape
    cell_w, cell_h = W0, H0

    # columns: ILM + Nmodels + GT
    n_models = len(models)
    n_cols = 1 + n_models + 1
    n_rows = len(SAMPLE_IDS)

    # header space
    header_h = 42
    pad = 4  # between cells
    canvas_w = n_cols * cell_w + (n_cols + 1) * pad
    canvas_h = header_h + n_rows * cell_h + (n_rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(22)
    id_font = get_font(18)

    # Column titles
    titles = ["ILM"] + [pn for _, pn, _, _ in models] + ["GT"]
    for j, t in enumerate(titles):
        x0 = pad + j * (cell_w + pad)
        draw.text((x0 + 8, 8), t, fill=(255, 255, 0), font=title_font)

    # Iterate samples
    for i, sid in enumerate(tqdm(SAMPLE_IDS, desc="Samples")):
        # load images
        p_full = find_image_file(DIR_FULL, sid)
        p_ilm  = find_image_file(DIR_ILM, sid)
        p_opl  = find_image_file(DIR_OPL, sid)
        p_gt   = find_image_file(DIR_GT, sid)

        im_full = load_gray(p_full)  # HxW
        im_ilm  = load_gray(p_ilm)
        im_opl  = load_gray(p_opl)
        gt      = load_gray(p_gt)
        gt_bin  = (gt >= 128).astype(np.uint8)

        # build 3ch float in [0,1]
        x = np.stack([im_full, im_ilm, im_opl], axis=0).astype(np.float32) / 255.0
        x_t = torch.from_numpy(x)  # (3,H,W)

        # pad to 448 then normalize (like dataset)
        x_pad, pad_tuple = pad_to(x_t, PAD_TO)
        x_pad = normalize_3ch(x_pad.to(DEVICE), mean, std).cpu()  # keep normalized on CPU for warmup/predict

        # Row position
        y0 = header_h + pad + i * (cell_h + pad)

        # ----- col 0: ILM image -----
        x0 = pad
        canvas.paste(gray_to_rgb(im_ilm), (x0, y0))

        # ID text in yellow (like example)
        # emulate "ID 247" and "10247" if file is 10000+sid, otherwise show sid twice
        show_id = sid
        alt_id = sid + 10000
        # if file we loaded is 10000+sid, show both; else just sid and sid
        stem = int(Path(p_ilm).stem) if Path(p_ilm).stem.isdigit() else sid
        if stem >= 10000:
            show_id = stem - 10000
            alt_id = stem
        draw.text((x0 + 8, y0 + 8), f"ID  {show_id}", fill=(255, 255, 0), font=id_font)
        draw.text((x0 + 8, y0 + 30), f"{alt_id}", fill=(255, 255, 0), font=id_font)

        # ----- middle cols: model preds -----
        for j, (mk, pn, model, ckpt_path) in enumerate(models, start=1):
            # warmup then strict load once per model (first time only)
            # We do it lazily to avoid needing a dummy input of correct shape earlier
            if not hasattr(model, "_loaded_ckpt"):
                # warmup for sswdual
                warmup_if_needed(mk, model, x_pad.unsqueeze(0).to(DEVICE))
                load_state_dict_strict(model, ckpt_path)
                model.eval()
                model._loaded_ckpt = True

            pred_pad = predict_one(model, x_pad)            # padded size
            pred = unpad(pred_pad, pad_tuple)               # back to original size
            pred_img = bin_to_white(pred)

            x_cell = pad + j * (cell_w + pad)
            canvas.paste(pred_img.convert("RGB"), (x_cell, y0))

        # ----- last col: GT green -----
        x_last = pad + (n_cols - 1) * (cell_w + pad)
        canvas.paste(gt_to_green(gt_bin), (x_last, y0))

    # Save
    out_path = OUT_DIR / "mosaic_vit_loss2_ids_247_250_272_284_300.png"
    canvas.save(out_path)
    print("\nSaved mosaic to:", out_path)


if __name__ == "__main__":
    main()

