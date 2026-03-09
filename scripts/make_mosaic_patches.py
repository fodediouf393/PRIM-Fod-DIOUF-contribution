# scripts/make_mosaic_patches.py
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = "/home/infres/diouf-25/prim-project"

PATCH_IDS = [964, 966, 972, 1082, 1117]

# Where inference outputs are stored
PRED_ROOT = f"{PROJECT_ROOT}/experiments/inference_outputs"

# 7 models: short name + relative dir for PATCH predictions
MODELS = [
    ("UNet",   "unet_3ch/seed_0/patches_bin"),
    ("UNet++", "unetpp_3ch/seed_0/patches_bin"),
    ("U++DS",  "unetpp_DS_3ch/seed_0/patches_bin"),
    ("ResU",   "resunet_3ch/seed_0/patches_bin"),
    ("R2U",    "r2unet_3ch/seed_0/patches_bin"),
    ("U3+",    "unet3plus_3ch/seed_0/patches_bin"),
    ("Attn",   "attention_unet_3ch/seed_0/patches_bin"),
]

# Patch images for ILM_OPL (the input patches)
# IMPORTANT: this must be your patch folder (256x256)
ILM_OPL_PATCH_DIR = f"{PROJECT_ROOT}/data/overlap_patches_ilm_opl"

# Patch labels (0/255)
LABEL_PATCH_DIR = f"{PROJECT_ROOT}/data/overlap_patches_labels"

PATCH_SIZE = 256

OUT_DIR = f"{PROJECT_ROOT}/experiments/mosaics"
OUT_PATH = f"{OUT_DIR}/mosaic_patches_964_966_972_1082_1117_ILM_OPL.png"


def find_image_any_ext(folder: str, stem: str):
    folder = Path(folder)
    for ext in [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None


def read_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def enhance(gray: np.ndarray) -> np.ndarray:
    # makes ILM_OPL patch more visible
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def put_text(img_bgr: np.ndarray, text: str, x=10, y=30, scale=0.9):
    cv2.putText(
        img_bgr, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 255, 255), 2, cv2.LINE_AA
    )
    return img_bgr


def gray_to_bgr(gray: np.ndarray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def mask_to_rgb_white(mask01: np.ndarray):
    m = (mask01.astype(np.uint8) * 255)
    return np.stack([m, m, m], axis=-1)


def mask_to_rgb_red(mask01: np.ndarray):
    m = (mask01.astype(np.uint8) * 255)
    rgb = np.zeros((mask01.shape[0], mask01.shape[1], 3), dtype=np.uint8)
    rgb[..., 2] = m  # red channel (BGR -> actually this is R in RGB, but OpenCV uses BGR ordering in arrays)
    # NOTE: OpenCV stores as BGR when writing, but array channel index 2 still maps to red visually.
    return rgb


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    H = W = PATCH_SIZE
    n_rows = len(PATCH_IDS)
    n_cols = 1 + len(MODELS) + 1  # ILM_OPL + 7 models + GT

    header_h = 70
    pad = 8

    mosaic_h = header_h + n_rows * (H + pad) + pad
    mosaic_w = pad + n_cols * (W + pad)

    canvas = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    # Headers
    put_text(canvas, "ILM", pad + 10, 45, 1.0)
    for j, (short, _) in enumerate(MODELS):
        x = pad + (1 + j) * (W + pad)
        put_text(canvas, short, x + 10, 45, 1.0)
    x = pad + (n_cols - 1) * (W + pad)
    put_text(canvas, "GT", x + 10, 45, 1.0)

    for i, pid in enumerate(PATCH_IDS):
        y = header_h + pad + i * (H + pad)

        # Col 0: ILM_OPL patch
        p_in = find_image_any_ext(ILM_OPL_PATCH_DIR, str(pid))
        if p_in is None:
            tile = np.zeros((H, W, 3), dtype=np.uint8)
            put_text(tile, f"P {pid}", 12, 32, 0.9)
            put_text(tile, "MISSING ILM", 12, 70, 0.8)
        else:
            gray = read_gray(p_in)
            gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
            gray = enhance(gray)
            tile = gray_to_bgr(gray)
            put_text(tile, f"P {pid}", 12, 32, 0.9)

        x0 = pad
        canvas[y:y + H, x0:x0 + W] = tile

        # Model predictions patch masks (white)
        for j, (_short, rel_dir) in enumerate(MODELS):
            col = 1 + j
            x = pad + col * (W + pad)
            pred_dir = f"{PRED_ROOT}/{rel_dir}"
            p_pred = find_image_any_ext(pred_dir, str(pid))

            if p_pred is None:
                t = np.zeros((H, W, 3), dtype=np.uint8)
                put_text(t, "MISSING", 12, 32, 0.7)
            else:
                pred = read_gray(p_pred)
                pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
                mask01 = (pred >= 128).astype(np.uint8)
                t = mask_to_rgb_white(mask01)

            canvas[y:y + H, x:x + W] = t

        # Last col: GT label patch (red)
        x = pad + (n_cols - 1) * (W + pad)
        p_gt = find_image_any_ext(LABEL_PATCH_DIR, str(pid))
        if p_gt is None:
            gt_tile = np.zeros((H, W, 3), dtype=np.uint8)
            put_text(gt_tile, "MISSING GT", 12, 32, 0.7)
        else:
            gt = read_gray(p_gt)
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            gt01 = (gt >= 128).astype(np.uint8)
            gt_tile = mask_to_rgb_red(gt01)

        canvas[y:y + H, x:x + W] = gt_tile

    cv2.imwrite(OUT_PATH, canvas)
    print(f"[OK] Saved patch mosaic to: {OUT_PATH}")


if __name__ == "__main__":
    main()

