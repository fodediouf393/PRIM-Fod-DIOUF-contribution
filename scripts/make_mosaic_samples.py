# scripts/make_mosaic_samples.py
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = "/home/infres/diouf-25/prim-project"

SAMPLES = [247, 250, 272, 284, 300]

PRED_ROOT = f"{PROJECT_ROOT}/experiments/inference_outputs"

MODELS = [
    ("UNet",   "unet_3ch/seed_0/full_400_bin"),
    ("UNet++", "unetpp_3ch/seed_0/full_400_bin"),
    ("U++DS",  "unetpp_DS_3ch/seed_0/full_400_bin"),
    ("ResU",   "resunet_3ch/seed_0/full_400_bin"),
    ("R2U",    "r2unet_3ch/seed_0/full_400_bin"),
    ("U3+",    "unet3plus_3ch/seed_0/full_400_bin"),
    ("Attn",   "attention_unet_3ch/seed_0/full_400_bin"),
]

LABEL_PATCH_DIR = f"{PROJECT_ROOT}/data/overlap_patches_labels"

# ✅ FIRST COLUMN: ILM_OPL projection directory
ILM_OPL_DIR = "/home/infres/diouf-25/prim-project/data/OCTA(ILM_OPL)"

# ✅ naming rule: ILM_OPL files are 10001..10300 (and we also try 1..300 as fallback)
NAME_OFFSET = 10000

FULL_SIZE = 400
PATCH_SIZE = 256
POSITIONS = [(0, 0), (144, 0), (0, 144), (144, 144)]
PATCHES_PER_IMAGE = 4

OUT_DIR = f"{PROJECT_ROOT}/experiments/mosaics"
OUT_PATH = f"{OUT_DIR}/mosaic_samples_247_250_272_284_300_ILM_OPL.png"


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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def put_text(img_bgr: np.ndarray, text: str, x=10, y=30, scale=0.9):
    cv2.putText(
        img_bgr, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 255, 255), 2, cv2.LINE_AA
    )
    return img_bgr


def patch_id(image_id: int, slot: int):
    return (image_id - 1) * PATCHES_PER_IMAGE + (slot + 1)


def stitch_gt_label(image_id: int):
    acc = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.float32)
    cnt = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.float32)

    for slot, (x, y) in enumerate(POSITIONS):
        pid = patch_id(image_id, slot)
        p = find_image_any_ext(LABEL_PATCH_DIR, str(pid))
        if p is None:
            continue
        lab = read_gray(p).astype(np.float32) / 255.0
        acc[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += lab
        cnt[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1.0

    avg = acc / np.maximum(cnt, 1e-6)
    return (avg >= 0.5).astype(np.uint8)


def mask_to_rgb_white(mask01: np.ndarray):
    m = (mask01.astype(np.uint8) * 255)
    return np.stack([m, m, m], axis=-1)


def mask_to_rgb_green(mask01: np.ndarray):
    m = (mask01.astype(np.uint8) * 255)
    rgb = np.zeros((mask01.shape[0], mask01.shape[1], 3), dtype=np.uint8)
    rgb[..., 1] = m
    return rgb


def gray_to_bgr(gray: np.ndarray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def load_ilm_opl(image_id: int):
    # try 10001..10300 first
    stem1 = str(NAME_OFFSET + image_id)
    p = find_image_any_ext(ILM_OPL_DIR, stem1)
    if p is not None:
        return p, stem1

    # fallback 1..300
    stem2 = str(image_id)
    p = find_image_any_ext(ILM_OPL_DIR, stem2)
    if p is not None:
        return p, stem2

    return None, None


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    H = W = FULL_SIZE
    n_rows = len(SAMPLES)
    n_cols = 1 + len(MODELS) + 1  # ILM_OPL + 7 preds + GT

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

    for i, image_id in enumerate(SAMPLES):
        y = header_h + pad + i * (H + pad)

        # Column 0: ILM_OPL
        p, used_stem = load_ilm_opl(image_id)
        if p is None:
            tile = np.zeros((H, W, 3), dtype=np.uint8)
            put_text(tile, f"ID {image_id}", 12, 32, 0.9)
            put_text(tile, "MISSING ILM_OPL", 12, 70, 0.8)
        else:
            gray = read_gray(p)
            gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
            gray = enhance(gray)
            tile = gray_to_bgr(gray)
            put_text(tile, f"ID {image_id}", 12, 32, 0.9)
            put_text(tile, f"{used_stem}", 12, 70, 0.8)

        x0 = pad
        canvas[y:y + H, x0:x0 + W] = tile

        # Model predictions (white masks)
        for j, (_short, rel_dir) in enumerate(MODELS):
            col = 1 + j
            x = pad + col * (W + pad)
            pred_dir = f"{PRED_ROOT}/{rel_dir}"
            pred_path = find_image_any_ext(pred_dir, str(image_id))

            if pred_path is None:
                t = np.zeros((H, W, 3), dtype=np.uint8)
                put_text(t, "MISSING", 12, 32, 0.7)
            else:
                pred = read_gray(pred_path)
                pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
                mask01 = (pred >= 128).astype(np.uint8)
                t = mask_to_rgb_white(mask01)

            canvas[y:y + H, x:x + W] = t

        # GT label (green)
        x = pad + (n_cols - 1) * (W + pad)
        gt01 = stitch_gt_label(image_id)
        gt_tile = mask_to_rgb_green(gt01)
        canvas[y:y + H, x:x + W] = gt_tile

    cv2.imwrite(OUT_PATH, canvas)
    print(f"[OK] Saved mosaic to: {OUT_PATH}")


if __name__ == "__main__":
    main()
