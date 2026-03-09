import re
import csv
from pathlib import Path
import cv2
import numpy as np


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_pngs(folder: Path):
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".png"], key=lambda p: natural_key(p.name))
    if not files:
        raise RuntimeError(f"No .png files found in {folder}")
    return files


def sliding_positions(L: int, window: int, stride: int):
    """
    Standard sliding window positions with last patch forced to touch the end.
    Example: L=832, window=512, stride=256 -> [0, 256, 320]
    """
    if L < window:
        return [0]
    pos = list(range(0, L - window + 1, stride))
    if pos[-1] != L - window:
        pos.append(L - window)
    return pos


def apply_clahe(gray_u8: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_u8)


def apply_dog(gray_f32_01: np.ndarray, sigma1: float = 2.0, sigma2: float = 3.2) -> np.ndarray:
    """
    DoG on float [0,1]:
      dog = G(sigma1) - G(sigma2)
    Then per-image min-max -> uint8 [0,255]
    """
    b1 = cv2.GaussianBlur(gray_f32_01, ksize=(0, 0), sigmaX=sigma1, sigmaY=sigma1)
    b2 = cv2.GaussianBlur(gray_f32_01, ksize=(0, 0), sigmaX=sigma2, sigmaY=sigma2)
    dog = b1 - b2

    mn, mx = float(dog.min()), float(dog.max())
    if mx - mn < 1e-12:
        return np.zeros_like(gray_f32_01, dtype=np.uint8)

    dog01 = (dog - mn) / (mx - mn)
    return (255.0 * dog01).astype(np.uint8)


def main():
    # ---- INPUT ----
    src_dir = Path("/home/infres/diouf-25/prim-project/data/capillaire_langevin_832")

    # ---- OUTPUT ROOT ----
    out_root = Path("/home/infres/diouf-25/prim-project/data/capillaire_langevin_512_pseudo3dirs")

    # ---- PATCHING ----
    patch_size = 512
    stride = 256  # recommended

    # ---- CLAHE ----
    clahe_clip = 2.0
    clahe_grid = (8, 8)

    # ---- DoG ----
    dog_sigma1 = 2.0
    dog_sigma2 = 3.2

    # Output folders (one per pseudo-channel)
    out_raw = out_root / "patches_raw"
    out_clahe = out_root / "patches_clahe"
    out_dog = out_root / "patches_dog"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_clahe.mkdir(parents=True, exist_ok=True)
    out_dog.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"

    files = list_pngs(src_dir)
    print(f"[INFO] Found {len(files)} images in {src_dir}")

    rows = []
    patch_idx = 1

    for p in files:
        img_id = p.stem  # "1", "2", ...
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read {p}")

        H, W = gray.shape
        if H != 832 or W != 832:
            print(f"[WARN] {p.name} is {H}x{W}, expected 832x832 (still processed).")

        raw = gray
        cla = apply_clahe(gray, clip_limit=clahe_clip, tile_grid_size=clahe_grid)
        dog = apply_dog(gray.astype(np.float32) / 255.0, sigma1=dog_sigma1, sigma2=dog_sigma2)

        ys = sliding_positions(H, patch_size, stride)
        xs = sliding_positions(W, patch_size, stride)

        for y in ys:
            for x in xs:
                pr = raw[y:y+patch_size, x:x+patch_size]
                pc = cla[y:y+patch_size, x:x+patch_size]
                pd = dog[y:y+patch_size, x:x+patch_size]

                if pr.shape != (patch_size, patch_size):
                    continue

                # SAME NAME across the 3 folders
                patch_name = f"{img_id}_y{y}_x{x}.png"

                cv2.imwrite(str(out_raw / patch_name), pr)
                cv2.imwrite(str(out_clahe / patch_name), pc)
                cv2.imwrite(str(out_dog / patch_name), pd)

                rows.append({
                    "patch_index": patch_idx,
                    "image_id": img_id,
                    "patch_name": patch_name,
                    "y": y,
                    "x": x,
                    "H": H,
                    "W": W,
                    "patch_size": patch_size,
                    "stride": stride,
                    "clahe_clip": clahe_clip,
                    "clahe_grid": f"{clahe_grid[0]}x{clahe_grid[1]}",
                    "dog_sigma1": dog_sigma1,
                    "dog_sigma2": dog_sigma2,
                })
                patch_idx += 1

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Saved RAW patches   -> {out_raw}")
    print(f"[OK] Saved CLAHE patches -> {out_clahe}")
    print(f"[OK] Saved DoG patches   -> {out_dog}")
    print(f"[OK] Manifest written    -> {manifest_path}")
    print(f"[OK] Total patches       -> {len(rows)}")
    print("Expected per-image patches for 832 with 512/stride256: 3x3 = 9 => total ~", len(files)*9)


if __name__ == "__main__":
    main()