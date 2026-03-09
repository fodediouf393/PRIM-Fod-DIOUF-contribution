# data/build_pseudo_channels.py
import argparse
from pathlib import Path

import cv2
import numpy as np


def robust_minmax01(x: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    """Robust normalization to [0,1] using percentiles."""
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    return (x01 * 255.0).round().astype(np.uint8)


def load_tif_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def make_clahe(ch0_u8: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(ch0_u8)


def make_dog(x01: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    g1 = cv2.GaussianBlur(x01, ksize=(0, 0), sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(x01, ksize=(0, 0), sigmaX=sigma2, sigmaY=sigma2)
    dog = np.abs(g1 - g2)
    dog01 = robust_minmax01(dog, 1.0, 99.0)
    return to_uint8(dog01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="Folder containing .tif images")
    ap.add_argument("--out_root", default="data", help="Project data folder (default: data)")
    ap.add_argument("--sigma1", type=float, default=2.0)
    ap.add_argument("--sigma2", type=float, default=3.2)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_tile", type=int, default=8)
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_root = Path(args.out_root)

    out_clahe = out_root / "CLAHE"
    out_dog = out_root / "DOG"
    out_clahe.mkdir(parents=True, exist_ok=True)
    out_dog.mkdir(parents=True, exist_ok=True)

    tifs = sorted(img_dir.glob("*.tif"))
    if not tifs:
        raise RuntimeError(f"No .tif found in {img_dir}")

    for p in tifs:
        stem = p.stem  # e.g., Test1r1
        img = load_tif_gray(p)

        # channel 0 normalized for deriving CLAHE/DoG (stable for float tif)
        x01 = robust_minmax01(img)
        ch0_u8 = to_uint8(x01)

        # CLAHE
        clahe_img = make_clahe(
            ch0_u8,
            clip_limit=args.clahe_clip,
            tile_grid_size=(args.clahe_tile, args.clahe_tile),
        )

        # DoG (sigma1=2.0, sigma2=3.2 by default)
        dog_img = make_dog(x01, sigma1=args.sigma1, sigma2=args.sigma2)

        # Save as PNG (uint8)
        cv2.imwrite(str(out_clahe / f"{stem}.png"), clahe_img)
        cv2.imwrite(str(out_dog / f"{stem}.png"), dog_img)

        print(f"[OK] {stem} -> CLAHE/{stem}.png , DOG/{stem}.png")

    print("\nDone.")
    print("CLAHE dir:", out_clahe)
    print("DOG dir  :", out_dog)
    print(f"DoG params: sigma1={args.sigma1}, sigma2={args.sigma2}")


if __name__ == "__main__":
    main()
