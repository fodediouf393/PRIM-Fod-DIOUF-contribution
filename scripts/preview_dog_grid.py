import argparse
from pathlib import Path

import cv2
import numpy as np


def robust_minmax01(x: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    return (x01 * 255.0).round().astype(np.uint8)


def dog_u8(x01: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    g1 = cv2.GaussianBlur(x01, ksize=(0, 0), sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(x01, ksize=(0, 0), sigmaX=sigma2, sigmaY=sigma2)
    dog = np.abs(g1 - g2)
    dog01 = robust_minmax01(dog, 1.0, 99.0)
    return to_uint8(dog01)


def load_tif(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def load_mask(mask_dir: Path, stem: str) -> np.ndarray | None:
    mp = mask_dir / f"{stem}_mask.png"
    if not mp.exists():
        return None
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    return m


def overlay_mask(gray_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    cnt = cv2.Canny(mask_u8, 50, 150)
    rgb[cnt > 0] = (0, 0, 255)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir", default="experiments/dog_grid_preview")
    ap.add_argument("--n", type=int, default=8, help="Number of images to export")
    ap.add_argument("--save_each", action="store_true", help="Also save each DoG variant as separate file")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid of DoG params
    sigma1_list = [0.7, 1.0, 1.4, 2.0]
    ratios = [1.6, 2.0, 2.5]
    pairs = [(s1, round(s1 * r, 2)) for s1 in sigma1_list for r in ratios]

    tifs = sorted(img_dir.glob("*.tif"))[: args.n]
    if not tifs:
        raise RuntimeError(f"No .tif found in {img_dir}")

    for p in tifs:
        stem = p.stem
        img = load_tif(p)

        # normalize original for display
        x01 = robust_minmax01(img)
        ch0 = to_uint8(x01)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ch1 = clahe.apply(ch0)

        # build row tiles: original, clahe, then DoGs
        tiles = [ch0, ch1]
        for (s1, s2) in pairs:
            d = dog_u8(x01, sigma1=s1, sigma2=s2)
            tiles.append(d)
            if args.save_each:
                cv2.imwrite(str(out_dir / f"{stem}_dog_s1_{s1}_s2_{s2}.png"), d)

        # create big panel (single row)
        panel = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(out_dir / f"{stem}_DOG_PANEL.png"), panel)

        # overlay sanity-check
        m = load_mask(mask_dir, stem)
        if m is not None:
            ov = overlay_mask(ch0, m)
            cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), ov)

        print(f"[OK] {stem} -> {out_dir}/{stem}_DOG_PANEL.png")

    # Save a legend text file for the panel order
    legend = ["[0] original", "[1] CLAHE"] + [f"[{i+2}] DoG(s1={s1}, s2={s2})" for i, (s1, s2) in enumerate(pairs)]
    (out_dir / "PANEL_LEGEND.txt").write_text("\n".join(legend) + "\n")

    print("\nDone.")
    print("Legend saved to:", out_dir / "PANEL_LEGEND.txt")
    print("Output folder:", out_dir)


if __name__ == "__main__":
    main()