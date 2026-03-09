import argparse
from pathlib import Path
import cv2
import numpy as np


def morph_open(mask01: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask01
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask01, cv2.MORPH_OPEN, kernel)


def remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask01
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    out = np.zeros_like(mask01)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == lab] = 1
    return out


def morph_close(mask01: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask01
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, kernel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Input folder containing binary masks 0/255")
    ap.add_argument("--out_dir", required=True, help="Output folder for postprocessed masks")
    ap.add_argument("--save_intermediate", action="store_true",
                    help="If set, also saves A/ and AB/ intermediate folders inside out_dir")
    ap.add_argument("--open_k", type=int, default=3, help="Step A: opening kernel size (0 disables)")
    ap.add_argument("--min_area", type=int, default=100, help="Step B: remove CC smaller than min_area (0 disables)")
    ap.add_argument("--close_k", type=int, default=3, help="Step C: closing kernel size (0 disables)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_intermediate:
        out_A = out_dir / "A_open"
        out_AB = out_dir / "AB_open_rmSmall"
        out_A.mkdir(parents=True, exist_ok=True)
        out_AB.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".png"], key=lambda p: p.name)
    if not files:
        raise RuntimeError(f"No PNG found in {in_dir}")

    for p in files:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        mask01 = (m >= 128).astype(np.uint8)

        # A
        a = morph_open(mask01, args.open_k)
        if args.save_intermediate:
            cv2.imwrite(str(out_A / p.name), (a * 255).astype(np.uint8))

        # B
        ab = remove_small_components(a, args.min_area)
        if args.save_intermediate:
            cv2.imwrite(str(out_AB / p.name), (ab * 255).astype(np.uint8))

        # C
        abc = morph_close(ab, args.close_k)

        # Final output
        cv2.imwrite(str(out_dir / p.name), (abc * 255).astype(np.uint8))

    print("[OK] Post-processing done.")
    print("Input :", in_dir)
    print("Output:", out_dir)
    if args.save_intermediate:
        print("Intermediates:", out_A, "and", out_AB)


if __name__ == "__main__":
    main()