# scripts/compute_norm_stats_3ch.py
import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def read_gray01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img.astype(np.float32) / 255.0

def main():
    project_root = Path(__file__).resolve().parents[1]
    split_path = project_root / "data" / "splits" / "split_v1.json"
    out_path = project_root / "configs" / "norm.yaml"

    with open(split_path, "r") as f:
        split = json.load(f)

    # 3 channel dirs
    ch1 = project_root / "data" / "overlap_patches_ilm_opl"
    ch2 = project_root / "data" / "overlap_patches_opl_bm"
    ch3 = project_root / "data" / "overlap_patches_full"

    train_files = sorted(split["train"], key=natural_key)

    # Welford style accumulation (mean/std) per channel
    sums = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    count = 0

    for name in train_files:
        a = read_gray01(ch1 / name)
        b = read_gray01(ch2 / name)
        c = read_gray01(ch3 / name)

        # flatten and accumulate
        for i, arr in enumerate([a, b, c]):
            sums[i] += arr.sum()
            sums2[i] += (arr * arr).sum()

        count += a.size

    means = (sums / count).tolist()
    vars_ = (sums2 / count - (sums / count) ** 2)
    stds = np.sqrt(np.maximum(vars_, 1e-12)).tolist()

    cfg = {"mean": means, "std": stds}
    out_path.write_text(yaml.safe_dump(cfg))

    print("[OK] Wrote", out_path)
    print("mean:", means)
    print("std :", stds)

if __name__ == "__main__":
    main()
