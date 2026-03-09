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
    root = Path(__file__).resolve().parents[1]
    split_path = root / "data" / "splits" / "split_fullimg_v1.json"
    out_path = root / "configs" / "norm_fullimg.yaml"

    with open(split_path, "r") as f:
        split = json.load(f)

    full_dir  = Path(split["full_dir"])
    ilm_dir   = Path(split["ilm_opl_dir"])
    opl_dir   = Path(split["opl_bm_dir"])

    train_files = sorted(split["train"], key=natural_key)

    sums = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    count = 0

    for name in train_files:
        a = read_gray01(ilm_dir / name)   # channel 0
        b = read_gray01(opl_dir / name)   # channel 1
        c = read_gray01(full_dir / name)  # channel 2

        for i, arr in enumerate([a, b, c]):
            sums[i] += arr.sum()
            sums2[i] += (arr * arr).sum()

        count += a.size

    means = (sums / count).tolist()
    vars_ = (sums2 / count - (sums / count) ** 2)
    stds = np.sqrt(np.maximum(vars_, 1e-12)).tolist()

    out_path.write_text(yaml.safe_dump({"mean": means, "std": stds}))
    print("[OK] Wrote", out_path)
    print("mean:", means)
    print("std :", stds)

if __name__ == "__main__":
    main()
