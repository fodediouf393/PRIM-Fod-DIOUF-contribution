import json
from pathlib import Path
import numpy as np
import cv2
import yaml


def robust_minmax01(x: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def read_tif01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return robust_minmax01(img.astype(np.float32))


def read_png01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return (img.astype(np.float32) / 255.0)


def main():
    project_root = Path(__file__).resolve().parents[1]
    split_path = project_root / "data" / "splits" / "split_newdomain_v1.json"
    out_path = project_root / "configs" / "norm_newdomain.yaml"

    split = json.loads(split_path.read_text())

    img_dir = Path(split["img_dir"])
    clahe_dir = Path(split["clahe_dir"])
    dog_dir = Path(split["dog_dir"])

    train_stems = split["train"]
    if len(train_stems) == 0:
        raise RuntimeError("Empty train split")

    sums = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    count = 0

    for stem in train_stems:
        ch0 = read_tif01(img_dir / f"{stem}.tif")
        ch1 = read_png01(clahe_dir / f"{stem}.png")
        ch2 = read_png01(dog_dir / f"{stem}.png")

        for i, arr in enumerate([ch0, ch1, ch2]):
            sums[i] += arr.sum()
            sums2[i] += (arr * arr).sum()

        count += ch0.size

    means = (sums / count)
    vars_ = (sums2 / count) - (means ** 2)
    stds = np.sqrt(np.maximum(vars_, 1e-12))

    cfg = {"mean": means.tolist(), "std": stds.tolist()}
    out_path.write_text(yaml.safe_dump(cfg))

    print("[OK] wrote", out_path)
    print("mean:", cfg["mean"])
    print("std :", cfg["std"])


if __name__ == "__main__":
    main()