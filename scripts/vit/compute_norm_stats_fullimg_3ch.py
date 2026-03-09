import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_yaml(p: str):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def pad_to(img: np.ndarray, target: int = 448) -> np.ndarray:
    """Center-pad a 2D image to (target, target) with zeros."""
    h, w = img.shape[:2]
    pad_h = max(0, target - h)
    pad_w = max(0, target - w)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )


def main():
    # read data config (paths + img_size + split_json)
    data_cfg = load_yaml("configs/data_fullimg.yaml")
    img_size = int(data_cfg["img_size"])

    split_path = Path(data_cfg["split_json"])
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path} (run make_split_fullimg.py first)")

    split = json.loads(split_path.read_text())

    # directories for the 3 projections
    dirs = {
        "ilm_opl": Path(data_cfg["ilm_opl_dir"]),
        "opl_bm": Path(data_cfg["opl_bm_dir"]),
        "full": Path(data_cfg["full_dir"]),
    }

    # incremental sums for mean/std over TRAIN only
    sums = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    count = 0

    train_names = split["train"]
    if len(train_names) == 0:
        raise RuntimeError("Split train list is empty.")

    for name in train_names:
        chans = []
        for key in ["ilm_opl", "opl_bm", "full"]:
            p = dirs[key] / name
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise FileNotFoundError(f"Could not read image: {p}")

            im = pad_to(im, target=img_size)
            im = im.astype(np.float32) / 255.0  # -> [0,1]
            chans.append(im)

        x = np.stack(chans, axis=-1)  # (H, W, 3)

        # accumulate per-channel
        flat = x.reshape(-1, 3)       # (H*W, 3)
        sums += flat.sum(axis=0)
        sums2 += (flat ** 2).sum(axis=0)
        count += flat.shape[0]

    mean = (sums / count)
    var = (sums2 / count) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-12))

    out = Path("configs/norm_fullimg.yaml")
    out.write_text(yaml.safe_dump({"mean": mean.tolist(), "std": std.tolist()}))

    print("[OK] Wrote:", out)
    print("mean:", mean.tolist())
    print("std :", std.tolist())


if __name__ == "__main__":
    main()
