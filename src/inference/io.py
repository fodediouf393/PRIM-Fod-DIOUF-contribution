from pathlib import Path
import cv2
import numpy as np


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def write_mask_binary(path: str, mask01: np.ndarray) -> None:
    """
    mask01: uint8 or bool, values in {0,1}
    Saved as 0/255 uint8 PNG.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = (mask01.astype(np.uint8) * 255)
    cv2.imwrite(path, out)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)
