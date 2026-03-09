import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class TargetPseudo3DirsDataset(Dataset):
    """
    Dataset target UDA (sans labels) basé sur 3 dossiers contenant les patches.

    On suppose 3 dossiers avec EXACTEMENT les mêmes noms de fichiers :
      - patches_raw/   : patch original
      - patches_clahe/ : patch CLAHE
      - patches_dog/   : patch DoG (sigma1=2.0, sigma2=3.2)

    Chaque item retourne :
      x : torch.FloatTensor (3, H, W)
      name : nom du patch (str)

    Normalisation:
      - norm_mode="per_image_zscore" : z-score calculé sur chaque patch (par canal)
      - norm_mode="none" : seulement /255 (donc [0,1])
    """

    def __init__(
        self,
        raw_dir: str,
        clahe_dir: str,
        dog_dir: str,
        filenames: Optional[List[str]] = None,
        norm_mode: str = "per_image_zscore",
        eps: float = 1e-6,
    ):
        self.raw_dir = Path(raw_dir)
        self.clahe_dir = Path(clahe_dir)
        self.dog_dir = Path(dog_dir)
        self.norm_mode = norm_mode
        self.eps = eps

        if filenames is None:
            files = [p.name for p in self.raw_dir.iterdir() if p.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"}]
            self.filenames = sorted(files, key=natural_key)
        else:
            self.filenames = sorted(filenames, key=natural_key)

        if len(self.filenames) == 0:
            raise RuntimeError("TargetPseudo3DirsDataset: no files found.")

        # optional sanity checks
        if not self.clahe_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.clahe_dir}")
        if not self.dog_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.dog_dir}")

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read: {path}")
        return img

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        x: (H, W, 3) float32 in [0,1]
        """
        if self.norm_mode == "none":
            return x

        if self.norm_mode == "per_image_zscore":
            flat = x.reshape(-1, 3)
            mu = flat.mean(axis=0)
            sd = flat.std(axis=0)
            sd = np.maximum(sd, self.eps)
            x = (x - mu.reshape(1, 1, 3)) / sd.reshape(1, 1, 3)
            return x

        raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]

        p_raw = self.raw_dir / name
        p_clahe = self.clahe_dir / name
        p_dog = self.dog_dir / name

        r = self._read_gray(p_raw)
        c = self._read_gray(p_clahe)
        d = self._read_gray(p_dog)

        # stack as (H,W,3) and convert to float [0,1]
        x = np.stack([r, c, d], axis=-1).astype(np.float32) / 255.0
        x = self._normalize(x)

        # to torch (3,H,W)
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)
        x_t = torch.from_numpy(x)

        return x_t, name