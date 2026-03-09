import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


def robust_minmax01(x: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


class NewDomainDataset3CH(Dataset):
    """
    3 channels for new domain:
      ch0: original .tif (robust normalized to [0,1] for stability)
      ch1: CLAHE .png (uint8 -> [0,1])
      ch2: DoG   .png (uint8 -> [0,1])

    mask: *_mask.png (0/255 -> 0/1)

    Optional mean/std normalization (per channel) after augmentation:
      x = (x - mean) / std
    """

    def __init__(
        self,
        img_dir: str,
        clahe_dir: str,
        dog_dir: str,
        mask_dir: str,
        stems: List[str],
        transform=None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.img_dir = Path(img_dir)
        self.clahe_dir = Path(clahe_dir)
        self.dog_dir = Path(dog_dir)
        self.mask_dir = Path(mask_dir)
        self.stems = stems
        self.transform = transform

        if mean is not None and std is not None:
            self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.stems)

    def _load_tif01(self, stem: str) -> np.ndarray:
        p = self.img_dir / f"{stem}.tif"
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read {p}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        return robust_minmax01(img)  # [0,1]

    def _load_png01(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read {p}")
        return (img.astype(np.float32) / 255.0)

    def _load_mask01(self, stem: str) -> np.ndarray:
        p = self.mask_dir / f"{stem}_mask.png"
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Cannot read {p}")
        m = (m.astype(np.float32) / 255.0)
        return (m > 0.5).astype(np.float32)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]

        ch0 = self._load_tif01(stem)
        ch1 = self._load_png01(self.clahe_dir / f"{stem}.png")
        ch2 = self._load_png01(self.dog_dir / f"{stem}.png")
        mask = self._load_mask01(stem)

        img = np.stack([ch0, ch1, ch2], axis=-1)  # (H,W,3)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        if self.mean is not None and self.std is not None:
            img = (img - self.mean) / self.std

        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # (3,H,W)

        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0), stem


def _load_norm_stats(norm_yaml: Optional[str]):
    if norm_yaml is None:
        return None, None
    p = Path(norm_yaml)
    if not p.exists():
        return None, None
    cfg = yaml.safe_load(p.read_text())
    mean = cfg.get("mean", None)
    std = cfg.get("std", None)
    if mean is None or std is None:
        return None, None
    return mean, std


def build_loaders_newdomain_3ch(
    split_json: str,
    batch_size: int,
    num_workers: int,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    norm_yaml: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    split = json.loads(Path(split_json).read_text())

    mean, std = _load_norm_stats(norm_yaml)

    train_set = NewDomainDataset3CH(
        split["img_dir"], split["clahe_dir"], split["dog_dir"], split["mask_dir"],
        split["train"], transform=train_transform, mean=mean, std=std
    )
    val_set = NewDomainDataset3CH(
        split["img_dir"], split["clahe_dir"], split["dog_dir"], split["mask_dir"],
        split["val"], transform=val_transform, mean=mean, std=std
    )
    test_set = NewDomainDataset3CH(
        split["img_dir"], split["clahe_dir"], split["dog_dir"], split["mask_dir"],
        split["test"], transform=test_transform, mean=mean, std=std
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, split