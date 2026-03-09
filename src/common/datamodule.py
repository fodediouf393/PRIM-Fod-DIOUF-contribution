import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class PatchDataset3CH(Dataset):
    """
    3 channels = [ILM_OPL, OPL_BM, FULL]
    Images are loaded as grayscale, converted to float in [0,1], then optionally normalized:
        x_norm = (x - mean) / std    (per channel)

    Mask: 0/255 -> 0/1
    Albumentations transform is applied on (H,W,3) image + (H,W) mask.
    """

    def __init__(
        self,
        ch1_dir: str,
        ch2_dir: str,
        ch3_dir: str,
        label_dir: str,
        filenames: List[str],
        transform=None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.ch1_dir = Path(ch1_dir)
        self.ch2_dir = Path(ch2_dir)
        self.ch3_dir = Path(ch3_dir)
        self.label_dir = Path(label_dir)

        self.filenames = sorted(filenames, key=natural_key)
        self.transform = transform

        # mean/std are lists of length 3
        if mean is not None and std is not None:
            self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        else:
            self.mean = None
            self.std = None

    @staticmethod
    def load_gray01(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img.astype(np.float32) / 255.0  # [0,1]

    @staticmethod
    def load_mask01(path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        m = (m.astype(np.float32) / 255.0)
        m = (m > 0.5).astype(np.float32)
        return m

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]

        p1 = self.ch1_dir / name
        p2 = self.ch2_dir / name
        p3 = self.ch3_dir / name
        pm = self.label_dir / name

        img1 = self.load_gray01(p1)  # ILM_OPL
        img2 = self.load_gray01(p2)  # OPL_BM
        img3 = self.load_gray01(p3)  # FULL
        mask = self.load_mask01(pm)

        img = np.stack([img1, img2, img3], axis=-1)  # (H,W,3) float32 in [0,1]

        # Apply same augmentation to all channels and mask
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        # Normalize per channel if stats provided
        if self.mean is not None and self.std is not None:
            img = (img - self.mean) / self.std

        # To torch: (C,H,W)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img_t = torch.from_numpy(img)                 # (3,H,W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)

        return img_t, mask_t, name


def _load_norm_stats(norm_yaml: Optional[str]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if norm_yaml is None:
        return None, None
    norm_path = Path(norm_yaml)
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")
    with open(norm_path, "r") as f:
        cfg = yaml.safe_load(f)
    mean = cfg.get("mean", None)
    std = cfg.get("std", None)
    if mean is None or std is None or len(mean) != 3 or len(std) != 3:
        raise ValueError(f"Invalid norm.yaml format (need mean/std length 3): {norm_path}")
    return mean, std


def build_loaders_3ch(
    split_json: str,
    batch_size: int,
    num_workers: int,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    norm_yaml: Optional[str] = "configs/norm.yaml",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Builds train/val/test loaders for 3-channel input.
    Uses split_json which contains file lists + label_dir.
    Channel dirs are assumed at:
      data/overlap_patches_ilm_opl
      data/overlap_patches_opl_bm
      data/overlap_patches_full
    """
    split_path = Path(split_json)
    with open(split_path, "r") as f:
        split = json.load(f)

    # project root: .../data/splits/split_v1.json -> parents[2] is project root
    project_root = split_path.parents[2]

    ch1_dir = project_root / "data" / "overlap_patches_ilm_opl"
    ch2_dir = project_root / "data" / "overlap_patches_opl_bm"
    ch3_dir = project_root / "data" / "overlap_patches_full"
    label_dir = Path(split["label_dir"])

    # Load normalization stats
    mean, std = _load_norm_stats(norm_yaml)

    train_set = PatchDataset3CH(
        str(ch1_dir), str(ch2_dir), str(ch3_dir),
        str(label_dir),
        split["train"],
        transform=train_transform,
        mean=mean, std=std,
    )
    val_set = PatchDataset3CH(
        str(ch1_dir), str(ch2_dir), str(ch3_dir),
        str(label_dir),
        split["val"],
        transform=val_transform,
        mean=mean, std=std,
    )
    test_set = PatchDataset3CH(
        str(ch1_dir), str(ch2_dir), str(ch3_dir),
        str(label_dir),
        split["test"],
        transform=test_transform,
        mean=mean, std=std,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, split
