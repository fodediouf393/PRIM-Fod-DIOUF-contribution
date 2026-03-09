import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

def _pad_to(img, target=448):
    h, w = img.shape[:2]
    pad_h = max(0, target - h)
    pad_w = max(0, target - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

class OCTAFullImage3Ch(Dataset):
    """
    Loads:
      - ILM_OPL grayscale
      - OPL_BM grayscale
      - FULL grayscale
    Stacks -> (H,W,3) float32 in [0,1], then normalized per channel.
    Labels loaded as 0/255 -> 0/1 float32.
    Applies padding to img_size (448) BEFORE augmentation.
    """
    def __init__(self, names, full_dir, ilm_opl_dir, opl_bm_dir, label_dir,
                 img_size=448, transform=None, mean=None, std=None):
        self.names = list(names)
        self.full_dir = Path(full_dir)
        self.ilm_opl_dir = Path(ilm_opl_dir)
        self.opl_bm_dir = Path(opl_bm_dir)
        self.label_dir = Path(label_dir)
        self.img_size = int(img_size)
        self.transform = transform

        self.mean = np.array(mean if mean is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array(std if std is not None else [1.0, 1.0, 1.0], dtype=np.float32)

    def __len__(self):
        return len(self.names)

    def _read_gray(self, p: Path):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(p)
        return im

    def __getitem__(self, idx):
        name = self.names[idx]

        ilm = _pad_to(self._read_gray(self.ilm_opl_dir / name), self.img_size)
        opl = _pad_to(self._read_gray(self.opl_bm_dir / name), self.img_size)
        ful = _pad_to(self._read_gray(self.full_dir / name), self.img_size)

        # stack H,W,3 in correct order
        x = np.stack([ilm, opl, ful], axis=-1).astype(np.float32) / 255.0

        m = _pad_to(self._read_gray(self.label_dir / name), self.img_size)
        # 0/255 -> 0/1
        y = (m.astype(np.float32) / 255.0)
        y = (y > 0.5).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=x, mask=y)
            x = aug["image"]
            y = aug["mask"]

        # normalize per channel
        x = (x - self.mean) / (self.std + 1e-8)

        # to torch: (C,H,W) and mask (1,H,W)
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y = torch.from_numpy(y[None, ...]).float()

        return x, y, name


def build_loaders_fullimg_3ch(
    data_cfg: dict,
    split_json: str,
    norm_yaml: str,
    batch_size: int,
    num_workers: int,
    train_tf,
    eval_tf,
):
    split = json.loads(Path(split_json).read_text())

    import yaml
    norm = yaml.safe_load(Path(norm_yaml).read_text())
    mean = norm["mean"]
    std = norm["std"]

    common = dict(
        full_dir=data_cfg["full_dir"],
        ilm_opl_dir=data_cfg["ilm_opl_dir"],
        opl_bm_dir=data_cfg["opl_bm_dir"],
        label_dir=data_cfg["label_dir"],
        img_size=data_cfg["img_size"],
        mean=mean,
        std=std,
    )

    ds_train = OCTAFullImage3Ch(split["train"], transform=train_tf, **common)
    ds_val   = OCTAFullImage3Ch(split["val"], transform=eval_tf, **common)
    ds_test  = OCTAFullImage3Ch(split["test"], transform=eval_tf, **common)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader
