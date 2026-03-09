import os
import cv2
import torch
import numpy as np
import natsort
from torch.utils.data import Dataset, DataLoader


class PatchDataset3CH(Dataset):
    def __init__(self, image_folders, label_paths=None):
        # image_folders doit contenir 3 éléments (3 canaux)
        assert len(image_folders) == 3

        self.img_paths = []

        for folder in image_folders:

            # ---- Cas 1 : folder est un chemin de dossier ----
            if isinstance(folder, str):
                paths = natsort.natsorted(
                    [
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                    ]
                )

            # ---- Cas 2 : folder est une liste de chemins d’images ----
            elif isinstance(folder, list):
                paths = natsort.natsorted(folder)

            else:
                raise TypeError(
                    f"image_folders must contain folders (str) or lists of image paths, got {type(folder)}"
                )

            self.img_paths.append(paths)

        n = len(self.img_paths[0])
        assert all(len(p) == n for p in self.img_paths)

        # Labels
        self.label_paths = label_paths
        if label_paths is not None:
            assert len(label_paths) == n

    def __len__(self):
        return len(self.img_paths[0])

    def load_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return img

    def load_label(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return mask / 255.0

    def __getitem__(self, index):
        # Lecture des 3 canaux
        img1 = self.load_gray(self.img_paths[0][index])
        img2 = self.load_gray(self.img_paths[1][index])
        img3 = self.load_gray(self.img_paths[2][index])

        img = np.stack([img1, img2, img3], axis=0)
        img = torch.from_numpy(img)

        if self.label_paths is not None:
            mask = self.load_label(self.label_paths[index])
            mask = torch.from_numpy(mask).unsqueeze(0)
            return img, mask, index
        else:
            return img, index


def build_dataloaders_3ch(image_folders, label_folder, batch_size=8):
    # Récupération des labels
    lbl_paths = natsort.natsorted(
        [
            os.path.join(label_folder, f)
            for f in os.listdir(label_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )

    # Récupération chemins des images
    img_paths_3ch = []
    for folder in image_folders:
        paths = natsort.natsorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
        )
        img_paths_3ch.append(paths)

    # Découpage train/val/test
    n_train, n_val, n_test = 700, 252, 248

    label_train = lbl_paths[:n_train]
    label_val = lbl_paths[n_train:n_train + n_val]
    label_test = lbl_paths[n_train + n_val:n_train + n_val + n_test]

    img_train = [paths[:n_train] for paths in img_paths_3ch]
    img_val = [paths[n_train:n_train + n_val] for paths in img_paths_3ch]
    img_test = [paths[n_train + n_val:n_train + n_val + n_test] for paths in img_paths_3ch]

    train_set = PatchDataset3CH(img_train, label_train)
    val_set = PatchDataset3CH(img_val, label_val)
    test_set = PatchDataset3CH(img_test, label_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
