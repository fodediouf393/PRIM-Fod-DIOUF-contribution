import albumentations as A


def build_train_transforms():
    """
    Augmentations d'entraînement (paper-like) pour segmentation vasculaire.
    Compatibles avec les versions récentes d'Albumentations (sans warnings).

    Appliquées identiquement sur :
    - image (H, W, 3)
    - masque (H, W)
    """
    return A.Compose(
        [
            # flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            # transformation géométrique (remplace ShiftScaleRotate)
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                fill=0,
                fill_mask=0,
                p=0.5,
            ),

            # variations photométriques
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),

            # flou léger
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2,
            ),

            # CoarseDropout (API récente)
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                fill_mask=0,
                p=0.2,
            ),
        ]
    )


def build_eval_transforms():
    """
    Pas d'augmentation pour validation / test.
    """
    return A.Compose([])
