import albumentations as A
import cv2


def build_train_transforms_newdomain(pad_to: int = 832):
    """
    - PadIfNeeded to (pad_to x pad_to)
      image: reflect padding (border_mode=cv2.BORDER_REFLECT_101)
      mask: 0 padding
    - light augmentations (safe)
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=pad_to,
                min_width=pad_to,
                border_mode=cv2.BORDER_REFLECT_101,
                value=0,
                mask_value=0,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.03, 0.03),
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                fill=0,
                fill_mask=0,
                p=0.3,
            ),
        ]
    )


def build_eval_transforms_newdomain(pad_to: int = 832):
    """Only padding to make divisible by 16."""
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=pad_to,
                min_width=pad_to,
                border_mode=cv2.BORDER_REFLECT_101,
                value=0,
                mask_value=0,
                p=1.0,
            ),
        ]
    )