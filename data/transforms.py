import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps


def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


def degrade_image(
    img: Image.Image,
    gamma_range=(1.5, 3.0),
    scale_range=(0.1, 0.4),
    poisson_level=255,
):
    """低光退化模型（transform 版本）。

    与 data.dataset.degrade_image 保持一致的物理意义，
    但不改变分辨率，直接在当前尺寸上施加暗化 + Poisson 噪声。
    """

    if not isinstance(img, Image.Image):
        raise TypeError(f"degrade_image 期望 PIL.Image，得到 {type(img)}")

    # 超声一般是单通道，确保为灰度；若为彩色也统一转为 L 再退化
    img = img.convert("L")

    img_np = np.array(img).astype(np.float32) / 255.0  # [H,W] in [0,1]

    # 非线性亮度下降
    gamma = random.uniform(*gamma_range)
    scale = random.uniform(*scale_range)
    img_dark = np.clip((img_np ** gamma) * scale, 0.0, 1.0)

    # Poisson 噪声
    lam = img_dark * float(poisson_level)
    noisy = np.random.poisson(lam).astype(np.float32) / float(poisson_level)
    noisy = np.clip(noisy, 0.0, 1.0)

    return Image.fromarray((noisy * 255).astype(np.uint8), mode="L")


def binarize_mask(mask, threshold=128):
    mask = mask.convert("L")
    mask = mask.point(lambda p: 255 if p > threshold else 0)
    return mask