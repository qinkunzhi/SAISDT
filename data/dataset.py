import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import numpy as np

to_tensor = transforms.ToTensor()
from data.transforms import get_transforms
from data.utils import load_image


def degrade_image(
    img,
    target_size=(256, 256),
    gamma_range=(1.5, 3.0),
    scale_range=(0.1, 0.4),
    poisson_level=255,
):
    """低光退化模型：非线性亮度下降 + 类 Poisson 噪声。

    - 先通过 gamma > 1 和缩放因子降低整体亮度，模拟曝光不足；
    - 再在暗光条件下添加近似 Poisson 的光子噪声。

    Args:
        img: 高质量输入，可以是 Tensor([C,H,W] 或 [B,C,H,W]) 或 PIL.Image。
        target_size: 输出分辨率。
        gamma_range: 亮度压缩的 gamma 范围 (>1 越暗)。
        scale_range: 整体强度缩放范围 (0~1)。
        poisson_level: Poisson 采样强度尺度（类似“曝光等级”）。
    """

    def _degrade_pil(pil_img: Image.Image) -> Image.Image:
        # 调整到目标分辨率
        pil_img = pil_img.resize(target_size, Image.LANCZOS)

        # 转为 [0,1] 浮点
        img_np = np.array(pil_img).astype(np.float32) / 255.0

        # 单通道 / 多通道兼容
        if img_np.ndim == 2:
            img_np = img_np[..., None]

        # 非线性亮度下降（gamma > 1 + 整体缩放）
        gamma = np.random.uniform(*gamma_range)
        scale = np.random.uniform(*scale_range)
        img_dark = np.clip((img_np ** gamma) * scale, 0.0, 1.0)

        # 近似 Poisson 噪声：按光子计数采样后再归一化
        lam = img_dark * float(poisson_level)
        noisy = np.random.poisson(lam).astype(np.float32) / float(poisson_level)
        noisy = np.clip(noisy, 0.0, 1.0)

        # 还原形状：如果原来是单通道，就挤掉通道维
        if noisy.shape[2] == 1:
            noisy = noisy[..., 0]

        return Image.fromarray((noisy * 255).astype(np.uint8))

    # 统一处理：
    # - PIL 输入：返回 PIL
    # - Tensor 输入：保持 batch 维度，返回 Tensor（与训练 loss 更兼容）
    if isinstance(img, torch.Tensor):
        if img.ndim not in (3, 4):
            raise ValueError(f"Unsupported tensor shape: {img.shape}, expected [C,H,W] or [B,C,H,W]")

        device = img.device
        dtype = img.dtype
        x = img.detach().clamp(0.0, 1.0).to("cpu")

        if x.ndim == 3:
            pil = transforms.ToPILImage()(x)
            out_pil = _degrade_pil(pil)
            out = transforms.ToTensor()(out_pil).to(device=device, dtype=dtype)
            return out

        outs = []
        for i in range(x.shape[0]):
            pil = transforms.ToPILImage()(x[i])
            out_pil = _degrade_pil(pil)
            out = transforms.ToTensor()(out_pil)
            outs.append(out)
        out_b = torch.stack(outs, dim=0).to(device=device, dtype=dtype)
        return out_b

    if not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported image type: {type(img)}")

    return _degrade_pil(img)

class UltrasoundDataset(Dataset):

    def __init__(self, root_dir, dataset_type="unpaired", mode="train"):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.mode = mode
        self.data = []

        dataset_path = os.path.join(root_dir, dataset_type)

        if mode == "train":
            high_quality_dir = os.path.join(dataset_path, "high quality")
            low_quality_dir = os.path.join(dataset_path, "low quality")

            # 过滤掉 `.DS_Store` 和其他非图片文件
            high_quality_files = sorted(
                [f for f in os.listdir(high_quality_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            )

            if dataset_type == "unpaired":
                low_quality_files = sorted(
                    [
                        f
                        for f in os.listdir(low_quality_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                )
                # 仅保存 HQ / LQ 路径对，不再使用 mask
                self.data = [
                    (os.path.join(high_quality_dir, hq), os.path.join(low_quality_dir, lq))
                    for hq, lq in zip(high_quality_files, low_quality_files)
                ]
            else:
                if not os.path.exists(low_quality_dir):
                    os.makedirs(low_quality_dir)

                low_quality_files = sorted(
                    [
                        f
                        for f in os.listdir(low_quality_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                )

                if len(low_quality_files) == 0:
                    print(
                        f"⚠️  Low-quality directory `{low_quality_dir}` is empty, generating images..."
                    )
                    self._generate_low_quality_images(high_quality_dir, low_quality_dir)
                else:
                    print(
                        f"✅ Low-quality directory `{low_quality_dir}` already has {len(low_quality_files)} files."
                    )

                # 仍按文件名一一对应 HQ / LQ，但不再读取 mask
                self.data = [
                    (os.path.join(high_quality_dir, hq), os.path.join(low_quality_dir, hq))
                    for hq in high_quality_files
                ]

        else:

            low_quality_dir = os.path.join(dataset_path, "LR")
            high_quality_dir = os.path.join(dataset_path, "HR")

            low_quality_files = sorted(
                [f for f in os.listdir(low_quality_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            )

            # 测试阶段同样只保留 (LQ, HQ) 对，不再使用 mask
            self.data = [
                (os.path.join(low_quality_dir, lq), os.path.join(high_quality_dir, lq))
                for lq in low_quality_files
            ]

    def _generate_low_quality_images(self, high_quality_dir, low_quality_dir):
        """
        生成低质量图像，并存入 low_quality 目录
        """
        print("📢 Generating low-quality images for `with_mask` dataset...")

        for img_name in os.listdir(high_quality_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(high_quality_dir, img_name)
            try:
                img = Image.open(img_path).convert("L")
                degraded_img = degrade_image(img)
                save_path = os.path.join(low_quality_dir, img_name)
                degraded_img.save(save_path)
                print(f"✅ Saved degraded image: {save_path}")
            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")

        print("✅ Finished generating low-quality images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            # 训练阶段：统一结构为 (hq_path, lq_path)
            hq_path, lq_path = self.data[idx]

            hq_img = load_image(hq_path)
            if hq_img is None:
                print(f"⚠️ Skipping sample `{hq_path}` due to loading error.")
                return self.__getitem__((idx + 1) % len(self.data))  # 递归获取下一个样本

            hq_img = to_tensor(hq_img)

            if os.path.exists(lq_path):
                lq_img = load_image(lq_path)
                if lq_img is None:
                    print(
                        f"⚠️ Warning: Failed to load `{lq_path}`, generating degraded image."
                    )
                    lq_img = degrade_image(hq_img)
            else:
                print(f"⚠️ `{lq_path}` not found, generating degraded version.")
                lq_img = degrade_image(hq_img)

            lq_img = to_tensor(lq_img)

            resize_transform = transforms.Resize((256, 256))
            lq_img = resize_transform(lq_img)
            hq_img = resize_transform(hq_img)

            # 统一返回 (lq_img, hq_img)
            return lq_img, hq_img

        else:
            # 测试/验证阶段：统一返回 (lq_img, hq_img)
            lq_path, hq_path = self.data[idx]
            lq_img = to_tensor(load_image(lq_path))
            hq_img = to_tensor(load_image(hq_path))
            return lq_img, hq_img

def _build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    return torch.utils.data.DataLoader(dataset, **kwargs)


def get_dataloader(
    root_dir,
    dataset_type="unpaired",
    mode="train",
    batch_size=8,
    shuffle=True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    dataset = UltrasoundDataset(root_dir, dataset_type, mode)
    return _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


class LowLightUnpairedDataset(Dataset):
    """通用低光增强数据集：给定低光根目录和高质量根目录，unpaired 采样。

    典型用法：
        - low_root  = /home/.../LOLv1/Train/input
        - high_root = /home/.../DIV2K_384
    """

    def __init__(self, low_root: str, high_root: str):
        self.low_root = low_root
        self.high_root = high_root

        def _list_images(root: str):
            return sorted(
                [
                    os.path.join(root, f)
                    for f in os.listdir(root)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )

        self.low_files = _list_images(low_root)
        self.high_files = _list_images(high_root)

        if len(self.low_files) == 0:
            raise RuntimeError(f"No low-light images found in {low_root}")
        if len(self.high_files) == 0:
            raise RuntimeError(f"No high-quality images found in {high_root}")

        self.resize = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        lq_path = self.low_files[idx]
        # 高质量域用 unpaired 采样：随机选择一张 DIV2K 图像
        hq_path = random.choice(self.high_files)

        lq_img = load_image(lq_path)
        hq_img = load_image(hq_path)

        lq_img = to_tensor(lq_img)
        hq_img = to_tensor(hq_img)

        lq_img = self.resize(lq_img)
        hq_img = self.resize(hq_img)

        return lq_img, hq_img


class LowLightOnlyDataset(Dataset):
    """Low-light-only dataset for unsupervised enhancement training.

    Normal-light images are still used elsewhere to compute the CLIP domain prior,
    but the training batch itself only needs the low-light image.
    """

    def __init__(self, low_root: str):
        self.low_root = low_root

        self.low_files = sorted(
            [
                os.path.join(low_root, f)
                for f in os.listdir(low_root)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if len(self.low_files) == 0:
            raise RuntimeError(f"No low-light images found in {low_root}")

        self.resize = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        lq_path = self.low_files[idx]
        lq_img = load_image(lq_path)
        lq_img = to_tensor(lq_img)
        lq_img = self.resize(lq_img)
        # Keep the project-wide training interface `(low, high_like)` stable.
        # The second item is not used by the unsupervised illumination diffusion path.
        return lq_img, lq_img


def degrade_low_light_tensor(
    img: torch.Tensor,
    gamma_range=(1.5, 4.0),
    scale_range=(0.05, 0.55),
    gaussian_std_range=(0.0, 0.035),
    color_shift_range=(0.85, 1.15),
    shadow_prob: float = 0.7,
) -> torch.Tensor:
    """Domain-randomized low-light degradation used for self-supervised pretraining.

    It follows the common synthetic LLIE setting: gamma/exposure darkening plus
    sensor-like noise and mild color/local illumination shifts. The goal is not
    to exactly model every real camera, but to expose the correction diffusion to
    a broad normal-light restoration distribution before real unsupervised
    adaptation.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(img.shape)}")
    x = img.detach().float().clamp(0.0, 1.0)
    c, h, w = x.shape

    gamma = random.uniform(float(gamma_range[0]), float(gamma_range[1]))
    scale = random.uniform(float(scale_range[0]), float(scale_range[1]))
    y = (x.pow(gamma) * scale).clamp(0.0, 1.0)

    if c == 3:
        gains = torch.empty(3, 1, 1).uniform_(float(color_shift_range[0]), float(color_shift_range[1]))
        y = (y * gains).clamp(0.0, 1.0)

    if random.random() < float(shadow_prob):
        yy = torch.linspace(-1.0, 1.0, h).view(1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w).view(1, 1, w)
        cx = random.uniform(-0.8, 0.8)
        cy = random.uniform(-0.8, 0.8)
        sx = random.uniform(0.35, 0.9)
        sy = random.uniform(0.35, 0.9)
        shadow = torch.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
        strength = random.uniform(0.15, 0.55)
        y = y * (1.0 - strength * shadow)

    # Poisson-like shot noise. Clamp the rate so very dark pixels stay finite.
    peak = random.uniform(30.0, 255.0)
    y = torch.poisson((y * peak).clamp_min(0.0)) / peak

    sigma = random.uniform(float(gaussian_std_range[0]), float(gaussian_std_range[1]))
    if sigma > 0:
        y = y + torch.randn_like(y) * sigma

    if random.random() < 0.25:
        y = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))(y)

    return y.clamp(0.0, 1.0)


class SyntheticLowLightDataset(Dataset):
    """Normal-light images with on-the-fly low-light degradation.

    Returns `(synthetic_low, normal)` so the diffusion target can be the
    correction from teacher coarse illumination to the known normal-light target.
    This is self-supervised pretraining: no real low/normal pairs are required.
    """

    def __init__(self, high_root: str, size=(256, 256)):
        self.high_root = high_root
        self.high_files = []
        for dirpath, _dirnames, filenames in os.walk(high_root):
            for f in filenames:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    self.high_files.append(os.path.join(dirpath, f))
        self.high_files = sorted(self.high_files)
        if len(self.high_files) == 0:
            raise RuntimeError(f"No normal-light images found in {high_root}")
        self.resize = transforms.Resize(tuple(size))

    def __len__(self):
        return len(self.high_files)

    def __getitem__(self, idx):
        hq_path = self.high_files[idx]
        hq_img = load_image(hq_path)
        hq = self.resize(to_tensor(hq_img))
        low = degrade_low_light_tensor(hq)
        return low, hq


def get_lowlight_dataloader(
    low_root: str,
    high_root: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    dataset = LowLightUnpairedDataset(low_root=low_root, high_root=high_root)
    return _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


class LowLightEvalDataset(Dataset):
    """Paired low-light evaluation dataset for PSNR/SSIM/NIQE/PI logging."""
    """低光评估数据集：成对 (input, target) 用于度量 PSNR / NIQE 等。

    例如：
        low_root = /.../LOLv1/Train/input
        gt_root  = /.../LOLv1/Train/target
    """

    def __init__(self, low_root: str, gt_root: str, resize_size=(256, 256)):
        self.low_root = low_root
        self.gt_root = gt_root

        files = [
            f
            for f in os.listdir(low_root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        files = sorted(files)

        self.pairs = []
        for f in files:
            low_path = os.path.join(low_root, f)
            gt_path = os.path.join(gt_root, f)
            if os.path.exists(gt_path):
                self.pairs.append((low_path, gt_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired low/gt images found between {low_root} and {gt_root}"
            )

        self.resize = transforms.Resize(tuple(resize_size)) if resize_size is not None else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, gt_path = self.pairs[idx]

        lq_img = load_image(low_path)
        gt_img = load_image(gt_path)

        lq = to_tensor(lq_img)
        gt = to_tensor(gt_img)

        if self.resize is not None:
            lq = self.resize(lq)
            gt = self.resize(gt)

        return lq, gt, os.path.basename(low_path)


def get_lowlight_eval_dataloader(
    low_root: str,
    gt_root: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    resize_size=(256, 256),
):
    dataset = LowLightEvalDataset(low_root=low_root, gt_root=gt_root, resize_size=resize_size)
    return _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
