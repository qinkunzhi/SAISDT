import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision import models
from data.dataset import degrade_image
import PIL

print(f"Using F.l1_loss function: {F.l1_loss}")
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        mu1 = TF.gaussian_blur(img1, self.window_size)
        mu2 = TF.gaussian_blur(img2, self.window_size)
        sigma1_sq = TF.gaussian_blur(img1 ** 2, self.window_size) - mu1 ** 2
        sigma2_sq = TF.gaussian_blur(img2 ** 2, self.window_size) - mu2 ** 2
        sigma12 = TF.gaussian_blur(img1 * img2, self.window_size) - mu1 * mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class LGenLoss(nn.Module):
    def __init__(self):
        super(LGenLoss, self).__init__()

    def forward(self, x_high, x_high_hat):
        return F.mse_loss(x_high, x_high_hat)

class LDegradeLoss(nn.Module):
    def __init__(self):
        super(LDegradeLoss, self).__init__()
        self.to_tensor = transforms.ToTensor()

    def forward(self, x_low, x_high):
        simulated_low = degrade_image(x_high)
        if isinstance(simulated_low, torch.Tensor):
            simulated_low_t = simulated_low.to(device=x_low.device, dtype=x_low.dtype)
            if simulated_low_t.ndim == 3:
                simulated_low_t = simulated_low_t.unsqueeze(0)
        elif isinstance(simulated_low, np.ndarray):
            simulated_low_t = torch.tensor(simulated_low, dtype=torch.float32)
            if simulated_low_t.ndim == 3:
                simulated_low_t = simulated_low_t.permute(2, 0, 1).unsqueeze(0)
            elif simulated_low_t.ndim == 2:
                simulated_low_t = simulated_low_t.unsqueeze(0).unsqueeze(0)
            simulated_low_t = simulated_low_t.to(device=x_low.device, dtype=x_low.dtype)
        else:
            # PIL.Image -> [1,C,H,W]
            simulated_low_t = self.to_tensor(simulated_low).unsqueeze(0).to(device=x_low.device, dtype=x_low.dtype)

        # If degrade is computed for a single image but x_low is batched, broadcast explicitly.
        if simulated_low_t.shape[0] == 1 and x_low.shape[0] > 1:
            simulated_low_t = simulated_low_t.expand(x_low.shape[0], -1, -1, -1)

        loss = F.mse_loss(x_low, simulated_low_t)
        return loss

class LCycleLoss(nn.Module):
    def __init__(self):
        super(LCycleLoss, self).__init__()

    def forward(self, x_low, x_high_hat_low):

        if isinstance(x_high_hat_low, PIL.Image.Image):
            x_high_hat_low = transforms.ToTensor()(x_high_hat_low)
            x_high_hat_low = x_high_hat_low.unsqueeze(0).to(x_low.device)

        if x_high_hat_low.ndim == 3:
            x_high_hat_low = x_high_hat_low.unsqueeze(0)

        # Safety: avoid implicit broadcasting on batch dimension.
        if x_high_hat_low.ndim == 4 and x_low.ndim == 4:
            if x_high_hat_low.shape[0] == 1 and x_low.shape[0] > 1:
                x_high_hat_low = x_high_hat_low.expand(x_low.shape[0], -1, -1, -1)

        return F.l1_loss(x_low, x_high_hat_low)


class PerceptualLoss(nn.Module):
    """基于 VGG16 的感知损失。

    低光增强任务中，视觉观感比逐像素误差更重要，
    该损失在高层语义特征空间中约束增强结果接近 GT。
    """

    def __init__(self):
        super(PerceptualLoss, self).__init__()

        # 尽量兼容不同 torchvision 版本
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            vgg = models.vgg16(pretrained=True)

        # 使用前几层特征（感知上较稳定）
        self.features = vgg.features[:16].eval()
        for p in self.features.parameters():
            p.requires_grad = False

        # ImageNet 归一化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"PerceptualLoss 输入需要为 [B, C, H, W]，得到 {x.shape}")

        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c != 3:
            raise ValueError(f"PerceptualLoss 仅支持 1 或 3 通道图像，得到 {c} 通道")

        # 缩放到 [0, 1]
        x = torch.clamp(x, 0.0, 1.0)

        # VGG 默认 224x224
        if h != 224 or w != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # ImageNet 归一化
        x = (x - self.mean) / self.std
        return x

    def forward(self, x_high_hat: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        # 预测结果与 GT 的感知距离
        x_high_hat = self._preprocess(x_high_hat)
        x_high = self._preprocess(x_high)

        with torch.no_grad():
            feat_gt = self.features(x_high)

        feat_pred = self.features(x_high_hat)
        return F.l1_loss(feat_pred, feat_gt)


class ColorConstancyLoss(nn.Module):
    """色彩恒常性损失（Gray-World 假设）。

    鼓励每个通道的平均亮度接近，从而抑制明显偏色。
    对单通道（灰度）图像，该损失自然接近 0。
    """

    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 4:
            raise ValueError(f"ColorConstancyLoss 输入需要为 [B, C, H, W]，得到 {img.shape}")

        b, c, h, w = img.shape
        if c == 1:
            # 灰度图不需要强约束
            return img.new_tensor(0.0)

        # [B, C]
        mean_per_channel = img.mean(dim=(2, 3))
        mean_rgb = mean_per_channel.mean(dim=1, keepdim=True)
        # 各通道与整体平均值的偏差
        diff = mean_per_channel - mean_rgb
        return (diff ** 2).mean()


class GradientLoss(nn.Module):
    """梯度一致性 / 局部对比度损失。

    约束增强结果与输入在梯度空间中保持一致，
    在整体变亮的同时尽量保留边缘和局部对比度，缓解“白雾”现象。
    """

    def __init__(self):
        super(GradientLoss, self).__init__()

    @staticmethod
    def _gradients(x: torch.Tensor):
        # x: [B, C, H, W]
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, enhanced: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if enhanced.shape != reference.shape:
            raise ValueError(f"GradientLoss 需要相同形状的输入，得到 {enhanced.shape} vs {reference.shape}")

        dx_e, dy_e = self._gradients(enhanced)
        dx_r, dy_r = self._gradients(reference)

        loss_dx = F.l1_loss(dx_e, dx_r)
        loss_dy = F.l1_loss(dy_e, dy_r)
        return 0.5 * (loss_dx + loss_dy)


class ExposureLoss(nn.Module):
    """曝光补偿损失（输入自适应 + 分块约束）。

    旧版固定目标均值会把所有图像推向同一曝光水平，容易导致 gain map 同质化。
    这里改为基于输入低光图亮度的自适应目标，并在局部分块上约束，保留图像间差异。
    """

    def __init__(
        self,
        target_delta: float = 0.20,
        target_min: float = 0.45,
        target_max: float = 0.85,
        patch_size: int = 16,
    ):
        super(ExposureLoss, self).__init__()
        self.target_delta = float(target_delta)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.patch_size = int(max(1, int(patch_size)))

    def forward(self, img: torch.Tensor, input_img: torch.Tensor = None) -> torch.Tensor:
        if img.dim() != 4:
            raise ValueError(f"ExposureLoss 输入需要为 [B, C, H, W]，得到 {img.shape}")

        lum_out = img.mean(dim=1, keepdim=True)
        k = int(self.patch_size)
        out_pool = F.avg_pool2d(lum_out, kernel_size=k, stride=k)

        if input_img is not None:
            if input_img.shape != img.shape:
                raise ValueError(f"ExposureLoss 需要输入与输出同形状，得到 {img.shape} vs {input_img.shape}")
            lum_in = input_img.mean(dim=1, keepdim=True)
            in_pool = F.avg_pool2d(lum_in, kernel_size=k, stride=k)
            target = (in_pool + float(self.target_delta)).clamp(float(self.target_min), float(self.target_max))
            return F.l1_loss(out_pool, target)

        # Backward-compatible fallback: keep old behavior when input is unavailable.
        mean_lum = lum_out.mean()
        mask = (lum_out > 0.05) & (lum_out < 0.95)
        if mask.sum() > 0:
            mean_lum = lum_out[mask].mean()
        return F.l1_loss(mean_lum, img.new_tensor(0.75))


class BrightnessIncreaseLoss(nn.Module):
    """相对亮度增强损失。

    惩罚增强结果比原始低光图更暗的区域，
    鼓励模型在整体上做到 I_enhanced >= I_input（以亮度近似衡量），
    从而在不依赖配对 GT 的前提下，显式推动模型向“提亮”方向学习。
    """

    def __init__(self, min_increase: float = 0.05):
        super(BrightnessIncreaseLoss, self).__init__()
        # 期望每个像素至少比输入亮 min_increase（在 [0,1] 归一化空间下），
        # 通过损失显式推动增强结果相对输入有可感知的亮度提升。
        self.min_increase = min_increase

    def forward(self, enhanced: torch.Tensor, input_img: torch.Tensor) -> torch.Tensor:
        if enhanced.shape != input_img.shape:
            raise ValueError(f"BrightnessIncreaseLoss 需要相同形状的输入，得到 {enhanced.shape} vs {input_img.shape}")

        # 亮度近似：多通道简单平均
        lum_enh = enhanced.mean(dim=1, keepdim=True)
        lum_in = input_img.mean(dim=1, keepdim=True)

        # 只惩罚增强结果没有达到“输入 + 期望增量”的部分：max(0, L_in + margin - L_enh)
        diff = lum_in + self.min_increase - lum_enh
        darker = torch.relu(diff)
        return darker.mean()

def build_losses():
    return {
        "gen": LGenLoss(),
        "degrade": LDegradeLoss(),
        "cycle": LCycleLoss(),
        # 感知损失：在高层特征空间约束增强结果
        "perceptual": PerceptualLoss(),
        # 色彩/亮度约束：抑制偏色与过曝
        "color_const": ColorConstancyLoss(),
        "exposure": ExposureLoss(),
        # 梯度 / 局部对比度约束：在提亮过程中保持边缘与局部对比度
        "contrast": GradientLoss(),
        # 相对亮度增强约束：显式惩罚“比输入更暗”的输出，鼓励整体提亮
        "bright": BrightnessIncreaseLoss(),
    }