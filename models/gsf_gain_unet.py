import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _luma(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[1]) == 3:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x.mean(dim=1, keepdim=True)


def edge_aware_log_gain_loss(log_gain: torch.Tensor, low_img: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    gray = _luma(low_img)
    dx_a = log_gain[:, :, :, 1:] - log_gain[:, :, :, :-1]
    dy_a = log_gain[:, :, 1:, :] - log_gain[:, :, :-1, :]
    dx_i = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy_i = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    wx = torch.exp(-float(alpha) * dx_i.abs()).to(dtype=log_gain.dtype)
    wy = torch.exp(-float(alpha) * dy_i.abs()).to(dtype=log_gain.dtype)
    return (dx_a.abs() * wx).mean() + (dy_a.abs() * wy).mean()


def flat_region_log_gain_loss(log_gain: torch.Tensor, low_img: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    gray = _luma(low_img)
    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    low_freq_gain = F.avg_pool2d(F.pad(log_gain, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
    high_freq_gain = log_gain - low_freq_gain

    dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    flat_weight = torch.exp(-20.0 * (dx + dy)).to(dtype=log_gain.dtype)
    dark_weight = (1.0 - gray).clamp(0.0, 1.0).pow(1.5).to(dtype=log_gain.dtype)
    weight = flat_weight * dark_weight
    return (high_freq_gain.abs() * weight).sum() / (weight.sum() + 1e-6)


def illumination_regularization_loss(log_gain: torch.Tensor, low_img: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    tv = edge_aware_log_gain_loss(log_gain, low_img, alpha=float(alpha))
    flat = flat_region_log_gain_loss(log_gain, low_img, kernel_size=7)
    flat_coarse = flat_region_log_gain_loss(log_gain, low_img, kernel_size=15)
    return tv + 0.5 * flat + 0.25 * flat_coarse


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, out_ch)), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, out_ch)), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GSFBlock(nn.Module):
    """Global semantic fusion block used in the SCIST gain decoder."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.film = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )
        self.gate = nn.Conv2d(channels * 2, 1, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        token = self.token(cond).view(cond.shape[0], -1, 1, 1).expand_as(feat)
        gamma, beta = self.film(cond).chunk(2, dim=1)
        gamma = gamma.view(cond.shape[0], -1, 1, 1)
        beta = beta.view(cond.shape[0], -1, 1, 1)
        gate = torch.sigmoid(self.gate(torch.cat([feat, token], dim=1)))
        return (1.0 + gamma) * feat + beta + gate * token


class SimpleConditionFusionBlock(nn.Module):
    """Ablation replacement for GSF: broadcast condition + 1x1 convolution."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        with torch.no_grad():
            eye = torch.eye(channels).view(channels, channels, 1, 1)
            self.fuse.weight[:, :channels].copy_(eye)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        token = self.token(cond).view(cond.shape[0], -1, 1, 1).expand_as(feat)
        return self.fuse(torch.cat([feat, token], dim=1))


class GSFConditionedGainUNet(nn.Module):
    """CLIP + target-state conditioned U-Net.

    The main branch predicts a single-channel log-gain.  A tiny SNR-gated
    reflectance branch predicts a bounded chroma/detail residual so SCIST can
    suppress amplified low-light noise and color casts without becoming an
    unconstrained image-to-image translator.
    """

    is_scist_gain_unet = True

    def __init__(
        self,
        image_channels: int = 3,
        cond_dim: int = 384,
        base_channels: int = 64,
        gain_range: Tuple[float, float] = (1.1, 10.0),
        initial_gain: float = 1.8,
        chroma_denoise: bool = True,
        chroma_strength: float = 0.45,
        chroma_luma_threshold: float = 0.10,
        chroma_texture_threshold: float = 0.018,
        restore_enabled: bool = True,
        restore_scale: float = 0.08,
        restore_gate_bias: float = -1.2,
        color_enabled: bool = True,
        color_scale: float = 0.18,
        color_smooth_kernel: int = 15,
        fusion_type: str = "gsf",
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.cond_dim = int(cond_dim)
        self.gain_min = float(gain_range[0])
        self.gain_max = float(gain_range[1])
        self.initial_gain = float(initial_gain)
        self.chroma_denoise = bool(chroma_denoise)
        self.chroma_strength = float(chroma_strength)
        self.chroma_luma_threshold = float(chroma_luma_threshold)
        self.chroma_texture_threshold = float(chroma_texture_threshold)
        self.restore_enabled = bool(restore_enabled)
        self.restore_scale = float(max(0.0, restore_scale))
        self.color_enabled = bool(color_enabled)
        self.color_scale = float(max(0.0, color_scale))
        self.color_smooth_kernel = int(max(3, color_smooth_kernel))
        if self.color_smooth_kernel % 2 == 0:
            self.color_smooth_kernel += 1
        ch = int(base_channels)
        fusion = str(fusion_type or "gsf").lower().strip()
        fusion_block = SimpleConditionFusionBlock if fusion in {"simple", "broadcast", "concat"} else GSFBlock

        self.enc1 = ConvBlock(self.image_channels + 1, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.mid = ConvBlock(ch * 4, ch * 4)
        self.down = nn.AvgPool2d(2)

        self.up3 = nn.Conv2d(ch * 4, ch * 2, kernel_size=1)
        self.dec3 = ConvBlock(ch * 4, ch * 2)
        self.gsf3 = fusion_block(ch * 2, self.cond_dim)
        self.up2 = nn.Conv2d(ch * 2, ch, kernel_size=1)
        self.dec2 = ConvBlock(ch * 2, ch)
        self.gsf2 = fusion_block(ch, self.cond_dim)
        self.dec1 = ConvBlock(ch * 2, ch)
        self.gsf1 = fusion_block(ch, self.cond_dim)
        self.gain_head = nn.Conv2d(ch, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.constant_(self.gain_head.bias, self._initial_logit())

        self.restore_head = nn.Conv2d(ch, self.image_channels, kernel_size=3, padding=1)
        self.restore_gate = nn.Conv2d(ch + 1, 1, kernel_size=1)
        self.color_head = nn.Conv2d(ch, self.image_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.restore_head.weight)
        nn.init.zeros_(self.restore_head.bias)
        nn.init.zeros_(self.restore_gate.weight)
        nn.init.constant_(self.restore_gate.bias, float(restore_gate_bias))
        nn.init.zeros_(self.color_head.weight)
        nn.init.zeros_(self.color_head.bias)

    def _initial_logit(self) -> float:
        gmin = max(1e-6, self.gain_min)
        gmax = max(gmin + 1e-6, self.gain_max)
        target = min(max(self.initial_gain, gmin + 1e-6), gmax - 1e-6)
        log_min, log_max = math.log(gmin), math.log(gmax)
        p = (math.log(target) - log_min) / max(1e-6, log_max - log_min)
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        return math.log(p / (1.0 - p))

    def _log_gain_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        log_min = math.log(max(1e-6, self.gain_min))
        log_max = math.log(max(self.gain_min + 1e-6, self.gain_max))
        return log_min + (log_max - log_min) * torch.sigmoid(logits)

    def _chroma_reliability_mask(self, x: torch.Tensor) -> torch.Tensor:
        gray = _luma(x)
        pad = 2
        blur = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), kernel_size=5, stride=1)
        texture = F.avg_pool2d(F.pad((gray - blur).abs(), (pad, pad, pad, pad), mode="reflect"), kernel_size=5, stride=1)
        dark = ((float(self.chroma_luma_threshold) - gray) / max(float(self.chroma_luma_threshold), 1e-4)).clamp(0.0, 1.0)
        flat = ((float(self.chroma_texture_threshold) - texture) / max(float(self.chroma_texture_threshold), 1e-4)).clamp(0.0, 1.0)
        return (dark * flat).clamp(0.0, 1.0)

    def _restore_residual(self, x: torch.Tensor, d1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (not self.restore_enabled) or self.restore_scale <= 0.0:
            zero_res = torch.zeros_like(x)
            zero_gate = torch.zeros_like(x[:, :1])
            return zero_res, zero_gate, zero_gate

        snr_mask = self._chroma_reliability_mask(x).to(dtype=x.dtype)
        learned_gate = torch.sigmoid(self.restore_gate(torch.cat([d1, snr_mask], dim=1)))
        gate = (snr_mask * learned_gate).clamp(0.0, 1.0)
        residual = torch.tanh(self.restore_head(d1)) * float(self.restore_scale)

        if int(x.shape[1]) == 3:
            # Keep illumination in the gain branch; the residual may adjust only
            # chroma/detail, which makes the method easier to defend.
            residual = residual - _luma(residual).repeat(1, 3, 1, 1)
        return residual, gate, snr_mask

    def _color_log_adjustment(self, x: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:
        if (not self.color_enabled) or self.color_scale <= 0.0 or int(x.shape[1]) != 3:
            return torch.zeros_like(x)
        color_log = torch.tanh(self.color_head(d1)) * float(self.color_scale)
        color_log = color_log - color_log.mean(dim=1, keepdim=True)

        k = int(self.color_smooth_kernel)
        pad = k // 2
        color_log = F.avg_pool2d(F.pad(color_log, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
        color_log = color_log - color_log.mean(dim=1, keepdim=True)
        return color_log

    def _compose(
        self,
        x: torch.Tensor,
        log_gain: torch.Tensor,
        d1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        gain = torch.exp(log_gain)
        illum_enhanced = (x * gain).clamp(0.0, 1.0)
        color_log = torch.zeros_like(x)
        color_enhanced = illum_enhanced
        enhanced = color_enhanced
        residual = torch.zeros_like(x)
        restore_gate = torch.zeros_like(x[:, :1])
        snr_mask = torch.zeros_like(x[:, :1])
        if d1 is not None:
            color_log = self._color_log_adjustment(x, d1)
            color_enhanced = (x * torch.exp(log_gain + color_log)).clamp(0.0, 1.0)
            enhanced = color_enhanced
            residual, restore_gate, snr_mask = self._restore_residual(x, d1)
            enhanced = (color_enhanced + restore_gate * residual).clamp(0.0, 1.0)
        if self.chroma_denoise and int(x.shape[1]) == 3:
            mask = self._chroma_reliability_mask(x).to(dtype=enhanced.dtype)
            strength = float(max(0.0, min(1.0, self.chroma_strength)))
            enh_luma = _luma(enhanced)
            gray_enh = enh_luma.repeat(1, 3, 1, 1)
            enhanced = enhanced * (1.0 - strength * mask) + gray_enh * (strength * mask)
            enhanced = enhanced.clamp(0.0, 1.0)
        extras = {
            "gain_enhanced": color_enhanced,
            "illum_enhanced": illum_enhanced,
            "color_log": color_log,
            "restore_residual": residual,
            "restore_gate": restore_gate,
            "snr_mask": snr_mask,
        }
        return enhanced, gain, extras

    def _forward_features(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        bright = (1.0 - _luma(x)).to(dtype=x.dtype)
        e1 = self.enc1(torch.cat([x, bright], dim=1))
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        m = self.mid(e3)

        d3 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(d3)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d3 = self.gsf3(d3, cond)

        d2 = F.interpolate(d3, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(d2)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d2 = self.gsf2(d2, cond)

        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        d1 = self.gsf1(d1, cond)
        return self._log_gain_from_logits(self.gain_head(d1)), d1

    def _forward_log_gain(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        log_gain, _d1 = self._forward_features(x, cond)
        return log_gain

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t=None, return_gain: bool = False):
        _ = t
        log_gain, d1 = self._forward_features(x, cond)
        enhanced, gain, _extras = self._compose(x, log_gain, d1=d1)
        if bool(return_gain):
            log_min = math.log(max(1e-6, self.gain_min))
            log_max = math.log(max(self.gain_min + 1e-6, self.gain_max))
            gain_norm = ((log_gain - log_min) / max(1e-6, log_max - log_min)).clamp(0.0, 1.0)
            return enhanced, gain_norm
        return enhanced

    def training_step(
        self,
        low_img: torch.Tensor,
        cond: torch.Tensor,
        lambda_smooth: float = 0.1,
        illum_edge_alpha: float = 10.0,
        **_kwargs,
    ) -> Dict[str, torch.Tensor]:
        log_gain, d1 = self._forward_features(low_img, cond)
        enhanced, gain, extras = self._compose(low_img, log_gain, d1=d1)
        l_illu = illumination_regularization_loss(log_gain, low_img, alpha=float(illum_edge_alpha))
        zero = low_img.new_tensor(0.0)
        return {
            "loss": float(lambda_smooth) * l_illu,
            "l_diff": zero,
            "l_ill": l_illu.detach(),
            "l_inv": zero,
            "l_recon": zero,
            "l_target_illum": zero,
            "l_teacher": zero,
            "l_refiner_aux": l_illu.detach(),
            "l_exp": zero,
            "l_noise": zero,
            "l_artifact": zero,
            "enhanced": enhanced,
            "gain": gain,
            "log_gain": log_gain,
            "gain_enhanced": extras["gain_enhanced"],
            "illum_enhanced": extras["illum_enhanced"],
            "color_log": extras["color_log"],
            "restore_residual": extras["restore_residual"],
            "restore_gate": extras["restore_gate"],
            "snr_mask": extras["snr_mask"],
            "correction": zero,
            "gain_norm": gain.detach(),
            "coarse_enhanced": extras["gain_enhanced"].detach(),
            "coarse_gain": gain.detach(),
            "coarse_log_gain": log_gain.detach(),
        }
