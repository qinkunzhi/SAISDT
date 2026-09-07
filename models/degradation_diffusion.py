import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_transformer_diffusion import timestep_embedding


def _luma(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[1]) == 3:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x.mean(dim=1, keepdim=True)


def _gradient_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax = a[:, :, :, 1:] - a[:, :, :, :-1]
    ay = a[:, :, 1:, :] - a[:, :, :-1, :]
    bx = b[:, :, :, 1:] - b[:, :, :, :-1]
    by = b[:, :, 1:, :] - b[:, :, :-1, :]
    return 0.5 * (F.l1_loss(ax, bx) + F.l1_loss(ay, by))


def _tv_loss(x: torch.Tensor) -> torch.Tensor:
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return dx + dy


def _color_constancy_loss(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[1]) != 3:
        return x.new_tensor(0.0)
    mean = x.mean(dim=(2, 3))
    center = mean.mean(dim=1, keepdim=True)
    return (mean - center).pow(2).mean()


def _exposure_band_loss(x: torch.Tensor, low: float = 0.45, high: float = 0.72, patch: int = 16) -> torch.Tensor:
    y = _luma(x)
    k = int(max(1, patch))
    pooled = F.avg_pool2d(y, kernel_size=k, stride=k)
    return F.relu(float(low) - pooled).mean() + F.relu(pooled - float(high)).mean()


def _highlight_loss(x: torch.Tensor, threshold: float = 0.92) -> torch.Tensor:
    y = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return F.relu(y - float(threshold)).pow(2).mean()


def _chromaticity_anchor_loss(enhanced: torch.Tensor, low_img: torch.Tensor) -> torch.Tensor:
    if int(enhanced.shape[1]) != 3 or int(low_img.shape[1]) != 3:
        return enhanced.new_tensor(0.0)
    e = enhanced / (enhanced.sum(dim=1, keepdim=True) + 1e-4)
    l = low_img / (low_img.sum(dim=1, keepdim=True) + 1e-4)
    lum = _luma(low_img.detach())
    # Very dark pixels have unreliable chromaticity, so use a soft confidence mask.
    weight = ((lum - 0.03) / 0.17).clamp(0.0, 1.0)
    return (F.smooth_l1_loss(e, l, reduction="none") * weight).sum() / (weight.sum() * 3.0 + 1e-6)


def _channel_ratio_anchor_loss(enhanced: torch.Tensor, low_img: torch.Tensor) -> torch.Tensor:
    if int(enhanced.shape[1]) != 3 or int(low_img.shape[1]) != 3:
        return enhanced.new_tensor(0.0)
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    low = torch.nan_to_num(low_img.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    valid = ((_luma(low) > 0.04) & (_luma(enh) < 0.92)).float()
    denom = valid.sum(dim=(2, 3)).clamp_min(1.0)
    enh_mean = (enh * valid).sum(dim=(2, 3)) / denom
    low_mean = (low * valid).sum(dim=(2, 3)) / denom
    enh_ratio = enh_mean / (enh_mean.sum(dim=1, keepdim=True) + 1e-4)
    low_ratio = low_mean / (low_mean.sum(dim=1, keepdim=True) + 1e-4)
    return F.smooth_l1_loss(enh_ratio, low_ratio)


class FiLMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(max(1, min(8, in_ch)), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(max(1, min(8, out_ch)), out_ch)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if int(in_ch) == int(out_ch) else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(F.silu(cond)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class DegradationUNet(nn.Module):
    def __init__(self, image_channels: int = 3, cond_dim: int = 1024, base_channels: int = 64):
        super().__init__()
        self.image_channels = int(image_channels)
        self.cond_dim = int(cond_dim)
        time_dim = int(base_channels * 4)
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(int(cond_dim), time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = int(image_channels) * 2 + 1
        self.in_conv = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)
        self.b1 = FiLMResBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.b2 = FiLMResBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.mid1 = FiLMResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid2 = FiLMResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up2 = FiLMResBlock(base_channels * 4, base_channels, time_dim)
        self.up1 = FiLMResBlock(base_channels * 2, base_channels, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(max(1, min(8, base_channels)), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, int(image_channels), kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_t: torch.Tensor, low_cond: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        temb = self.time_mlp(timestep_embedding(t.float(), self.time_dim))
        cemb = self.cond_mlp(cond.float())
        emb = temb + cemb
        x = torch.cat([x_t, low_cond, _luma(low_cond)], dim=1)
        h0 = self.in_conv(x)
        h1 = self.b1(h0, emb)
        h2 = self.b2(self.down1(h1), emb)
        hm = self.mid2(self.mid1(self.down2(h2), emb), emb)
        u2 = F.interpolate(hm, size=h2.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u2, h2], dim=1), emb)
        u1 = F.interpolate(u2, size=h1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, h1], dim=1), emb)
        return self.out(u1)


class CLIPGuidedDegradationDiffusion(nn.Module):
    """Cold/degradation diffusion for unsupervised low-light enhancement.

    Forward process: normal-light x0 -> low-light x_t through exposure/gamma/noise
    degradation. The network predicts x0 from x_t and the CLIP domain condition.
    Iterative sampling uses x_{t-1}=x_t-D_t(x0_hat)+D_{t-1}(x0_hat).
    """

    def __init__(
        self,
        image_channels: int = 3,
        cond_dim: int = 1024,
        timesteps: int = 100,
        base_channels: int = 64,
        min_exposure: float = 0.08,
        max_gamma: float = 3.2,
        noise_std: float = 0.035,
        color_shift_strength: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.cond_dim = int(cond_dim)
        self.timesteps = int(max(2, timesteps))
        self.min_exposure = float(min_exposure)
        self.max_gamma = float(max_gamma)
        self.noise_std = float(noise_std)
        self.color_shift_strength = float(max(0.0, color_shift_strength))
        self.residual_scale = float(residual_scale)
        self.denoiser = DegradationUNet(
            image_channels=int(image_channels),
            cond_dim=int(cond_dim),
            base_channels=int(base_channels),
        )

    def severity(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        return (t.float() / float(max(1, self.timesteps - 1))).clamp(0.0, 1.0)

    def degrade(self, x0: torch.Tensor, t: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
        x = torch.nan_to_num(x0, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        s = self.severity(t).view(-1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        gamma = 1.0 + (float(self.max_gamma) - 1.0) * s
        exposure = torch.exp(math.log(max(1e-4, float(self.min_exposure))) * s)
        y = x.clamp_min(1e-6).pow(gamma) * exposure

        if int(x.shape[1]) == 3 and float(self.color_shift_strength) > 0.0:
            strength = float(self.color_shift_strength)
            if stochastic:
                jitter = (torch.rand((int(x.shape[0]), 3, 1, 1), device=x.device, dtype=x.dtype) * 2.0 - 1.0)
                color = 1.0 + strength * s * jitter
            else:
                # Deterministic reverse/cycle degradation must not inject a fixed
                # warm/cold bias, otherwise real adaptation learns compensating casts.
                color = torch.ones((int(x.shape[0]), 3, 1, 1), device=x.device, dtype=x.dtype)
            y = y * color

        if stochastic and float(self.noise_std) > 0.0:
            sigma = float(self.noise_std) * s
            y = y + torch.randn_like(y) * sigma
        return y.clamp(0.0, 1.0)

    def predict_x0(
        self,
        x_t: torch.Tensor,
        low_cond: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        low_cond = torch.nan_to_num(low_cond, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        delta = torch.tanh(self.denoiser(x_t, low_cond, t, cond)) * float(self.residual_scale)
        return (x_t + delta).clamp(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor],
        cond: torch.Tensor,
        return_illum: bool = False,
    ):
        if t is None:
            t = torch.full((int(x.shape[0]),), self.timesteps - 1, device=x.device, dtype=torch.long)
        if t.dim() == 2 and t.shape[1] == 1:
            t_use = t[:, 0].long()
        else:
            t_use = t.long()
        enhanced = self.predict_x0(x, x, t_use, cond)
        if not return_illum:
            return enhanced
        gain = (_luma(enhanced) / (_luma(x).clamp_min(1e-3))).clamp(0.0, 10.0) / 10.0
        return enhanced, gain.clamp(0.0, 1.0)

    def synthetic_training_step(
        self,
        normal_img: torch.Tensor,
        cond: torch.Tensor,
        low_like: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        lambda_recon: float = 1.0,
        lambda_luma: float = 0.5,
        lambda_structure: float = 0.2,
        lambda_color: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        x0 = torch.nan_to_num(normal_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        b = int(x0.shape[0])
        if t is None:
            t = torch.randint(1, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.to(device=x0.device, dtype=torch.long)
        if low_like is None:
            x_t = self.degrade(x0, t, stochastic=True)
        else:
            x_t = torch.nan_to_num(low_like, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        x0_pred = self.predict_x0(x_t, x_t, t, cond)
        l_recon = F.l1_loss(x0_pred, x0)
        l_luma = F.l1_loss(_luma(x0_pred), _luma(x0))
        l_structure = _gradient_loss(x0_pred, x0)
        l_color = F.l1_loss(x0_pred.mean(dim=(2, 3)), x0.mean(dim=(2, 3)))
        loss = (
            float(lambda_recon) * l_recon
            + float(lambda_luma) * l_luma
            + float(lambda_structure) * l_structure
            + float(lambda_color) * l_color
        )
        return {
            "loss": loss,
            "enhanced": x0_pred,
            "low_like": x_t,
            "t": t,
            "l_recon": l_recon.detach(),
            "l_luma": l_luma.detach(),
            "l_structure": l_structure.detach(),
            "l_color": l_color.detach(),
            "l_cycle": x0.new_tensor(0.0),
            "l_exposure": x0.new_tensor(0.0),
            "l_smooth": x0.new_tensor(0.0),
            "l_highlight": x0.new_tensor(0.0),
            "l_chroma": x0.new_tensor(0.0),
        }

    def real_adapt_step(
        self,
        low_img: torch.Tensor,
        cond: torch.Tensor,
        t_value: int,
        lambda_cycle: float = 1.0,
        lambda_exposure: float = 0.5,
        lambda_smooth: float = 0.05,
        lambda_structure: float = 0.1,
        lambda_color: float = 0.05,
        lambda_highlight: float = 0.2,
        lambda_chroma: float = 0.1,
        lambda_channel_anchor: float = 0.2,
        exposure_low: float = 0.45,
        exposure_high: float = 0.72,
    ) -> Dict[str, torch.Tensor]:
        low = torch.nan_to_num(low_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        b = int(low.shape[0])
        t = torch.full((b,), int(max(1, min(int(t_value), self.timesteps - 1))), device=low.device, dtype=torch.long)
        enhanced = self.predict_x0(low, low, t, cond)
        low_re = self.degrade(enhanced, t, stochastic=False)
        l_cycle = F.l1_loss(low_re, low)
        l_exposure = _exposure_band_loss(enhanced, low=float(exposure_low), high=float(exposure_high))
        l_smooth = _tv_loss(_luma(enhanced))
        l_structure = _gradient_loss(enhanced, low)
        l_color = _color_constancy_loss(enhanced)
        l_highlight = _highlight_loss(enhanced)
        l_chroma = _chromaticity_anchor_loss(enhanced, low)
        l_channel_anchor = _channel_ratio_anchor_loss(enhanced, low)
        loss = (
            float(lambda_cycle) * l_cycle
            + float(lambda_exposure) * l_exposure
            + float(lambda_smooth) * l_smooth
            + float(lambda_structure) * l_structure
            + float(lambda_color) * l_color
            + float(lambda_highlight) * l_highlight
            + float(lambda_chroma) * l_chroma
            + float(lambda_channel_anchor) * l_channel_anchor
        )
        return {
            "loss": loss,
            "enhanced": enhanced,
            "low_like": low,
            "t": t,
            "l_recon": low.new_tensor(0.0),
            "l_luma": low.new_tensor(0.0),
            "l_structure": l_structure.detach(),
            "l_color": l_color.detach(),
            "l_cycle": l_cycle.detach(),
            "l_exposure": l_exposure.detach(),
            "l_smooth": l_smooth.detach(),
            "l_highlight": l_highlight.detach(),
            "l_chroma": l_chroma.detach(),
            "l_channel_anchor": l_channel_anchor.detach(),
        }

    @torch.no_grad()
    def sample(self, low_img: torch.Tensor, cond: torch.Tensor, steps: int = 25) -> torch.Tensor:
        low = torch.nan_to_num(low_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        x = low
        b = int(low.shape[0])
        steps = int(max(1, min(int(steps), self.timesteps - 1)))
        times = torch.linspace(self.timesteps - 1, 1, steps, device=low.device).long()
        for tt in times:
            t = torch.full((b,), int(tt.item()), device=low.device, dtype=torch.long)
            t_prev = torch.clamp(t - max(1, self.timesteps // steps), min=0)
            x0_hat = self.predict_x0(x, low, t, cond)
            d_t = self.degrade(x0_hat, t, stochastic=False)
            d_prev = self.degrade(x0_hat, t_prev, stochastic=False)
            x = (x - d_t + d_prev).clamp(0.0, 1.0)
        t0 = torch.zeros((b,), device=low.device, dtype=torch.long)
        return self.predict_x0(x, low, t0, cond).clamp(0.0, 1.0)
