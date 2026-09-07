import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_transformer_diffusion import (
    Downsample,
    LinearBetaSchedule,
    ResBlock,
    SobelStructure,
    TimeMLP,
    Upsample,
    _extract_1d,
    timestep_embedding,
)


def _safe_luma(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    if int(x.shape[1]) == 3:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x.mean(dim=1, keepdim=True)


def _edge_aware_tv(x: torch.Tensor, guide: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    if x.ndim != 4 or guide.ndim != 4:
        raise ValueError(f"Expected 4D tensors, got x={tuple(x.shape)} guide={tuple(guide.shape)}")
    g = _safe_luma(guide)
    dx_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx_g = g[:, :, :, 1:] - g[:, :, :, :-1]
    dy_g = g[:, :, 1:, :] - g[:, :, :-1, :]
    wx = torch.exp(-float(alpha) * dx_g.abs()).to(dtype=x.dtype)
    wy = torch.exp(-float(alpha) * dy_g.abs()).to(dtype=x.dtype)
    return (dx_x.abs() * wx).mean() + (dy_x.abs() * wy).mean()


def _exposure_loss(img: torch.Tensor, target: float = 0.55, patch_size: int = 16) -> torch.Tensor:
    lum = _safe_luma(img)
    k = int(max(1, patch_size))
    pooled = F.avg_pool2d(lum, kernel_size=k, stride=k)
    return (pooled - float(target)).pow(2).mean()


def _exposure_band_loss(img: torch.Tensor, low: float = 0.42, high: float = 0.72, patch_size: int = 16) -> torch.Tensor:
    lum = _safe_luma(img)
    k = int(max(1, patch_size))
    pooled = F.avg_pool2d(lum, kernel_size=k, stride=k)
    low = float(low)
    high = float(max(low + 1e-3, high))
    under = F.relu(low - pooled).pow(2)
    over = F.relu(pooled - high).pow(2)
    return (under + 2.0 * over).mean()


def _color_constancy_loss(img: torch.Tensor) -> torch.Tensor:
    if int(img.shape[1]) < 3:
        return img.new_tensor(0.0)
    mean_rgb = img.mean(dim=(2, 3))
    mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
    return ((mr - mg).pow(2) + (mr - mb).pow(2) + (mg - mb).pow(2)).mean()


def _spatial_consistency_loss(enhanced: torch.Tensor, low_img: torch.Tensor) -> torch.Tensor:
    enh = F.avg_pool2d(_safe_luma(enhanced), kernel_size=4, stride=4)
    low = F.avg_pool2d(_safe_luma(low_img), kernel_size=4, stride=4)

    def diffs(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left = x[:, :, :, 1:] - x[:, :, :, :-1]
        right = -left
        down = x[:, :, 1:, :] - x[:, :, :-1, :]
        up = -down
        return left, right, up, down

    e_l, e_r, e_u, e_d = diffs(enh)
    l_l, l_r, l_u, l_d = diffs(low)
    return (
        F.mse_loss(e_l, l_l)
        + F.mse_loss(e_r, l_r)
        + F.mse_loss(e_u, l_u)
        + F.mse_loss(e_d, l_d)
    )


def _saturation_loss(img: torch.Tensor, high: float = 0.98, low: float = 0.0) -> torch.Tensor:
    over = F.relu(img - float(high)).pow(2).mean()
    under = F.relu(float(low) - img).pow(2).mean()
    return over + under


def _dark_region_noise_loss(enhanced: torch.Tensor, low_img: torch.Tensor) -> torch.Tensor:
    lum = _safe_luma(low_img)
    dark = (1.0 - lum).clamp(0.0, 1.0)
    hp = enhanced - F.avg_pool2d(enhanced, kernel_size=3, stride=1, padding=1)
    return (dark * hp.abs()).mean()


def _blur_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)


def _flat_region_artifact_loss(
    enhanced: torch.Tensor,
    low_img: torch.Tensor,
    reference_img: Optional[torch.Tensor] = None,
    kernel_size: int = 5,
    margin: float = 0.01,
    edge_alpha: float = 20.0,
    dark_power: float = 1.5,
) -> torch.Tensor:
    """Suppress newly-created high frequency in dark/flat regions.

    This is an unsupervised artifact guard: real image edges are protected by the
    edge-aware mask, while high-pass content that exceeds the coarse/reference
    enhancement in flat dark areas is penalized.
    """
    enhanced = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    low_img = torch.nan_to_num(low_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ref = low_img if reference_img is None else reference_img.detach()
    ref = torch.nan_to_num(ref, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if int(ref.shape[1]) != int(enhanced.shape[1]):
        if int(ref.shape[1]) == 1 and int(enhanced.shape[1]) == 3:
            ref = ref.repeat(1, 3, 1, 1)
        elif int(ref.shape[1]) == 3 and int(enhanced.shape[1]) == 1:
            ref = _safe_luma(ref)

    hp_enh = enhanced - _blur_same(enhanced, int(kernel_size))
    hp_ref = ref - _blur_same(ref, int(kernel_size))

    guide = _safe_luma(low_img)
    dx = F.pad((guide[:, :, :, 1:] - guide[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((guide[:, :, 1:, :] - guide[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    flat_weight = torch.exp(-float(edge_alpha) * (dx + dy))
    dark_weight = (1.0 - guide).clamp(0.0, 1.0).pow(float(max(0.1, dark_power)))
    weight = flat_weight * dark_weight

    excess = F.relu(hp_enh.abs() - hp_ref.abs() - float(max(0.0, margin)))
    return (excess * weight).sum() / (weight.sum() * float(enhanced.shape[1]) + 1e-6)


class ZeroReferenceIlluminationTeacher(nn.Module):
    """Zero-DCE/SCI-style deterministic coarse illumination estimator.

    The teacher intentionally uses only low-level image evidence, not CLIP semantics.
    CLIP guidance is injected in the diffusion refiner and CLIP-domain loss.
    """

    def __init__(
        self,
        image_channels: int,
        base_channels: int = 32,
        gain_range: Tuple[float, float] = (1.1, 4.0),
        initial_gain: float = 1.4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.gain_min, self.gain_max = float(gain_range[0]), float(gain_range[1])
        self.initial_gain = float(initial_gain)
        self.temperature = float(max(1e-6, temperature))

        in_ch = int(image_channels) + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, self._initial_logit())

    def _initial_logit(self) -> float:
        gmin = max(1e-6, float(self.gain_min))
        gmax = max(gmin + 1e-6, float(self.gain_max))
        target = min(max(float(self.initial_gain), gmin + 1e-6), gmax - 1e-6)
        log_min, log_max = math.log(gmin), math.log(gmax)
        p = (math.log(target) - log_min) / max(1e-6, (log_max - log_min))
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        return math.log(p / (1.0 - p))

    def _log_gain_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        gmin = max(1e-6, float(self.gain_min))
        gmax = max(gmin + 1e-6, float(self.gain_max))
        log_min, log_max = math.log(gmin), math.log(gmax)
        return log_min + (log_max - log_min) * torch.sigmoid(logits / self.temperature)

    def forward(self, low_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        brightness = (1.0 - _safe_luma(low_img)).to(dtype=low_img.dtype)
        logits = self.net(torch.cat([low_img, brightness], dim=1))
        log_gain = self._log_gain_from_logits(logits)
        gain = torch.exp(log_gain)
        return log_gain, gain


class IlluminationDiffusionRefiner(nn.Module):
    """Conditional U-Net for the normal-light illumination correction field."""

    def __init__(
        self,
        image_channels: int,
        cond_dim: int,
        base_channels: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        _ = num_heads
        self.image_channels = int(image_channels)
        self.cond_dim = int(cond_dim)
        self.struct = SobelStructure()

        time_dim = int(base_channels * 4)
        self.time_dim = time_dim
        self.time_mlp = TimeMLP(time_dim, time_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = 1 + int(image_channels) + 1 + 1
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = Downsample(base_channels)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.downsample2 = Downsample(base_channels * 2)
        self.mid1 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.upsample2 = Upsample(base_channels * 2)
        self.up2 = ResBlock(base_channels * 4, base_channels, time_dim)
        self.upsample1 = Upsample(base_channels)
        self.up1 = ResBlock(base_channels * 2, base_channels, time_dim)
        self.out_norm = nn.GroupNorm(num_groups=max(1, min(8, base_channels)), num_channels=base_channels)
        self.eps_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.eps_head.weight)
        nn.init.zeros_(self.eps_head.bias)

    def forward(
        self,
        noisy_log_gain: torch.Tensor,
        low_img: torch.Tensor,
        coarse_log_gain: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        temb = temb + self.cond_proj(cond.float()).to(device=temb.device, dtype=temb.dtype)

        edge = self.struct(low_img).to(dtype=low_img.dtype)
        x = torch.cat([noisy_log_gain, low_img, coarse_log_gain, edge], dim=1)
        h0 = self.in_conv(x)
        h1 = self.down1(h0, temb)
        h2 = self.down2(self.downsample1(h1), temb)
        hm = self.mid2(self.mid1(self.downsample2(h2), temb), temb)
        hu2 = self.upsample2(hm)
        if hu2.shape[-2:] != h2.shape[-2:]:
            hu2 = F.interpolate(hu2, size=h2.shape[-2:], mode="nearest")
        hu2 = self.up2(torch.cat([hu2, h2], dim=1), temb)
        hu1 = self.upsample1(hu2)
        if hu1.shape[-2:] != h1.shape[-2:]:
            hu1 = F.interpolate(hu1, size=h1.shape[-2:], mode="nearest")
        hu1 = self.up1(torch.cat([hu1, h1], dim=1), temb)
        return self.eps_head(F.silu(self.out_norm(hu1)))


class TeacherConditionedIlluminationDiffusion(nn.Module):
    """Teacher-conditioned CLIP normal-light correction diffusion.

    Teacher: zero-reference low-level coarse illumination estimator A_c.
    Diffusion: learns the residual correction Delta_A from A_c to the normal-light
    illumination A_ref on synthetic self-supervised data, then adapts on real
    low-light data with CLIP-domain and zero-reference constraints.
    Output: I_enh = I_low * exp(A_c + Delta_A).
    """

    is_diffusion = True
    uses_timestep = True
    is_teacher_illum_diffusion = True

    def __init__(
        self,
        image_channels: int,
        cond_dim: int,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        base_channels: int = 64,
        num_heads: int = 4,
        gain_range: Tuple[float, float] = (1.1, 4.0),
        initial_gain: float = 1.4,
        infer_timestep: int = 10,
        exposure_target: float = 0.62,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.cond_dim = int(cond_dim)
        self.gain_min, self.gain_max = float(gain_range[0]), float(gain_range[1])
        self.exposure_target = float(exposure_target)
        self.highlight_compression = False
        self.highlight_knee = 0.88
        self.highlight_ceiling = 0.98
        self.default_infer_timestep = int(max(0, int(infer_timestep)))
        self.schedule = LinearBetaSchedule(
            timesteps=int(timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
        )
        self.teacher = ZeroReferenceIlluminationTeacher(
            image_channels=int(image_channels),
            base_channels=max(16, int(base_channels) // 2),
            gain_range=gain_range,
            initial_gain=float(initial_gain),
        )
        self.refiner = IlluminationDiffusionRefiner(
            image_channels=int(image_channels),
            cond_dim=int(cond_dim),
            base_channels=int(base_channels),
            num_heads=int(num_heads),
        )

    @property
    def timesteps(self) -> int:
        return int(self.schedule.timesteps)

    def _log_bounds(self) -> Tuple[float, float]:
        gmin = max(1e-6, float(self.gain_min))
        gmax = max(gmin + 1e-6, float(self.gain_max))
        return math.log(gmin), math.log(gmax)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.dtype != torch.long:
            t = t.long()
        sqrt_ab = _extract_1d(self.schedule.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mab = _extract_1d(self.schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.dtype != torch.long:
            t = t.long()
        sqrt_recip_ab = _extract_1d(self.schedule.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_ab = _extract_1d(self.schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_ab * x_t - sqrt_recipm1_ab * eps

    def _gain_norm(self, log_gain: torch.Tensor) -> torch.Tensor:
        log_gain = self._sanitize_log_gain(log_gain)
        log_min, log_max = self._log_bounds()
        return ((log_gain - log_min) / max(1e-6, (log_max - log_min))).clamp(0.0, 1.0)

    def _sanitize_log_gain(self, log_gain: torch.Tensor) -> torch.Tensor:
        log_min, log_max = self._log_bounds()
        mid = 0.5 * (float(log_min) + float(log_max))
        log_gain = torch.nan_to_num(log_gain, nan=mid, posinf=float(log_max), neginf=float(log_min))
        return log_gain.clamp(min=float(log_min), max=float(log_max))

    def _compose(self, low_img: torch.Tensor, log_gain: torch.Tensor) -> torch.Tensor:
        low_img = torch.nan_to_num(low_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        log_gain = self._sanitize_log_gain(log_gain)
        out = low_img * torch.exp(log_gain)
        if bool(getattr(self, "highlight_compression", False)):
            knee = float(max(0.5, min(0.98, getattr(self, "highlight_knee", 0.88))))
            ceiling = float(max(knee + 1e-3, min(1.0, getattr(self, "highlight_ceiling", 0.98))))
            over = F.relu(out - knee)
            out = torch.where(out > knee, knee + (ceiling - knee) * (1.0 - torch.exp(-over / max(1e-6, ceiling - knee))), out)
        return out.clamp(0.0, 1.0)

    def _target_log_gain(self, low_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        low_lum = _safe_luma(low_img).clamp_min(eps)
        target_lum = _safe_luma(target_img).clamp_min(eps)
        log_gain = torch.log(target_lum) - torch.log(low_lum)
        return self._sanitize_log_gain(log_gain)

    def _sanitize_correction(self, correction: torch.Tensor) -> torch.Tensor:
        log_min, log_max = self._log_bounds()
        span = max(1e-6, float(log_max - log_min))
        correction = torch.nan_to_num(correction, nan=0.0, posinf=span, neginf=-span)
        return correction.clamp(min=-span, max=span)

    def _zero_ref_loss(self, low_img: torch.Tensor, enhanced: torch.Tensor, log_gain: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "exp": _exposure_loss(enhanced, target=float(self.exposure_target), patch_size=16),
            "spa": _spatial_consistency_loss(enhanced, low_img),
            "col": _color_constancy_loss(enhanced),
            "smooth": _edge_aware_tv(log_gain, low_img, alpha=10.0),
            "sat": _saturation_loss(enhanced, high=0.98),
            "noise": _dark_region_noise_loss(enhanced, low_img),
        }

    @torch.no_grad()
    def sample(
        self,
        low_img: torch.Tensor,
        cond: torch.Tensor,
        steps: int = 50,
        method: str = "ddim",
        eta: float = 0.0,
        return_gain: bool = False,
        control_map: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _ = steps, method, eta, control_map
        b = int(low_img.shape[0])
        coarse_log_gain, _coarse_gain = self.teacher(low_img)
        if t is None:
            tt = int(max(0, min(int(self.timesteps - 1), int(self.default_infer_timestep)))) if self.timesteps > 0 else 0
            t = torch.full((b,), tt, device=low_img.device, dtype=torch.long)
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0].long()
        else:
            t = t.long()
        coarse_log_gain = self._sanitize_log_gain(coarse_log_gain)
        zero_correction = torch.zeros_like(coarse_log_gain)
        noisy = self.q_sample(zero_correction, t, torch.zeros_like(zero_correction))
        eps_pred = self.refiner(noisy, low_img, coarse_log_gain, t, cond)
        eps_pred = torch.nan_to_num(eps_pred, nan=0.0, posinf=0.0, neginf=0.0)
        correction = self._sanitize_correction(self.predict_x0_from_eps(noisy, t, eps_pred))
        refined_log_gain = self._sanitize_log_gain(coarse_log_gain + correction)
        enhanced = self._compose(low_img, refined_log_gain)
        if bool(return_gain):
            return enhanced, self._gain_norm(refined_log_gain)
        return enhanced

    def training_step(
        self,
        low_img: torch.Tensor,
        cond: torch.Tensor,
        lambda_diffusion: float = 0.2,
        lambda_teacher_zero_ref: float = 1.0,
        lambda_refiner_aux: float = 0.5,
        lambda_recon: float = 0.0,
        lambda_correction: float = 1.0,
        lambda_exposure: float = 1.0,
        exposure_mode: str = "symmetric",
        exposure_low: float = 0.42,
        exposure_high: float = 0.72,
        lambda_smooth: float = 0.1,
        lambda_noise: float = 0.05,
        lambda_artifact: float = 0.0,
        artifact_kernel: int = 5,
        artifact_margin: float = 0.01,
        lambda_structure: float = 0.0,
        lambda_cycle: float = 0.0,
        lambda_inverse: float = 0.05,
        illum_edge_alpha: float = 10.0,
        target_img: Optional[torch.Tensor] = None,
        degrade_fn=None,
        structure_loss_fn: Optional[nn.Module] = None,
        control_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _ = lambda_structure, lambda_cycle, degrade_fn, structure_loss_fn, control_map
        b = int(low_img.shape[0])
        coarse_log_gain, coarse_gain = self.teacher(low_img)
        coarse_log_gain = self._sanitize_log_gain(coarse_log_gain)
        coarse_prior = coarse_log_gain.detach()
        coarse_enhanced = self._compose(low_img, coarse_log_gain)

        target_img = None if target_img is None else torch.nan_to_num(target_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        has_target = target_img is not None

        if has_target:
            target_log_gain = self._target_log_gain(low_img, target_img)
            x0_correction = self._sanitize_correction(target_log_gain - coarse_prior)
            t = torch.randint(0, int(self.timesteps), (b,), device=low_img.device, dtype=torch.long)
            noise = torch.randn_like(x0_correction)
            noisy_correction = self.q_sample(x0_correction, t, noise)
            eps_pred = self.refiner(noisy_correction, low_img, coarse_prior, t, cond)
            eps_pred = torch.nan_to_num(eps_pred, nan=0.0, posinf=0.0, neginf=0.0)
            l_diff = F.mse_loss(eps_pred, noise)
        else:
            x0_correction = torch.zeros_like(coarse_log_gain)
            l_diff = coarse_log_gain.new_tensor(0.0)

        # Image-domain guidance is applied to the same deterministic path used at inference:
        # start from zero correction with zero noise at a small timestep and let the
        # conditional refiner inject CLIP-domain normal-light corrections beyond A_c.
        t_ref = int(max(0, min(int(self.timesteps - 1), int(self.default_infer_timestep)))) if self.timesteps > 0 else 0
        t_ref_tensor = torch.full((b,), t_ref, device=low_img.device, dtype=torch.long)
        ref_noisy = self.q_sample(torch.zeros_like(coarse_prior), t_ref_tensor, torch.zeros_like(coarse_prior))
        eps_ref = self.refiner(ref_noisy, low_img, coarse_prior, t_ref_tensor, cond)
        eps_ref = torch.nan_to_num(eps_ref, nan=0.0, posinf=0.0, neginf=0.0)
        correction_pred = self._sanitize_correction(self.predict_x0_from_eps(ref_noisy, t_ref_tensor, eps_ref))
        refined_log_gain = self._sanitize_log_gain(coarse_prior + correction_pred)
        refined_gain = torch.exp(refined_log_gain)
        enhanced = self._compose(low_img, refined_log_gain)

        teacher_losses = self._zero_ref_loss(low_img, coarse_enhanced, coarse_log_gain)
        refined_losses = self._zero_ref_loss(low_img, enhanced, refined_log_gain)
        exposure_mode = str(exposure_mode or "symmetric").lower().strip()
        if exposure_mode in {"band", "range", "bounded"} and not has_target:
            l_exp_refiner = _exposure_band_loss(
                enhanced,
                low=float(exposure_low),
                high=float(exposure_high),
                patch_size=16,
            )
        elif exposure_mode in {"under", "dark", "dark_only"} and not has_target:
            lum = _safe_luma(enhanced)
            pooled = F.avg_pool2d(lum, kernel_size=16, stride=16)
            l_exp_refiner = F.relu(float(exposure_low) - pooled).pow(2).mean()
        else:
            l_exp_refiner = refined_losses["exp"]
        l_artifact = _flat_region_artifact_loss(
            enhanced,
            low_img,
            reference_img=coarse_enhanced,
            kernel_size=int(artifact_kernel),
            margin=float(artifact_margin),
            edge_alpha=float(max(5.0, illum_edge_alpha * 2.0)),
        )
        l_teacher = (
            teacher_losses["exp"]
            + teacher_losses["spa"]
            + 0.5 * teacher_losses["col"]
            + 0.1 * teacher_losses["smooth"]
            + 0.5 * teacher_losses["sat"]
            + float(lambda_noise) * teacher_losses["noise"]
        )
        l_refiner_aux = (
            float(lambda_exposure) * l_exp_refiner
            + 0.5 * refined_losses["spa"]
            + 0.25 * refined_losses["col"]
            + float(lambda_smooth) * _edge_aware_tv(refined_log_gain, low_img, alpha=float(illum_edge_alpha))
            + 0.25 * refined_losses["sat"]
            + float(lambda_noise) * refined_losses["noise"]
            + float(lambda_artifact) * l_artifact
        )
        l_prior = F.smooth_l1_loss(correction_pred, torch.zeros_like(correction_pred))
        if has_target:
            l_recon = F.l1_loss(enhanced, target_img)
            l_target_illum = F.smooth_l1_loss(correction_pred, x0_correction.detach())
        else:
            l_recon = enhanced.new_tensor(0.0)
            l_target_illum = enhanced.new_tensor(0.0)

        loss = (
            float(lambda_diffusion) * l_diff
            + float(lambda_teacher_zero_ref) * l_teacher
            + float(lambda_refiner_aux) * l_refiner_aux
            + float(lambda_inverse) * l_prior
            + float(lambda_recon) * l_recon
            + (float(lambda_correction) if has_target else 0.0) * l_target_illum
        )

        return {
            "loss": loss,
            "l_diff": l_diff.detach(),
            "l_ill": refined_losses["smooth"].detach(),
            "l_inv": l_prior.detach(),
            "l_recon": l_recon.detach(),
            "l_target_illum": l_target_illum.detach(),
            "l_teacher": l_teacher.detach(),
            "l_refiner_aux": l_refiner_aux.detach(),
            "l_exp": l_exp_refiner.detach(),
            "l_noise": refined_losses["noise"].detach(),
            "l_artifact": l_artifact.detach(),
            "enhanced": enhanced,
            "gain": refined_gain,
            "log_gain": refined_log_gain,
            "correction": correction_pred,
            "gain_norm": self._gain_norm(refined_log_gain),
            "coarse_enhanced": coarse_enhanced,
            "coarse_gain": coarse_gain,
            "coarse_log_gain": coarse_log_gain,
        }
