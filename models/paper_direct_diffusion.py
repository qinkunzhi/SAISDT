from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_transformer_diffusion import LinearBetaSchedule, _extract_1d, timestep_embedding


class TimeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(max(1, min(8, in_ch)), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(max(1, min(8, out_ch)), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalDenoiserUNet(nn.Module):
    """Predict noise eps from noisy normal-light image x_t conditioned on low-light image."""

    def __init__(self, image_channels: int = 3, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_dim = int(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        in_ch = int(image_channels) * 2
        self.in_conv = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)

        self.b1 = TimeBlock(base_channels, base_channels, self.time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.b2 = TimeBlock(base_channels, base_channels * 2, self.time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.mid1 = TimeBlock(base_channels * 2, base_channels * 4, self.time_dim)
        self.mid2 = TimeBlock(base_channels * 4, base_channels * 2, self.time_dim)
        self.up2 = TimeBlock(base_channels * 4, base_channels, self.time_dim)
        self.up1 = TimeBlock(base_channels * 2, base_channels, self.time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(max(1, min(8, base_channels)), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, int(image_channels), kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_t: torch.Tensor, low_img: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        x = torch.cat([x_t, low_img], dim=1)
        h0 = self.in_conv(x)
        h1 = self.b1(h0, temb)
        h2 = self.b2(self.down1(h1), temb)
        hm = self.mid2(self.mid1(self.down2(h2), temb), temb)

        u2 = F.interpolate(hm, size=h2.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u2, h2], dim=1), temb)
        u1 = F.interpolate(u2, size=h1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, h1], dim=1), temb)
        return self.out(u1)


class PaperDirectDiffusionEnhancer(nn.Module):
    """Paper-aligned conditional diffusion: low-light input -> enhanced image.

    Training:
        x0 = normal-light target image
        x_t = q(x_t | x0)
        eps_pred = D_theta(x_t, I_low, t)
        L_diff = ||eps_pred - eps||

    Inference:
        start from Gaussian noise and denoise conditioned on I_low.
    """

    def __init__(
        self,
        image_channels: int = 3,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        base_channels: int = 64,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.schedule = LinearBetaSchedule(int(timesteps), float(beta_start), float(beta_end))
        self.denoiser = ConditionalDenoiserUNet(
            image_channels=int(image_channels),
            base_channels=int(base_channels),
        )

    @property
    def timesteps(self) -> int:
        return int(self.schedule.timesteps)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.long()
        sqrt_ab = _extract_1d(self.schedule.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mab = _extract_1d(self.schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.long()
        sqrt_recip_ab = _extract_1d(self.schedule.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_ab = _extract_1d(self.schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_ab * x_t - sqrt_recipm1_ab * eps

    def p_sample(self, x_t: torch.Tensor, low_img: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = self.denoiser(x_t, low_img, t)
        x0 = self.predict_x0_from_eps(x_t, t, eps).clamp(0.0, 1.0)
        coef1 = _extract_1d(self.schedule.posterior_mean_coef1, t.long(), x_t.shape)
        coef2 = _extract_1d(self.schedule.posterior_mean_coef2, t.long(), x_t.shape)
        mean = coef1 * x0 + coef2 * x_t
        log_var = _extract_1d(self.schedule.posterior_log_variance_clipped, t.long(), x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero = (t.long() != 0).float().view(-1, 1, 1, 1)
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    def training_step(
        self,
        low_img: torch.Tensor,
        target_img: torch.Tensor,
        lambda_recon: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        low_img = torch.nan_to_num(low_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        target_img = torch.nan_to_num(target_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        b = int(low_img.shape[0])
        t = torch.randint(0, self.timesteps, (b,), device=low_img.device, dtype=torch.long)
        noise = torch.randn_like(target_img)
        x_t = self.q_sample(target_img, t, noise)
        eps_pred = self.denoiser(x_t, low_img, t)
        l_diff = F.mse_loss(eps_pred, noise)

        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred).clamp(0.0, 1.0)
        l_recon = F.l1_loss(x0_pred, target_img)
        loss = l_diff + float(lambda_recon) * l_recon
        return {
            "loss": loss,
            "l_diff": l_diff.detach(),
            "l_recon": l_recon.detach(),
            "enhanced": x0_pred.detach(),
        }

    @torch.no_grad()
    def sample(self, low_img: torch.Tensor, steps: int = 50) -> torch.Tensor:
        low_img = torch.nan_to_num(low_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        b, c, h, w = low_img.shape
        x = torch.randn((b, c, h, w), device=low_img.device, dtype=low_img.dtype)
        steps = int(max(1, min(int(steps), self.timesteps)))
        times = torch.linspace(self.timesteps - 1, 0, steps, device=low_img.device).long()
        for tt in times:
            t = torch.full((b,), int(tt.item()), device=low_img.device, dtype=torch.long)
            x = self.p_sample(x, low_img, t)
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)
