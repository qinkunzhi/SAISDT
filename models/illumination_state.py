from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _luma(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    if int(x.shape[1]) == 3:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x.mean(dim=1, keepdim=True)


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = float(max(1e-3, sigma))
    radius = int(max(1, round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum().clamp_min(1e-12)


def _separable_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gaussian_kernel1d(float(sigma), x.device, x.dtype)
    pad = int(k.numel() // 2)
    c = int(x.shape[1])
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), ky, groups=c)
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kx, groups=c)
    return x


class IlluminationStateExtractor(nn.Module):
    """Fixed 128D illumination-state extractor.

    The descriptor deliberately ignores high-frequency texture and encodes
    global exposure, soft luminance histogram, and coarse spatial illumination.
    """

    state_dim = 128

    def __init__(
        self,
        eps: float = 1e-4,
        sigma_f: float = 3.0,
        sigma_c: float = 7.0,
        hist_bins: int = 32,
        hist_sigma: float = 1.0 / 32.0,
        soft_kappa: float = 20.0,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.sigma_f = float(sigma_f)
        self.sigma_c = float(sigma_c)
        self.hist_bins = int(hist_bins)
        self.hist_sigma = float(hist_sigma)
        self.soft_kappa = float(soft_kappa)
        thresholds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
        centers = (torch.arange(self.hist_bins, dtype=torch.float32) + 0.5) / float(self.hist_bins)
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("hist_centers", centers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        y = _luma(image)
        eps = float(self.eps)
        l = torch.log(y + eps)
        ln = ((l - torch.log(y.new_tensor(eps))) / (torch.log(y.new_tensor(1.0 + eps)) - torch.log(y.new_tensor(eps))))
        ln = ln.clamp(0.0, 1.0)

        uf = _separable_gaussian_blur(ln, self.sigma_f)
        uc = _separable_gaussian_blur(ln, self.sigma_c)

        flat = uc.flatten(2)
        mean = flat.mean(dim=2)
        std = flat.std(dim=2, unbiased=False)
        mad = (flat - mean.unsqueeze(-1)).abs().mean(dim=2)
        rms = (flat.pow(2).mean(dim=2)).sqrt()

        tau = self.thresholds.to(device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
        dark = torch.sigmoid(float(self.soft_kappa) * (tau - uc)).mean(dim=(2, 3))
        bright = torch.sigmoid(float(self.soft_kappa) * (uc - tau)).mean(dim=(2, 3))
        global_state = torch.cat([mean, std, mad, rms, dark, bright], dim=1)

        centers = self.hist_centers.to(device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
        uf_exp = uf.expand(-1, self.hist_bins, -1, -1)
        hist = torch.exp(-((uf_exp - centers) ** 2) / (2.0 * float(self.hist_sigma) ** 2))
        hist = hist.mean(dim=(2, 3))
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-6)

        s4 = F.adaptive_avg_pool2d(uc, (4, 4)).flatten(1)
        s8 = F.adaptive_avg_pool2d(uf, (8, 8)).flatten(1)
        state = torch.cat([global_state, hist, s4, s8], dim=1)
        if int(state.shape[1]) != self.state_dim:
            raise RuntimeError(f"Illumination state must be 128D, got {tuple(state.shape)}")
        return state


class IlluminationStateNormalizer(nn.Module):
    def __init__(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mean is None:
            mean = torch.zeros(128, dtype=torch.float32)
        if std is None:
            std = torch.ones(128, dtype=torch.float32)
        self.register_buffer("mean", mean.float().view(1, -1))
        self.register_buffer("std", std.float().view(1, -1))
        self.eps = float(eps)

    def forward(self, state_raw: torch.Tensor) -> torch.Tensor:
        return (state_raw.float() - self.mean.to(state_raw.device)) / (self.std.to(state_raw.device) + float(self.eps))

    @classmethod
    def from_file(cls, path: str, device: Optional[torch.device] = None) -> "IlluminationStateNormalizer":
        data = torch.load(path, map_location="cpu")
        if "state_mean" in data and "state_std" in data:
            mean, std = data["state_mean"], data["state_std"]
        elif "mean" in data and "std" in data:
            mean, std = data["mean"], data["std"]
        else:
            raise KeyError(f"State statistics file must contain state_mean/state_std or mean/std: {path}")
        module = cls(mean=mean, std=std)
        if device is not None:
            module = module.to(device)
        return module


@torch.no_grad()
def fit_state_normalizer(states: torch.Tensor) -> Dict[str, torch.Tensor]:
    states = states.float()
    mean = states.mean(dim=0)
    std = states.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"state_mean": mean.cpu(), "state_std": std.cpu()}


def split_state_groups(state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return state[:, :16], state[:, 16:48], state[:, 48:128]
