import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding.

    Args:
        t: [B] or [B,1] float tensor. Values typically in [0, T).
        dim: embedding dimension.

    Returns:
        [B, dim] float tensor.
    """

    if t.dim() == 2 and t.shape[1] == 1:
        t = t[:, 0]
    if t.dim() != 1:
        raise ValueError(f"timestep_embedding expects t shape [B] or [B,1], got {tuple(t.shape)}")

    half = dim // 2
    if half <= 0:
        return t.new_zeros((t.shape[0], dim))

    freqs = torch.exp(-math.log(float(max_period)) * torch.arange(0, half, device=t.device).float() / float(half))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, emb.new_zeros((emb.shape[0], 1))], dim=1)
    return emb


class TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)

        g1 = min(int(groups), int(in_ch))
        g2 = min(int(groups), int(out_ch))

        self.norm1 = nn.GroupNorm(num_groups=max(1, g1), num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

        self.norm2 = nn.GroupNorm(num_groups=max(1, g2), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM from time embedding
        scale_shift = self.time_proj(F.silu(temb))  # [B,2*out_ch]
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))

        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class AdaLN2(nn.Module):
    """Adaptive LayerNorm (no affine) modulated by r_ci."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_n = self.norm(x)
        gb = self.to_gamma_beta(cond).unsqueeze(1)  # [B,1,2C]
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma) * x_n + beta


class TokenGate(nn.Module):
    """Token-wise gating sigma(x, cond) in [0,1]."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.to_sigma_x = nn.Linear(dim, 1)
        self.to_sigma_c = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.to_sigma_x.weight)
        nn.init.zeros_(self.to_sigma_x.bias)
        nn.init.zeros_(self.to_sigma_c.weight)
        nn.init.zeros_(self.to_sigma_c.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.to_sigma_x(x) + self.to_sigma_c(cond).unsqueeze(1))


class RCITransformerBlock(nn.Module):
    """Transformer block with r_ci injected via:

    - adaLN (scale/shift)
    - cross-attention (K/V from r_ci token)
    - gating (token-wise gate on attention/FFN updates)

    Self-attention is optional (to avoid O(N^2) at larger resolutions).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        cond_tokens: int = 8,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        control_dino_dim: int = 0,
        control_base_dim: Optional[int] = None,
        control_inject_scale: float = 1.0,
        self.cond_tokens = int(max(1, int(cond_tokens)))

        self.adaln1 = AdaLN2(dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.cond_to_tokens = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * self.cond_tokens),
        )

        self.gate_attn = TokenGate(dim, cond_dim)
        self.drop1 = nn.Dropout(proj_dropout)

        self.adaln2 = AdaLN2(dim, cond_dim)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden, dim),
        )
        self.gate_mlp = TokenGate(dim, cond_dim)
        self.drop2 = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, use_self_attn: bool) -> torch.Tensor:
        x1 = self.adaln1(x, cond)

        if use_self_attn:
            sa, _ = self.self_attn(x1, x1, x1, need_weights=False)
            x = x + self.drop1(self.gate_attn(x, cond) * sa)
        else:
            # Skip quadratic self-attn; keep cross-attn below.
            x = x

        # Cross-attention to r_ci tokens (length=K, O(N*K))
        cond_toks = self.cond_to_tokens(cond).view(cond.shape[0], self.cond_tokens, self.dim)  # [B,K,C]
        ca, _ = self.cross_attn(x1, cond_toks, cond_toks, need_weights=False)
        x = x + self.drop1(self.gate_attn(x, cond) * ca)

        x2 = self.adaln2(x, cond)
        ff = self.mlp(x2)
        x = x + self.drop2(self.gate_mlp(x, cond) * ff)
        return x

    def forward_chunked(self, x: torch.Tensor, cond: torch.Tensor, chunk_tokens: int) -> torch.Tensor:
        """Chunked forward for large N without self-attn.

        This keeps memory bounded for high-resolution feature maps.
        Only uses cross-attn (K small) + MLP, both tokenwise and safe to chunk.
        """

        b, n, c = x.shape
        if chunk_tokens <= 0 or n <= chunk_tokens:
            return self.forward(x, cond=cond, use_self_attn=False)

        cond_toks = self.cond_to_tokens(cond).view(cond.shape[0], self.cond_tokens, self.dim)  # [B,K,C]

        outs = []
        for s in range(0, n, int(chunk_tokens)):
            e = min(n, s + int(chunk_tokens))
            xc = x[:, s:e, :]

            x1 = self.adaln1(xc, cond)
            ca, _ = self.cross_attn(x1, cond_toks, cond_toks, need_weights=False)
            xc = xc + self.drop1(self.gate_attn(xc, cond) * ca)

            x2 = self.adaln2(xc, cond)
            ff = self.mlp(x2)
            xc = xc + self.drop2(self.gate_mlp(xc, cond) * ff)
            outs.append(xc)

        return torch.cat(outs, dim=1)


class Attn2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        depth: int = 1,
        max_tokens: int = 0,
        self_attn_max_tokens: int = 1024,
        cond_tokens: int = 8,
        chunk_tokens: int = 8192,
    ):
        super().__init__()
        self.dim = int(dim)
        self.max_tokens = int(max_tokens)
        self.self_attn_max_tokens = int(self_attn_max_tokens)
        self.chunk_tokens = int(max(0, int(chunk_tokens)))
        self.blocks = nn.ModuleList(
            [
                RCITransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    cond_dim=cond_dim,
                    cond_tokens=int(cond_tokens),
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = int(h * w)
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B,N,C]
        # If max_tokens is set (>0), hard-disable all attention beyond that.
        if self.max_tokens > 0 and n > self.max_tokens:
            return x

        use_self = (self.self_attn_max_tokens > 0) and (n <= self.self_attn_max_tokens)
        for blk in self.blocks:
            if use_self:
                tokens = blk(tokens, cond=cond, use_self_attn=True)
            else:
                tokens = blk.forward_chunked(tokens, cond=cond, chunk_tokens=self.chunk_tokens)
        out = tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return out


class UNetTransformerGain(nn.Module):
    """Shallow conv robust feature + hierarchical UNet + Transformer blocks.

    Decoder dual-head outputs:
    - denoised image I_den (same channels as input)
    - gain map G in [gain_min, gain_max]

    Final enhanced image is composed as: enhanced = I_den * G.

    Interface matches GainMapFormer core:
        forward(x, cond, t=None, return_gain=False, return_dual=False)
            -> enhanced
            -> (enhanced, gain_norm) when return_gain=True
            -> (denoised, gain_norm) when return_dual=True
            -> (enhanced, gain_norm, denoised) when both enabled
    """

    def __init__(
        self,
        in_channels: int,
        cond_dim: int,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_heads: int = 4,
        mid_transformer_depth: int = 4,
        gain_range: Tuple[float, float] = (1.1, 4.0),
        gain_logit_clip: Optional[float] = 6.0,
        gain_sigmoid_temperature: float = 1.0,
        attn_max_tokens: int = 0,
        self_attn_max_tokens: int = 1024,
        attn_cond_tokens: int = 8,
        attn_chunk_tokens: int = 8192,
        gain_channels: int = 1,
        denoise_residual_scale: float = 0.2,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.cond_dim = int(cond_dim)
        self.gain_min, self.gain_max = float(gain_range[0]), float(gain_range[1])
        self.gain_logit_clip = None if gain_logit_clip is None else float(gain_logit_clip)
        self.gain_sigmoid_temperature = float(gain_sigmoid_temperature)
        self.gain_channels = int(max(1, int(gain_channels)))
        self.denoise_residual_scale = float(max(0.0, denoise_residual_scale))

        time_dim = int(base_channels * 4)
        self.time_dim = time_dim
        self.time_mlp = TimeMLP(in_dim=time_dim, out_dim=time_dim)

        # Shallow conv robust feature extractor
        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.down = nn.ModuleList()
        self.skip_channels = []
        ch = int(base_channels)

        for level, mult in enumerate(channel_mults):
            out_ch = int(base_channels * int(mult))
            for _ in range(int(num_res_blocks)):
                self.down.append(ResBlock(ch, out_ch, time_dim=time_dim))
                ch = out_ch
                self.down.append(
                    Attn2D(
                        dim=ch,
                        num_heads=num_heads,
                        cond_dim=cond_dim,
                        depth=1,
                        max_tokens=attn_max_tokens,
                        self_attn_max_tokens=self_attn_max_tokens,
                        cond_tokens=attn_cond_tokens,
                        chunk_tokens=attn_chunk_tokens,
                    )
                )
                self.skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(Downsample(ch))

        self.mid1 = ResBlock(ch, ch, time_dim=time_dim)
        self.mid_attn = Attn2D(
            dim=ch,
            num_heads=num_heads,
            cond_dim=cond_dim,
            depth=int(mid_transformer_depth),
            max_tokens=attn_max_tokens,
            self_attn_max_tokens=self_attn_max_tokens,
            cond_tokens=attn_cond_tokens,
            chunk_tokens=attn_chunk_tokens,
        )
        self.mid2 = ResBlock(ch, ch, time_dim=time_dim)

        self.up = nn.ModuleList()
        skip_chs = list(self.skip_channels)
        for level, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = int(base_channels * int(mult))
            for _ in range(int(num_res_blocks)):
                skip_ch = int(skip_chs.pop())
                self.up.append(ResBlock(ch + skip_ch, out_ch, time_dim=time_dim))
                ch = out_ch
                self.up.append(
                    Attn2D(
                        dim=ch,
                        num_heads=num_heads,
                        cond_dim=cond_dim,
                        depth=1,
                        max_tokens=attn_max_tokens,
                        self_attn_max_tokens=self_attn_max_tokens,
                        cond_tokens=attn_cond_tokens,
                        chunk_tokens=attn_chunk_tokens,
                    )
                )
            if level != 0:
                self.up.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(num_groups=max(1, min(8, ch)), num_channels=ch)
        # Dual-head decoder heads:
        # 1) denoising head predicts a clean-but-still-dark image base
        # 2) gain head predicts illumination amplification field
        self.denoise_head = nn.Conv2d(ch, self.in_channels, kernel_size=3, padding=1)
        self.gain_head = nn.Conv2d(ch, self.gain_channels, kernel_size=3, padding=1)
        # Brightness-guided prior: explicitly feed input brightness into gain prediction.
        self.brightness_guide = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, self.gain_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.brightness_guide[-1].weight)
        nn.init.zeros_(self.brightness_guide[-1].bias)

    def _gain_from_logits(self, gain_logits: torch.Tensor) -> torch.Tensor:
        if self.gain_logit_clip is not None:
            clip = float(abs(self.gain_logit_clip))
            if clip > 0:
                gain_logits = clip * (2.0 / math.pi) * torch.atan(gain_logits / clip)

        temp = float(self.gain_sigmoid_temperature)
        if temp <= 0:
            temp = 1.0

        gain = self.gain_min + (self.gain_max - self.gain_min) * torch.sigmoid(gain_logits / temp)
        return gain

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_gain: bool = False,
        return_dual: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        b, c, h, w = x.shape

        if t is None:
            t = x.new_zeros((b, 1))
        elif t.dim() == 1:
            t = t.view(b, 1)

        temb = timestep_embedding(t, self.time_dim)
        temb = self.time_mlp(temb)

        h0 = self.in_conv(x)

        hs = []
        h_cur = h0
        for m in self.down:
            if isinstance(m, ResBlock):
                h_cur = m(h_cur, temb)
            elif isinstance(m, Attn2D):
                h_cur = m(h_cur, cond)
                hs.append(h_cur)
            else:
                h_cur = m(h_cur)

        h_cur = self.mid1(h_cur, temb)
        h_cur = self.mid_attn(h_cur, cond)
        h_cur = self.mid2(h_cur, temb)

        for m in self.up:
            if isinstance(m, ResBlock):
                if not hs:
                    raise RuntimeError("UNet skip stack empty; architecture mismatch")
                skip = hs.pop()
                h_cur = torch.cat([h_cur, skip], dim=1)
                h_cur = m(h_cur, temb)
            elif isinstance(m, Attn2D):
                h_cur = m(h_cur, cond)
            else:
                h_cur = m(h_cur)

        feat = F.silu(self.out_norm(h_cur))

        # Keep denoising as local correction around input, so illumination is mainly handled by gain.
        denoise_delta = torch.tanh(self.denoise_head(feat))
        denoised = (x + self.denoise_residual_scale * denoise_delta).clamp(0.0, 1.0)
        brightness_prior = 1.0 - x.mean(dim=1, keepdim=True)
        gain_logits = self.gain_head(feat) + self.brightness_guide(brightness_prior)
        gain = self._gain_from_logits(gain_logits)

        out = (denoised * gain).clamp(0.0, 1.0)
        gain_norm = (gain - self.gain_min) / max(1e-6, (self.gain_max - self.gain_min))
        gain_norm = gain_norm.clamp(0.0, 1.0)

        if return_gain and return_dual:
            return out, gain_norm, denoised
        if return_gain:
            return out, gain_norm
        if return_dual:
            return denoised, gain_norm
        return out


def _extract_1d(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract a[t] and reshape to broadcast over x.

    a: [T]
    t: [B] long
    returns: [B,1,1,1] (broadcastable)
    """

    if t.dim() == 2 and t.shape[1] == 1:
        t = t[:, 0]
    if t.dim() != 1:
        raise ValueError(f"t must be [B] or [B,1], got {tuple(t.shape)}")
    out = a.gather(0, t.clamp(0, a.shape[0] - 1))
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


class LinearBetaSchedule(nn.Module):
    """Linear beta schedule for DDPM."""

    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        ts = int(max(1, int(timesteps)))
        b0 = float(beta_start)
        b1 = float(beta_end)
        betas = torch.linspace(b0, b1, steps=ts, dtype=torch.float32)
        betas = betas.clamp(1e-8, 0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([alphas_cumprod[:1], alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(1e-20))
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @property
    def timesteps(self) -> int:
        return int(self.betas.shape[0])


class SobelStructure(nn.Module):
    """Edge/gradient structure extractor producing a 1-channel map in [0, 1]."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] in [0,1]
        if x.ndim != 4:
            raise ValueError(f"SobelStructure expects [B,C,H,W], got {tuple(x.shape)}")
        lum = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(lum, self.kx, padding=1)
        gy = F.conv2d(lum, self.ky, padding=1)
        g = torch.sqrt(gx * gx + gy * gy + 1e-12)
        # normalize per-image for stability
        b = int(g.shape[0])
        g_flat = g.view(b, -1)
        denom = g_flat.amax(dim=1).clamp(min=1e-6).view(b, 1, 1, 1)
        g = (g / denom).clamp(0.0, 1.0)
        return g


def _tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation loss for spatial smoothness (expects [B,C,H,W])."""

    if x.ndim != 4:
        raise ValueError(f"_tv_loss expects 4D tensor [B,C,H,W], got {tuple(x.shape)}")
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


def _edge_aware_log_gain_loss(log_gain: torch.Tensor, low_img: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """Edge-aware smoothness for the single-channel log illumination gain map."""

    if log_gain.ndim != 4 or low_img.ndim != 4:
        raise ValueError(f"Expected 4D tensors, got log_gain={tuple(log_gain.shape)} low_img={tuple(low_img.shape)}")
    if int(log_gain.shape[1]) != 1:
        log_gain = log_gain.mean(dim=1, keepdim=True)

    gray = low_img.mean(dim=1, keepdim=True)
    dx_a = log_gain[:, :, :, 1:] - log_gain[:, :, :, :-1]
    dy_a = log_gain[:, :, 1:, :] - log_gain[:, :, :-1, :]
    dx_i = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy_i = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    wx = torch.exp(-float(alpha) * dx_i.abs()).to(dtype=log_gain.dtype)
    wy = torch.exp(-float(alpha) * dy_i.abs()).to(dtype=log_gain.dtype)
    return (dx_a.abs() * wx).mean() + (dy_a.abs() * wy).mean()


def _inverse_illumination_loss(enhanced: torch.Tensor, gain: torch.Tensor, low_img: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Saturation-aware inverse consistency: enhanced / G should recover the low-light input."""

    recovered = enhanced / (gain + float(eps))
    if int(recovered.shape[1]) == 1 and int(low_img.shape[1]) > 1:
        recovered = recovered.expand(-1, int(low_img.shape[1]), -1, -1)
    return F.l1_loss(recovered, low_img)


class UNetTransformerEpsIllum(nn.Module):
    """Hybrid CNN-Transformer U-Net backbone with semantic (r_ci) cross-attn.

    Inputs:
      - x_t: noisy target image
      - low_img: low-light conditioning image
      - structure map extracted from low_img is concatenated to input

    Outputs:
      - eps_pred: predicted diffusion noise ε
      - gain_map: illumination gain G=exp(A), where A is a single-channel log-gain map
      - gain_norm: normalized to [0,1]
    """

    def __init__(
        self,
        image_channels: int,
        cond_dim: int,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_heads: int = 4,
        mid_transformer_depth: int = 4,
        gain_range: Tuple[float, float] = (1.1, 4.0),
        gain_logit_clip: Optional[float] = 6.0,
        gain_sigmoid_temperature: float = 1.0,
        initial_gain: float = 1.4,
        attn_max_tokens: int = 0,
        self_attn_max_tokens: int = 1024,
        attn_cond_tokens: int = 8,
        attn_chunk_tokens: int = 8192,
        gain_channels: int = 1,
        fic_module: Optional[nn.Module] = None,
        control_dino_dim: int = 0,
        control_base_dim: Optional[int] = None,
        control_inject_scale: float = 1.0,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.cond_dim = int(cond_dim)
        self.gain_min, self.gain_max = float(gain_range[0]), float(gain_range[1])
        self.gain_logit_clip = None if gain_logit_clip is None else float(gain_logit_clip)
        self.gain_sigmoid_temperature = float(gain_sigmoid_temperature)
        self.initial_gain = float(initial_gain)
        self.gain_channels = int(max(1, int(gain_channels)))
        self.fic_module = fic_module

        # ControlNet-like conditioning (optional): DINOv2 spatial map + global semantic embedding.
        self.control_dino_dim = int(max(0, int(control_dino_dim)))
        self.control_base_dim = int(control_base_dim) if control_base_dim is not None else int(base_channels)
        self.control_inject_scale = float(control_inject_scale)

        # concat([x_t, low_img, edge(low_img)])
        self.struct = SobelStructure()
        in_ch_total = int(self.image_channels * 2 + 1)

        time_dim = int(base_channels * 4)
        self.time_dim = time_dim
        self.time_mlp = TimeMLP(in_dim=time_dim, out_dim=time_dim)

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch_total, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.down = nn.ModuleList()
        self.skip_channels = []
        self._down_resblock_chs: List[int] = []
        ch = int(base_channels)
        for level, mult in enumerate(channel_mults):
            out_ch = int(base_channels * int(mult))
            for _ in range(int(num_res_blocks)):
                self.down.append(ResBlock(ch, out_ch, time_dim=time_dim))
                ch = out_ch
                self._down_resblock_chs.append(int(ch))
                self.down.append(
                    Attn2D(
                        dim=ch,
                        num_heads=num_heads,
                        cond_dim=cond_dim,
                        depth=1,
                        max_tokens=attn_max_tokens,
                        self_attn_max_tokens=self_attn_max_tokens,
                        cond_tokens=attn_cond_tokens,
                        chunk_tokens=attn_chunk_tokens,
                    )
                )
                self.skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(Downsample(ch))

        self.mid1 = ResBlock(ch, ch, time_dim=time_dim)
        self.mid_attn = Attn2D(
            dim=ch,
            num_heads=num_heads,
            cond_dim=cond_dim,
            depth=int(mid_transformer_depth),
            max_tokens=attn_max_tokens,
            self_attn_max_tokens=self_attn_max_tokens,
            cond_tokens=attn_cond_tokens,
            chunk_tokens=attn_chunk_tokens,
        )
        self.mid2 = ResBlock(ch, ch, time_dim=time_dim)

        self.up = nn.ModuleList()
        self._up_resblock_chs: List[int] = []
        skip_chs = list(self.skip_channels)
        for level, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = int(base_channels * int(mult))
            for _ in range(int(num_res_blocks)):
                skip_ch = int(skip_chs.pop())
                self.up.append(ResBlock(ch + skip_ch, out_ch, time_dim=time_dim))
                ch = out_ch
                self._up_resblock_chs.append(int(ch))
                self.up.append(
                    Attn2D(
                        dim=ch,
                        num_heads=num_heads,
                        cond_dim=cond_dim,
                        depth=1,
                        max_tokens=attn_max_tokens,
                        self_attn_max_tokens=self_attn_max_tokens,
                        cond_tokens=attn_cond_tokens,
                        chunk_tokens=attn_chunk_tokens,
                    )
                )
            if level != 0:
                self.up.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(num_groups=max(1, min(8, ch)), num_channels=ch)

        # Dual heads
        self.eps_head = nn.Conv2d(ch, self.image_channels, kernel_size=3, padding=1)
        self.illum_head = nn.Conv2d(ch, self.gain_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.illum_head.weight)
        nn.init.constant_(self.illum_head.bias, self._initial_gain_logit())

        # Brightness prior from low-light only (not from x_t)
        self.brightness_guide = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, self.gain_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.brightness_guide[-1].weight)
        nn.init.zeros_(self.brightness_guide[-1].bias)

        # Build control fusion and projection heads.
        if self.control_dino_dim > 0:
            self.ctrl_dino_adapter = nn.Sequential(
                nn.Conv2d(self.control_dino_dim, self.control_base_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.control_base_dim, self.control_base_dim, kernel_size=1),
            )
            self.ctrl_sem_adapter = nn.Sequential(
                nn.Linear(self.cond_dim, self.control_base_dim),
                nn.SiLU(),
                nn.Linear(self.control_base_dim, self.control_base_dim),
            )
            # Gating uses: [dino_map_adapt, semantic_map, brightness_prior]
            self.ctrl_gate = nn.Sequential(
                nn.Conv2d(self.control_base_dim * 2 + 1, self.control_base_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.control_base_dim, 1, kernel_size=1),
            )
            nn.init.zeros_(self.ctrl_gate[-1].weight)
            nn.init.zeros_(self.ctrl_gate[-1].bias)

            self.ctrl_proj_down = nn.ModuleList(
                [nn.Conv2d(self.control_base_dim, int(c), kernel_size=1) for c in self._down_resblock_chs]
            )
            self.ctrl_proj_up = nn.ModuleList(
                [nn.Conv2d(self.control_base_dim, int(c), kernel_size=1) for c in self._up_resblock_chs]
            )
            self.ctrl_proj_mid = nn.Conv2d(self.control_base_dim, int(self.mid1.out_ch), kernel_size=1)

    def _gain_from_logits(self, gain_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gain_logit_clip is not None:
            clip = float(abs(self.gain_logit_clip))
            if clip > 0:
                gain_logits = clip * (2.0 / math.pi) * torch.atan(gain_logits / clip)

        temp = float(self.gain_sigmoid_temperature)
        if temp <= 0:
            temp = 1.0
        gain_min = max(1e-6, float(self.gain_min))
        gain_max = max(gain_min + 1e-6, float(self.gain_max))
        log_min = math.log(gain_min)
        log_max = math.log(gain_max)
        log_gain = log_min + (log_max - log_min) * torch.sigmoid(gain_logits / temp)
        return torch.exp(log_gain), log_gain

    def _initial_gain_logit(self) -> float:
        gain_min = max(1e-6, float(self.gain_min))
        gain_max = max(gain_min + 1e-6, float(self.gain_max))
        target = min(max(float(self.initial_gain), gain_min + 1e-6), gain_max - 1e-6)
        log_min = math.log(gain_min)
        log_max = math.log(gain_max)
        p = (math.log(target) - log_min) / max(1e-6, (log_max - log_min))
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        return math.log(p / (1.0 - p))

    def forward(
        self,
        x_t: torch.Tensor,
        low_img: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        return_gain: bool = False,
        return_log_gain: bool = False,
        control_map: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if x_t.ndim != 4 or low_img.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got x_t={tuple(x_t.shape)} low_img={tuple(low_img.shape)}")
        if x_t.shape[:2] != low_img.shape[:2] or x_t.shape[-2:] != low_img.shape[-2:]:
            raise ValueError(f"x_t and low_img must have same shape, got {tuple(x_t.shape)} vs {tuple(low_img.shape)}")
        b, _c, _h, _w = x_t.shape

        if t.dim() == 2 and t.shape[1] == 1:
            t_in = t[:, 0]
        elif t.dim() == 1:
            t_in = t
        else:
            raise ValueError(f"t must be [B] or [B,1], got {tuple(t.shape)}")

        temb = timestep_embedding(t_in, self.time_dim)
        temb = self.time_mlp(temb)

        edge = self.struct(low_img)
        x_in = torch.cat([x_t, low_img, edge], dim=1)
        h_cur = self.in_conv(x_in)

        # Lightweight single-point fusion: inject CLIP semantic information once at the shallow feature level.
        # When FIC is enabled, we intentionally skip the heavier multi-scale control-map path.
        if self.fic_module is not None:
            h_cur = self.fic_module(h_cur, cond)

        ctrl_base: Optional[torch.Tensor] = None
        if self.fic_module is None and self.control_dino_dim > 0 and control_map is not None:
            if control_map.ndim != 4:
                raise ValueError(f"control_map must be [B,C,h,w], got {tuple(control_map.shape)}")
            if int(control_map.shape[0]) != int(b):
                raise ValueError(f"control_map batch mismatch: {int(control_map.shape[0])} vs {int(b)}")
            if int(control_map.shape[1]) != int(self.control_dino_dim):
                raise ValueError(
                    f"control_map channel mismatch: expected {int(self.control_dino_dim)}, got {int(control_map.shape[1])}"
                )

            dino = self.ctrl_dino_adapter(control_map.to(device=h_cur.device, dtype=h_cur.dtype))
            dino_up = F.interpolate(dino, size=low_img.shape[-2:], mode="bilinear", align_corners=False)

            sem = self.ctrl_sem_adapter(cond.to(dtype=torch.float32)).to(device=h_cur.device, dtype=h_cur.dtype)
            sem_map = sem.view(b, self.control_base_dim, 1, 1).expand(-1, -1, low_img.shape[-2], low_img.shape[-1])

            bright = (1.0 - low_img.mean(dim=1, keepdim=True)).to(dtype=h_cur.dtype)
            gate = torch.sigmoid(self.ctrl_gate(torch.cat([dino_up, sem_map, bright], dim=1)))
            ctrl_base = gate * dino_up + (1.0 - gate) * sem_map

        hs = []
        down_i = 0
        for m in self.down:
            if isinstance(m, ResBlock):
                h_cur = m(h_cur, temb)
                if ctrl_base is not None:
                    ctrl = F.interpolate(ctrl_base, size=h_cur.shape[-2:], mode="bilinear", align_corners=False)
                    h_cur = h_cur + (self.control_inject_scale * self.ctrl_proj_down[down_i](ctrl))
                down_i += 1
            elif isinstance(m, Attn2D):
                h_cur = m(h_cur, cond)
                hs.append(h_cur)
            else:
                h_cur = m(h_cur)

        h_cur = self.mid1(h_cur, temb)
        if ctrl_base is not None:
            ctrl = F.interpolate(ctrl_base, size=h_cur.shape[-2:], mode="bilinear", align_corners=False)
            h_cur = h_cur + (self.control_inject_scale * self.ctrl_proj_mid(ctrl))
        h_cur = self.mid_attn(h_cur, cond)
        h_cur = self.mid2(h_cur, temb)

        up_i = 0
        for m in self.up:
            if isinstance(m, ResBlock):
                if not hs:
                    raise RuntimeError("UNet skip stack empty; architecture mismatch")
                skip = hs.pop()
                h_cur = torch.cat([h_cur, skip], dim=1)
                h_cur = m(h_cur, temb)
                if ctrl_base is not None:
                    ctrl = F.interpolate(ctrl_base, size=h_cur.shape[-2:], mode="bilinear", align_corners=False)
                    h_cur = h_cur + (self.control_inject_scale * self.ctrl_proj_up[up_i](ctrl))
                up_i += 1
            elif isinstance(m, Attn2D):
                h_cur = m(h_cur, cond)
            else:
                h_cur = m(h_cur)

        feat = F.silu(self.out_norm(h_cur))

        eps_pred = self.eps_head(feat)
        brightness_prior = 1.0 - low_img.mean(dim=1, keepdim=True)
        illum_logits = self.illum_head(feat) + self.brightness_guide(brightness_prior)
        gain, log_gain = self._gain_from_logits(illum_logits)
        gain_min = max(1e-6, float(self.gain_min))
        gain_max = max(gain_min + 1e-6, float(self.gain_max))
        log_min = math.log(gain_min)
        log_max = math.log(gain_max)
        gain_norm = (log_gain - log_min) / max(1e-6, (log_max - log_min))
        gain_norm = gain_norm.clamp(0.0, 1.0)

        if return_gain:
            if return_log_gain:
                return eps_pred, gain, gain_norm, log_gain
            return eps_pred, gain, gain_norm
        return eps_pred, gain


class DiffusionControlNetEnhancer(nn.Module):
    """Full diffusion enhancer (DDPM/DDIM) with semantic r_ci conditioning.

    - Semantic prior r_ci is injected via cross-attention blocks in UNetTransformerEpsIllum.
    - Structure prior is Sobel edge map concatenated into the UNet input.
    - Dual heads: ε prediction and illumination gain map.
    - Final enhanced image is always composed as: enhanced = low_img * gain.
    """

    is_diffusion = True
    uses_timestep = True

    def __init__(
        self,
        image_channels: int,
        cond_dim: int,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_heads: int = 4,
        mid_transformer_depth: int = 4,
        gain_range: Tuple[float, float] = (1.1, 4.0),
        gain_logit_clip: Optional[float] = 6.0,
        gain_sigmoid_temperature: float = 1.0,
        initial_gain: float = 1.4,
        gain_channels: int = 1,
        fic_module: Optional[nn.Module] = None,
        control_dino_dim: int = 0,
        control_base_dim: Optional[int] = None,
        control_inject_scale: float = 1.0,
        attn_max_tokens: int = 0,
        self_attn_max_tokens: int = 1024,
        attn_cond_tokens: int = 8,
        attn_chunk_tokens: int = 8192,
        infer_timestep: int = 10,
    ):
        super().__init__()
        self.default_infer_timestep = int(max(0, int(infer_timestep)))
        self.schedule = LinearBetaSchedule(
            timesteps=int(timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
        )
        self.model = UNetTransformerEpsIllum(
            image_channels=int(image_channels),
            cond_dim=int(cond_dim),
            base_channels=int(base_channels),
            channel_mults=tuple(channel_mults),
            num_res_blocks=int(num_res_blocks),
            num_heads=int(num_heads),
            mid_transformer_depth=int(mid_transformer_depth),
            gain_range=gain_range,
            gain_logit_clip=gain_logit_clip,
            gain_sigmoid_temperature=gain_sigmoid_temperature,
            initial_gain=float(initial_gain),
            gain_channels=int(gain_channels),
            fic_module=fic_module,
            control_dino_dim=int(control_dino_dim),
            control_base_dim=control_base_dim,
            control_inject_scale=float(control_inject_scale),
            attn_max_tokens=int(attn_max_tokens),
            self_attn_max_tokens=int(self_attn_max_tokens),
            attn_cond_tokens=int(attn_cond_tokens),
            attn_chunk_tokens=int(attn_chunk_tokens),
        )
        self.image_channels = int(image_channels)

    @property
    def timesteps(self) -> int:
        return int(self.schedule.timesteps)

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

    @torch.no_grad()
    def p_sample_ddpm(
        self,
        x_t: torch.Tensor,
        low_img: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        control_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns x_{t-1} and gain_norm (for inspection)
        if t.dim() == 2 and t.shape[1] == 1:
            t1 = t[:, 0]
        else:
            t1 = t
        if t1.dtype != torch.long:
            t1 = t1.long()

        eps_pred, gain, gain_norm = self.model(x_t, low_img, t1, cond, return_gain=True, control_map=control_map)
        x0_pred = self.predict_x0_from_eps(x_t, t1, eps_pred).clamp(0.0, 1.0)

        coef1 = _extract_1d(self.schedule.posterior_mean_coef1, t1, x_t.shape)
        coef2 = _extract_1d(self.schedule.posterior_mean_coef2, t1, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t
        log_var = _extract_1d(self.schedule.posterior_log_variance_clipped, t1, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero = (t1 != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + nonzero * torch.exp(0.5 * log_var) * noise
        return x_prev, gain_norm

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
        """Single-pass inference from a clean low-light representation.

        Diffusion is used only as a training regularizer. DDPM/DDIM sampling
        arguments are accepted for backward compatibility but intentionally ignored.
        """

        _ = steps, method, eta
        b, c, _h, _w = low_img.shape
        if c != self.image_channels:
            raise ValueError(f"low_img channels {c} != expected {self.image_channels}")

        if t is None:
            tt = int(max(0, min(int(self.timesteps - 1), int(self.default_infer_timestep)))) if self.timesteps > 0 else 0
            t = torch.full((b,), tt, device=low_img.device, dtype=torch.long)
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0].long()
        else:
            t = t.long()

        _eps, gain, gain_norm = self.model(low_img, low_img, t, cond, return_gain=True, control_map=control_map)
        enhanced = (low_img * gain).clamp(0.0, 1.0)
        if bool(return_gain):
            return enhanced, gain_norm
        return enhanced

    def training_step(
        self,
        low_img: torch.Tensor,
        cond: torch.Tensor,
        lambda_diffusion: float = 1.0,
        lambda_recon: float = 0.0,
        lambda_smooth: float = 0.01,
        lambda_structure: float = 0.05,
        lambda_cycle: float = 2.0,
        lambda_inverse: float = 1.0,
        illum_edge_alpha: float = 10.0,
        target_img: Optional[torch.Tensor] = None,
        degrade_fn=None,
        structure_loss_fn: Optional[nn.Module] = None,
        control_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """One paper-aligned training step.

        Forward diffusion perturbs the low-light input during training. The shared
        U-Net predicts both injected noise and a single-channel log-gain map A.
        This local objective contains Ldiff, Lill, and Linv; train.py adds Ldom.
        """

        _ = lambda_recon, lambda_structure, lambda_cycle, target_img, degrade_fn, structure_loss_fn
        b = int(low_img.shape[0])
        x0 = low_img

        t = torch.randint(0, int(self.timesteps), (b,), device=low_img.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        eps_pred, gain, gain_norm, log_gain = self.model(
            x_t,
            low_img,
            t,
            cond,
            return_gain=True,
            return_log_gain=True,
            control_map=control_map,
        )
        l_diff = F.mse_loss(eps_pred, noise)
        enhanced = (low_img * gain).clamp(0.0, 1.0)
        l_ill = _edge_aware_log_gain_loss(log_gain, low_img, alpha=float(illum_edge_alpha))
        l_inv = _inverse_illumination_loss(enhanced, gain, low_img)

        loss = (
            float(lambda_diffusion) * l_diff
            + float(lambda_smooth) * l_ill
            + float(lambda_inverse) * l_inv
        )

        return {
            "loss": loss,
            "l_diff": l_diff.detach(),
            "l_ill": l_ill.detach(),
            "l_inv": l_inv.detach(),
            "enhanced": enhanced,
            "gain": gain,
            "log_gain": log_gain,
            "gain_norm": gain_norm,
        }
