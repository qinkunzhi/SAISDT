import inspect
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_2d_sincos_pos_embed(
    h: int,
    w: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """2D sin-cos absolute positional embedding.

    Returns: [1, h*w, dim]
    Works with dynamic input sizes (no learned table).
    """

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid pos-embed grid size: h={h}, w={w}")

    def _build_1d(embed_dim: int, pos_1d: torch.Tensor) -> torch.Tensor:
        # pos_1d: [N]
        if embed_dim <= 0:
            return pos_1d.new_zeros((pos_1d.numel(), 0))
        if embed_dim % 2 != 0:
            raise ValueError(f"1D sincos embed_dim must be even, got {embed_dim}")

        half = embed_dim // 2
        # omega: [half]
        omega = torch.arange(half, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / float(half)))
        out = pos_1d.to(dtype=dtype).unsqueeze(1) * omega.unsqueeze(0)  # [N,half]
        return torch.cat([out.sin(), out.cos()], dim=1)  # [N,embed_dim]

    dim_y = dim // 2
    dim_x = dim - dim_y
    # make each axis even for sin/cos pairs
    dim_y_even = dim_y if dim_y % 2 == 0 else dim_y - 1
    dim_x_even = dim_x if dim_x % 2 == 0 else dim_x - 1

    gy = torch.arange(h, device=device, dtype=dtype)
    gx = torch.arange(w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # [h,w]

    pos_y = grid_y.reshape(-1)  # [N]
    pos_x = grid_x.reshape(-1)  # [N]

    emb_y = _build_1d(dim_y_even, pos_y)
    emb_x = _build_1d(dim_x_even, pos_x)
    emb = torch.cat([emb_y, emb_x], dim=1)  # [N, <=dim]

    if emb.shape[1] < dim:
        emb = torch.cat([emb, emb.new_zeros((emb.shape[0], dim - emb.shape[1]))], dim=1)

    return emb.unsqueeze(0)  # [1,N,dim]


class AdaLN(nn.Module):
    """GSF-adaLN：语义驱动的 Adaptive LayerNorm。

    在每个 Transformer Block 的 Self-Attention 与 FFN 之前做调制：
      y = (1 + gamma(r_ci)) * LN(x) + beta(r_ci)

    初始化为接近恒等：gamma≈0, beta≈0。
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_gamma = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        nn.init.zeros_(self.to_gamma[-1].weight)
        nn.init.zeros_(self.to_gamma[-1].bias)
        nn.init.zeros_(self.to_beta[-1].weight)
        nn.init.zeros_(self.to_beta[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,N,C], cond: [B,cond_dim]
        x_n = self.norm(x)
        gamma = self.to_gamma(cond).unsqueeze(1)
        beta = self.to_beta(cond).unsqueeze(1)
        return (1.0 + gamma) * x_n + beta


class TokenWiseGate(nn.Module):
    """token-wise gating：给每个 token 一个标量门控系数 σ。"""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.to_sigma_x = nn.Linear(dim, 1)
        self.to_sigma_c = nn.Linear(cond_dim, 1)

        nn.init.zeros_(self.to_sigma_x.weight)
        nn.init.zeros_(self.to_sigma_x.bias)
        nn.init.zeros_(self.to_sigma_c.weight)
        nn.init.zeros_(self.to_sigma_c.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,N,C], cond: [B,cond_dim]
        return torch.sigmoid(self.to_sigma_x(x) + self.to_sigma_c(cond).unsqueeze(1))  # [B,N,1]


class GSFTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.adaln1 = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.gate1 = TokenWiseGate(dim, cond_dim)
        self.dropout1 = nn.Dropout(proj_dropout)

        self.adaln2 = AdaLN(dim, cond_dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden, dim),
        )
        self.gate2 = TokenWiseGate(dim, cond_dim)
        self.dropout2 = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Attention
        x1 = self.adaln1(x, cond)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + self.dropout1(self.gate1(x, cond) * attn_out)

        # FFN
        x2 = self.adaln2(x, cond)
        ffn_out = self.mlp(x2)
        x = x + self.dropout2(self.gate2(x, cond) * ffn_out)
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping Patchify：用 Conv 产生带重叠的 patch tokens。"""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # x: [B,C,H,W]
        f = self.proj(x)  # [B,embed,H',W']
        b, c, h, w = f.shape
        tokens = f.flatten(2).transpose(1, 2).contiguous()  # [B,N,C]
        return tokens, (h, w)


class GainMapFormer(nn.Module):
    """ConvStem→OverlapPatchify→(GSF-adaLN Transformer blocks)→Unpatchify→GainMap→Retinex."""

    def __init__(
        self,
        in_channels: int = 1,
        cond_dim: int = 512,
        embed_dim: int = 96,
        depth: int = 6,
        num_heads: int = 4,
        patch_size: int = 7,
        patch_stride: int = 4,
        gain_range: Tuple[float, float] = (1.1, 4.0),
        gain_logit_clip: Optional[float] = 6.0,
        gain_sigmoid_temperature: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.gain_min, self.gain_max = gain_range
        self.gain_logit_clip = None if gain_logit_clip is None else float(gain_logit_clip)
        self.gain_sigmoid_temperature = float(gain_sigmoid_temperature)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
        )
        self.patch_embed = OverlapPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride=patch_stride,
        )
        self.blocks = nn.ModuleList(
            [
                GSFTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    cond_dim=cond_dim,
                )
                for _ in range(depth)
            ]
        )

        self.pre_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gain_head = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

        # Cache for dynamic 2D sin-cos pos embed (per module instance).
        self._pos_cache_key = None
        self._pos_cache = None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_gain: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B,C,H,W], cond: [B,cond_dim]
        _ = t
        b, c, h, w = x.shape
        x0 = self.conv_stem(x)

        tokens, (hp, wp) = self.patch_embed(x0)
        pos_key = (int(hp), int(wp), int(tokens.shape[-1]), tokens.device, tokens.dtype)
        if getattr(self, "_pos_cache_key", None) != pos_key or getattr(self, "_pos_cache", None) is None:
            self._pos_cache = _build_2d_sincos_pos_embed(
                h=int(hp),
                w=int(wp),
                dim=int(tokens.shape[-1]),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            self._pos_cache_key = pos_key
        tokens = tokens + self._pos_cache

        for blk in self.blocks:
            tokens = blk(tokens, cond)

        feat = tokens.transpose(1, 2).reshape(b, -1, hp, wp).contiguous()  # [B,embed,hp,wp]
        feat = self.pre_head(feat)
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)

        gain_logits = self.gain_head(feat)

        # Avoid numerical sigmoid saturation (sigmoid(logit) becoming exactly 1.0 or 0.0).
        # IMPORTANT: do NOT use hard clamp here; if logits frequently exceed the bound,
        # clamp makes gain_map a spatial constant (min=mean=max) and kills gradients.
        # Use a smooth squashing so gradients remain non-zero.
        if self.gain_logit_clip is not None:
            clip = float(abs(self.gain_logit_clip))
            if clip > 0:
                gain_logits = clip * (2.0 / math.pi) * torch.atan(gain_logits / clip)
        temp = float(self.gain_sigmoid_temperature)
        if temp is None or temp <= 0:
            temp = 1.0
        gain = self.gain_min + (self.gain_max - self.gain_min) * torch.sigmoid(gain_logits / temp)

        out = (x * gain).clamp(0.0, 1.0)
        if return_gain:
            gain_norm = (gain - self.gain_min) / max(1e-6, (self.gain_max - self.gain_min))
            gain_norm = gain_norm.clamp(0.0, 1.0)
            return out, gain_norm
        return out


class GeneratorModel(nn.Module):
    """为保持旧训练代码兼容，保留 forward(x, t, rci) 与 generate() 接口；t 被忽略。"""

    def __init__(
        self,
        core: nn.Module,
        uses_timestep: bool,
    ):
        super().__init__()
        self.core = core
        self.uses_timestep = bool(uses_timestep)
        # Backward-compatible capability flags across generator variants.
        self.supports_return_dual = hasattr(core, "denoise_head")

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor],
        rci: torch.Tensor,
        return_illum: bool = False,
        return_dual: bool = False,
    ):
        # Diffusion core: prefer calling training_step()/sample() from the training script.
        # Keep a backward-compatible forward() as a slow but functional fallback.
        if bool(getattr(self.core, "is_diffusion", False)):
            # Interpret `x` as low-light conditioning image. The paper-aligned
            # diffusion core uses a single conditional forward pass at inference.
            steps = int(getattr(self.core, "default_sample_steps", 50) or 50)
            method = str(getattr(self.core, "default_sample_method", "ddim") or "ddim")
            eta = float(getattr(self.core, "default_sample_eta", 0.0) or 0.0)
            out = self.core.sample(low_img=x, cond=rci, steps=steps, method=method, eta=eta, return_gain=bool(return_illum), t=t)
            if bool(return_illum):
                enhanced, gain = out
                if bool(return_dual):
                    return enhanced, gain, x
                return enhanced, gain
            if bool(return_dual):
                dummy_gain = torch.zeros_like(x[:, :1, :, :])
                return out, dummy_gain
            return out

        if bool(return_dual) and bool(self.supports_return_dual):
            return self.core(x, rci, t=t, return_gain=return_illum, return_dual=True)

        if bool(return_dual) and not bool(self.supports_return_dual):
            # GainMapFormer has no separate denoise branch; keep interface stable.
            out_gain = self.core(x, rci, t=t, return_gain=True)
            if isinstance(out_gain, tuple):
                out, gain = out_gain
            else:
                out = out_gain
                gain = torch.zeros_like(out[:, :1, :, :])
            if bool(return_illum):
                return out, gain, x
            return x, gain

        return self.core(x, rci, t=t, return_gain=return_illum)

    def training_step(self, *args, **kwargs):
        if hasattr(self.core, "training_step"):
            try:
                sig = inspect.signature(self.core.training_step)
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            except (TypeError, ValueError):
                pass
            return self.core.training_step(*args, **kwargs)
        raise AttributeError("Underlying core has no training_step")

    def sample(self, *args, **kwargs):
        if hasattr(self.core, "sample"):
            return self.core.sample(*args, **kwargs)
        raise AttributeError("Underlying core has no sample")

    def generate(self, x_noisy: torch.Tensor, timesteps: int, rci: torch.Tensor):
        # 兼容旧接口：
        # - 非扩散：单次前向。
        # - 带 timestep 的模型：默认用 t=0（确定性）。
        if bool(self.uses_timestep):
            t0 = x_noisy.new_zeros((x_noisy.shape[0], 1))
            return self.core(x_noisy, rci, t=t0, return_gain=False)
        return self.core(x_noisy, rci, t=None, return_gain=False)


def build_generator(
    fic_module=None,
    timesteps: int = 1000,
    in_channels: int = 1,
    out_channels: int = 1,
    cond_dim: int = 512,
    generator_arch: str = "gainmapformer",
    embed_dim: int = 96,
    depth: int = 6,
    num_heads: int = 4,
    patch_size: int = 7,
    patch_stride: int = 4,
    gain_range: Tuple[float, float] = (1.1, 4.0),
    gain_logit_clip: Optional[float] = 6.0,
    gain_sigmoid_temperature: float = 1.0,
    initial_gain: float = 1.4,
    exposure_target: float = 0.62,
    unet_gain_channels: int = 1,
    denoise_residual_scale: float = 0.2,
    diffusion_beta_start: float = 1e-4,
    diffusion_beta_end: float = 2e-2,
    control_dino_dim: int = 0,
    control_base_dim: Optional[int] = None,
    control_inject_scale: float = 1.0,
    infer_timestep: int = 10,
    chroma_denoise: bool = True,
    chroma_strength: float = 0.65,
    chroma_luma_threshold: float = 0.10,
    chroma_texture_threshold: float = 0.018,
    restore_enabled: bool = True,
    restore_scale: float = 0.08,
    restore_gate_bias: float = -1.2,
    color_enabled: bool = True,
    color_scale: float = 0.18,
    color_smooth_kernel: int = 15,
    fusion_type: str = "gsf",
):
    # fic_module / timesteps / out_channels 暂时保留参数以避免外部调用报错。
    _ = fic_module
    _ = timesteps
    _ = out_channels

    arch = str(generator_arch or "gainmapformer").lower().strip()
    if arch in {"gainmapformer", "gmf", "gain"}:
        core = GainMapFormer(
            in_channels=in_channels,
            cond_dim=cond_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            patch_stride=patch_stride,
            gain_range=gain_range,
            gain_logit_clip=gain_logit_clip,
            gain_sigmoid_temperature=gain_sigmoid_temperature,
        )
        return GeneratorModel(core, uses_timestep=False)

    if arch in {"unet_transformer", "unet", "diffunet", "diffusion_unet"}:
        from models.unet_transformer_diffusion import UNetTransformerGain

        # Reuse existing CLI knobs with minimal new flags:
        # - embed_dim -> base_channels
        # - depth -> mid_transformer_depth
        core = UNetTransformerGain(
            in_channels=in_channels,
            cond_dim=cond_dim,
            base_channels=int(embed_dim),
            num_heads=int(num_heads),
            mid_transformer_depth=int(depth),
            gain_range=gain_range,
            gain_logit_clip=gain_logit_clip,
            gain_sigmoid_temperature=gain_sigmoid_temperature,
            gain_channels=int(unet_gain_channels),
            denoise_residual_scale=float(denoise_residual_scale),
        )
        return GeneratorModel(core, uses_timestep=True)

    if arch in {"gsf_gain_unet", "scist_gsf", "gsf_unet"}:
        from models.gsf_gain_unet import GSFConditionedGainUNet

        core = GSFConditionedGainUNet(
            image_channels=in_channels,
            cond_dim=cond_dim,
            base_channels=int(embed_dim),
            gain_range=gain_range,
            initial_gain=float(initial_gain),
            chroma_denoise=bool(chroma_denoise),
            chroma_strength=float(chroma_strength),
            chroma_luma_threshold=float(chroma_luma_threshold),
            chroma_texture_threshold=float(chroma_texture_threshold),
            restore_enabled=bool(restore_enabled),
            restore_scale=float(restore_scale),
            restore_gate_bias=float(restore_gate_bias),
            color_enabled=bool(color_enabled),
            color_scale=float(color_scale),
            color_smooth_kernel=int(color_smooth_kernel),
            fusion_type=str(fusion_type),
        )
        return GeneratorModel(core, uses_timestep=False)

    if arch in {"diffusion_controlnet", "controlnet_diffusion", "control_diffusion", "ddpm_controlnet"}:
        from models.unet_transformer_diffusion import DiffusionControlNetEnhancer

        core = DiffusionControlNetEnhancer(
            image_channels=in_channels,
            cond_dim=cond_dim,
            timesteps=int(timesteps),
            beta_start=float(diffusion_beta_start),
            beta_end=float(diffusion_beta_end),
            base_channels=int(embed_dim),
            num_heads=int(num_heads),
            mid_transformer_depth=int(depth),
            gain_range=gain_range,
            gain_logit_clip=gain_logit_clip,
            gain_sigmoid_temperature=gain_sigmoid_temperature,
            initial_gain=float(initial_gain),
            gain_channels=int(unet_gain_channels),
            fic_module=fic_module,
            control_dino_dim=int(control_dino_dim),
            control_base_dim=control_base_dim,
            control_inject_scale=float(control_inject_scale),
            infer_timestep=int(infer_timestep),
        )
        return GeneratorModel(core, uses_timestep=True)

    if arch in {"teacher_illum_diffusion", "teacher_conditioned_illum_diffusion", "clip_illum_diffusion"}:
        from models.teacher_illum_diffusion import TeacherConditionedIlluminationDiffusion

        core = TeacherConditionedIlluminationDiffusion(
            image_channels=in_channels,
            cond_dim=cond_dim,
            timesteps=int(timesteps),
            beta_start=float(diffusion_beta_start),
            beta_end=float(diffusion_beta_end),
            base_channels=int(embed_dim),
            num_heads=int(num_heads),
            gain_range=gain_range,
            initial_gain=float(initial_gain),
            infer_timestep=int(infer_timestep),
        )
        setattr(core, "exposure_target", float(exposure_target))
        return GeneratorModel(core, uses_timestep=True)

    raise ValueError(
        f"Unknown generator_arch={generator_arch!r}. Supported: gainmapformer | unet_transformer | gsf_gain_unet | diffusion_controlnet | teacher_illum_diffusion"
    )
