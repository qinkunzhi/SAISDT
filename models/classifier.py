from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

"""CLIP-based image classifier / feature extractor.

核心改动：
- 删除原来的 CNNFeatureExtractor 和 ACPClassifier（KMeans 聚类特征）。
- 使用 CLIP Image Encoder 提取图像语义特征（512/768 维）。
- 在送入 CLIP 之前，对低光图像做简单 Gamma 校正以增强感知亮度。

说明：
- 默认优先使用 open_clip，如果不可用则回退到 transformers 中的 CLIPVisionModel。
- 保留 build_classifier 工厂函数，方便外部代码继续按原方式构造分类器。
"""

try:
    import open_clip  # type: ignore
    _HAS_OPEN_CLIP = True
except ImportError:  # pragma: no cover - 仅在缺少依赖时触发
    _HAS_OPEN_CLIP = False

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor  # type: ignore

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_TRANSFORMERS = False


class CLIPImageClassifier(nn.Module):
    """使用 CLIP Image Encoder 的特征提取模块。

    输入：
        x: Tensor, 形状 [B, 1, H, W] 或 [B, 3, H, W]，数值范围建议在 [0, 1]。

    输出：
        特征向量，形状 [B, D]，D 为 CLIP 的输出维度（如 512/768），
        或由 feature_dim 指定的维度（通过线性投影）。
    """

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        dataset_type: str = "unpaired",
        gamma: float = 0.5,
        image_size: int = 224,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
    ) -> None:
        super().__init__()

        if not _HAS_OPEN_CLIP and not _HAS_TRANSFORMERS:
            raise ImportError(
                "需要安装 open_clip 或 transformers 才能使用 CLIPImageClassifier，"
                "请先安装其中之一，例如：pip install open-clip-torch 或 pip install transformers"
            )

        self.dataset_type = dataset_type  # 目前仅保留接口，不参与逻辑
        self.gamma = gamma
        self.image_size = image_size

        # 注册 CLIP 标准归一化参数（OpenAI CLIP）
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # 优先使用 open_clip，其次回退到 transformers
        if _HAS_OPEN_CLIP:
            self.backend = "open_clip"
            # open_clip 使用 (model_name, pretrained) 标识模型
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=clip_pretrained
            )
            self.clip_model.eval()
            clip_dim = getattr(self.clip_model.visual, "output_dim", None)
            if clip_dim is None:
                # 兜底做一次前向推理确定维度
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, image_size, image_size)
                    feat = self.clip_model.encode_image(dummy)
                    clip_dim = feat.shape[-1]
            self.clip_dim = int(clip_dim)
        else:
            self.backend = "transformers"

            # 将简单名称映射到 transformers 的权重名称
            name_map = {
                "ViT-B-32": "openai/clip-vit-base-patch32",
                "ViT-B-16": "openai/clip-vit-base-patch16",
                "ViT-L-14": "openai/clip-vit-large-patch14",
            }
            hf_name = name_map.get(clip_model_name, clip_model_name)

            # 这里只使用 VisionModel，从像素直接提特征
            self.clip_processor = CLIPImageProcessor.from_pretrained(hf_name)
            self.clip_model = CLIPVisionModel.from_pretrained(hf_name)
            self.clip_model.eval()
            self.clip_dim = int(self.clip_model.config.hidden_size)

        # 默认返回 CLIP 原始维度（512/768 等）。
        # 如果显式指定 feature_dim，使用线性层投影到该维度。
        if feature_dim is not None and feature_dim != self.clip_dim:
            self.proj = nn.Linear(self.clip_dim, feature_dim)
            self.out_dim = int(feature_dim)
        else:
            self.proj = None
            self.out_dim = int(self.clip_dim)

        # 默认将 CLIP 冻结为感知编码器
        for p in self.clip_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向推理：低光图像 -> Gamma 校正 -> CLIP 特征。

        Args:
            x: [B, 1, H, W] 或 [B, 3, H, W]，值建议在 [0, 1]。

        Returns:
            Tensor: [B, out_dim] 的特征表示。
        """

        if x.dim() != 4:
            raise ValueError(f"期望输入为 4 维张量 [B, C, H, W]，得到 {x.shape}")

        b, c, h, w = x.shape

        # 单通道低光图 -> 复制到 3 通道
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c != 3:
            raise ValueError(f"CLIPImageClassifier 仅支持 1 或 3 通道输入，得到 {c} 通道")

        # 简单感知增强：Gamma 校正提升暗部亮度
        # 假定输入已大致归一化到 [0, 1]
        x = torch.clamp(x, 0.0, 1.0)
        if self.gamma is not None and self.gamma > 0:
            x = torch.pow(x, self.gamma)

        # 调整到 CLIP 模型分辨率
        if (h, w) != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # CLIP 标准归一化
        x = (x - self.mean) / self.std


        feats = self._encode_image_feature(x)

        # 与原 ACPClassifier 一样，仅作为特征返回，不参与梯度回传
        return feats.detach()

    def _encode_image_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed image tensor into CLIP feature (keeps gradients).

        Notes:
            - Expects x already resized to (image_size, image_size) and normalized by mean/std.
            - This is used by residual-guidance losses to backpropagate into the generator output.
            - CLIP weights stay frozen by default, but gradients flow w.r.t. x.
        """

        if x.dim() != 4:
            raise ValueError(f"期望输入为 4 维张量 [B, C, H, W]，得到 {x.shape}")
        if x.shape[1] != 3:
            raise ValueError(f"CLIP 预处理后的输入必须是 3 通道，得到 {x.shape[1]} 通道")

        if self.backend == "open_clip":
            feats = self.clip_model.encode_image(x)
        else:  # transformers backend
            outputs = self.clip_model(pixel_values=x)
            feats = outputs.pooler_output

        if self.proj is not None:
            feats = self.proj(feats)
        return feats


def build_classifier(
    feature_dim: Optional[int] = None,
    dataset_type: str = "unpaired",
    gamma: float = 0.5,
    image_size: int = 224,
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
) -> nn.Module:
    """工厂函数，构建基于 CLIP 的图像特征提取器。

    参数：
        feature_dim: 期望的输出特征维度；
            - 为 None 时，直接返回 CLIP 原始维度（如 512/768）。
            - 为整数时，添加一层线性映射到该维度。
        dataset_type: 保留原接口参数，目前未在内部使用。
        gamma: Gamma 校正系数（<1 增强暗部亮度，默认 0.5）。
        image_size: 输入到 CLIP 的图像分辨率（默认 224）。
        clip_model_name: CLIP 模型名称；
            - open_clip: 如 "ViT-B-32"、"ViT-L-14" 等。
            - transformers 回退时，会自动映射到对应的 HF 名称。
        clip_pretrained: open_clip 的预训练权重标签（默认 "openai"）。
    """

    return CLIPImageClassifier(
        feature_dim=feature_dim,
        dataset_type=dataset_type,
        gamma=gamma,
        image_size=image_size,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
    )


class DINOv2SpatialEncoder(nn.Module):
    """DINOv2 spatial feature extractor.

    Returns a low-resolution spatial feature map (patch grid), suitable for structure/geometric priors.
    This module is frozen by default.

    Notes:
      - Uses HuggingFace transformers AutoModel so users can choose dinov2 variants.
      - Applies a simple gamma correction (brighten) before feeding into DINOv2.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        image_size: int = 448,
        gamma: float = 0.5,
        patch_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "需要安装 transformers 才能使用 DINOv2SpatialEncoder，例如：pip install transformers"
            )

        from transformers import AutoModel  # type: ignore

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.gamma = float(gamma)

        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # Infer dims / patch size
        out_dim = getattr(getattr(self.model, "config", None), "hidden_size", None)
        if out_dim is None:
            # best-effort fallback
            out_dim = getattr(getattr(self.model, "config", None), "dim", None)
        if out_dim is None:
            raise ValueError("Could not infer DINOv2 hidden size from model config")
        self.out_dim = int(out_dim)

        if patch_size is None:
            ps = getattr(getattr(self.model, "config", None), "patch_size", None)
            if ps is None:
                ps = getattr(getattr(self.model, "config", None), "patch" , None)
            patch_size = ps
        if patch_size is None:
            # Most dinov2 models use 14.
            patch_size = 14
        self.patch_size = int(patch_size)

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"dino image_size must be divisible by patch_size, got image_size={self.image_size} patch_size={self.patch_size}"
            )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c != 3:
            raise ValueError(f"DINOv2SpatialEncoder expects 1 or 3 channels, got {c}")

        x = x.clamp(0.0, 1.0)
        if self.gamma is not None and self.gamma > 0:
            # gamma < 1 brightens.
            x = torch.pow(x, self.gamma)

        if (h, w) != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        x = (x - self.mean) / self.std

        out = self.model(pixel_values=x)
        # [B, 1+N, D] with CLS token at 0
        hs = getattr(out, "last_hidden_state", None)
        if hs is None:
            raise ValueError("DINOv2 output missing last_hidden_state")

        tokens = hs[:, 1:, :]  # [B, N, D]
        gh = self.image_size // self.patch_size
        gw = gh
        if tokens.shape[1] != gh * gw:
            # best-effort: try to infer square grid
            n = int(tokens.shape[1])
            side = int(round(n ** 0.5))
            if side * side != n:
                raise ValueError(f"Cannot reshape tokens to grid: N={n}")
            gh = side
            gw = side

        feat = tokens.transpose(1, 2).contiguous().view(b, self.out_dim, gh, gw)
        return feat