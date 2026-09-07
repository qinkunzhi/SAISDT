import math
import torch
import torch.nn as nn


class FICModule(nn.Module):
    """Feature Interaction via Cross-Attention (FIC).

    - U-Net 的空间特征 ef_lf 作为 Query（Q）
    - CLIP 语义特征 rci 作为 Key/Value（K/V）

    通过交叉注意力，将全局语义特征注入到空间特征中，
    比简单拼接 + MLP 更精细地建模语义与空间的交互。
    """

    def __init__(self, clip_dim: int = 512, target_dim: int = 256, num_tokens: int = 4):
        """Args:
        clip_dim:   CLIP 图像编码器输出维度（如 512 / 768），作为 rci_proj 的输入维度。
        target_dim: U-Net 特征通道数以及交叉注意力中使用的隐藏维度。
        num_tokens: 将 CLIP 全局特征映射成多少个语义 token，作为 K/V。
        """

        super(FICModule, self).__init__()

        self.clip_dim = clip_dim
        self.target_dim = target_dim
        self.num_tokens = num_tokens

        # 将 CLIP 全局向量映射为若干语义 token，用作 K/V
        # 注意：这里将 rci_proj 的输入维度改为 CLIP 的输出维度 clip_dim
        self.rci_proj = nn.Linear(clip_dim, target_dim * num_tokens)

        # 将 U-Net 的空间特征映射到注意力空间并做简单变换
        self.q_proj = nn.Linear(target_dim, target_dim)
        self.out_proj = nn.Linear(target_dim, target_dim)

        # 特征对齐补偿分支：gamma, beta
        self.gamma_proj = nn.Sequential(
            nn.Linear(clip_dim, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim)
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(clip_dim, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim)
        )
        # 门控注意力分支：sigma
        self.sigma_proj = nn.Sequential(
            nn.Linear(clip_dim, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
            nn.Sigmoid()
        )

    def forward(self, ef_lf: torch.Tensor, rci: torch.Tensor) -> torch.Tensor:
        """Args:
        ef_lf: U-Net 低光增强特征，形状 [B, C, H, W]，要求 C == target_dim。
        rci:   来自 CLIP 的语义特征，形状 [B, clip_dim]。

        Returns:
        融合后的特征，形状 [B, target_dim, H, W]。
        """

        if ef_lf.dim() != 4:
            raise ValueError(f"ef_lf 期望为 4D [B, C, H, W]，但得到 {ef_lf.shape}")

        B, C, H, W = ef_lf.shape

        if C != self.target_dim:
            raise ValueError(
                f"FICModule 要求输入通道数等于 target_dim={self.target_dim}，"
                f"但得到 C={C}"
            )

        if rci.ndim != 2 or rci.shape[1] != self.clip_dim:
            raise ValueError(
                f"错误：RCI 形状应为 [B, {self.clip_dim}]，但得到 {rci.shape}"
            )

        # 准备 Query：来自 U-Net 的空间特征
        # [B, C, H, W] -> [B, H*W, C]
        x = ef_lf.permute(0, 2, 3, 1).reshape(B, H * W, C)
        q = self.q_proj(x)  # [B, H*W, target_dim]

        # 准备 Key/Value：来自 CLIP 的全局语义特征
        # 不要 detach：否则 residual_bias 等可学习项无法通过 FIC 反传梯度。
        rci = rci.contiguous().to(torch.float32).to(ef_lf.device)
        # [B, clip_dim] -> [B, num_tokens, target_dim]
        kv_tokens = self.rci_proj(rci).view(B, self.num_tokens, self.target_dim)
        k = kv_tokens  # [B, T, D]
        v = kv_tokens  # [B, T, D]


        # 计算交叉注意力：Q*K^T / sqrt(D)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.target_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, N, T]
        attn_out = torch.matmul(attn_weights, v)  # [B, N, D]

        # 输出投影
        F_attn = self.out_proj(attn_out)  # [B, N, D]

        # 生成 gamma, beta, sigma
        gamma = self.gamma_proj(rci).unsqueeze(1)  # [B, 1, D]
        beta = self.beta_proj(rci).unsqueeze(1)    # [B, 1, D]
        sigma = self.sigma_proj(rci).unsqueeze(1)  # [B, 1, D], in [0,1]

        # 仿射变换 + 门控
        F_affine = gamma * F_attn + beta  # [B, N, D]
        # 残差分支：原始空间特征
        F_sp = x  # [B, N, D]，即 ef_lf 展平
        F_out = sigma * F_affine + F_sp  # [B, N, D]

        # 还原回 [B, D, H, W]
        F_out = F_out.view(B, H, W, self.target_dim).permute(0, 3, 1, 2)
        return F_out


def build_fic(clip_dim: int = 512, target_dim: int = 256, num_tokens: int = 4):
    """构建 FICModule。

    Args:
        clip_dim:   CLIP 图像编码器输出维度（例如 512 或 768）。
        target_dim: U-Net 对应层的通道数（通常为 256 等）。
        num_tokens: 将 CLIP 全局向量映射成的语义 token 数量。
    """

    return FICModule(clip_dim=clip_dim, target_dim=target_dim, num_tokens=num_tokens)