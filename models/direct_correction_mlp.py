import torch
import torch.nn as nn
import torch.nn.functional as F

from models.illumination_imf import FiLMResBlock


class DirectCorrectionMLP(nn.Module):
    """Deterministic correction regressor matched to the iMF conditioning trunk."""

    def __init__(
        self,
        state_dim: int = 128,
        clip_dim: int = 512,
        hidden_dim: int = 256,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.clip_dim = int(clip_dim)
        self.hidden_dim = int(hidden_dim)
        self.low_state_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.sem_proj = nn.Sequential(
            nn.Linear(self.clip_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FiLMResBlock(self.hidden_dim, self.hidden_dim) for _ in range(int(depth))]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

    def forward(self, state_low: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        cond = self.low_state_proj(state_low.float()) + self.sem_proj(q)
        x = cond
        for block in self.blocks:
            x = block(x, cond)
        return self.head(x)

    @torch.no_grad()
    def infer_correction(self, state_low: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.forward(state_low, q)

    @torch.no_grad()
    def infer_target_state(self, state_low: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return state_low + self.infer_correction(state_low, q)
