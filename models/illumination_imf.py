import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x[:, None]
    half = int(dim // 2)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / max(1, half - 1)
    )
    args = x.float() * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if int(emb.shape[1]) < int(dim):
        emb = F.pad(emb, (0, int(dim) - int(emb.shape[1])))
    return emb


class FiLMResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.film = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = self.film(cond).chunk(2, dim=1)
        h = h * (1.0 + gamma) + beta
        return x + self.net(h)


class IlluminationStateIMF(nn.Module):
    """Lightweight conditional Improved MeanFlow in 128D illumination-state space."""

    def __init__(
        self,
        state_dim: int = 128,
        clip_dim: int = 512,
        hidden_dim: int = 256,
        depth: int = 4,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.clip_dim = int(clip_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_embed_dim = int(time_embed_dim)

        self.state_proj = nn.Linear(self.state_dim, self.hidden_dim)
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
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_embed_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList([FiLMResBlock(self.hidden_dim, self.hidden_dim) for _ in range(int(depth))])
        self.u_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.state_dim))
        self.v_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.state_dim))

    def _time_condition(self, r_time: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = t - r_time
        emb_t = sinusoidal_embedding(t, self.time_embed_dim)
        emb_h = sinusoidal_embedding(h, self.time_embed_dim)
        return self.time_proj(torch.cat([emb_t, emb_h], dim=1))

    def forward(
        self,
        z: torch.Tensor,
        r_time: torch.Tensor,
        t: torch.Tensor,
        state_low: torch.Tensor,
        q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if r_time.dim() == 1:
            r_time = r_time[:, None]
        if t.dim() == 1:
            t = t[:, None]
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        cond = self.sem_proj(q) + self.low_state_proj(state_low.float()) + self._time_condition(r_time.float(), t.float())
        x = self.state_proj(z.float()) + cond
        for block in self.blocks:
            x = block(x, cond)
        return self.u_head(x), self.v_head(x)

    @torch.no_grad()
    def infer_correction(self, state_low: torch.Tensor, q: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        b = int(state_low.shape[0])
        eps = torch.randn_like(state_low) if noise is None else noise.to(device=state_low.device, dtype=state_low.dtype)
        r_time = state_low.new_zeros((b, 1))
        t = state_low.new_ones((b, 1))
        u, _v = self.forward(eps, r_time, t, state_low, q)
        return eps - u

    @torch.no_grad()
    def infer_target_state(self, state_low: torch.Tensor, q: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        return state_low + self.infer_correction(state_low, q, noise=noise)
