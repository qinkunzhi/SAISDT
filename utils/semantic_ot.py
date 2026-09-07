from typing import Tuple

import torch
import torch.nn.functional as F


def content_feature_from_clip(z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z.float(), p=2, dim=-1, eps=1e-6)
    r = F.normalize(r.float().view(1, -1).to(device=z.device), p=2, dim=-1, eps=1e-6)
    q = z - (z * r).sum(dim=1, keepdim=True) * r
    return F.normalize(q, p=2, dim=-1, eps=1e-6)


def sinkhorn_plan(cost: torch.Tensor, epsilon: float = 0.05, iterations: int = 50) -> torch.Tensor:
    if cost.ndim != 2:
        raise ValueError(f"Expected cost [B,N], got {tuple(cost.shape)}")
    b, n = int(cost.shape[0]), int(cost.shape[1])
    log_k = -cost.float() / float(max(1e-6, epsilon))
    log_a = cost.new_full((b,), -torch.log(cost.new_tensor(float(b))))
    log_b = cost.new_full((n,), -torch.log(cost.new_tensor(float(n))))
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(int(max(1, iterations))):
        u = log_a - torch.logsumexp(log_k + v.view(1, n), dim=1)
        v = log_b - torch.logsumexp(log_k + u.view(b, 1), dim=0)
    log_pi = log_k + u.view(b, 1) + v.view(1, n)
    return torch.exp(log_pi).clamp_min(0.0)


class SemanticOTCoupler:
    """Semantic-layout OT coupler for unpaired low/high illumination states.

    The layout term uses only the 4x4 and 8x8 spatial state groups after
    per-sample normalization, so it compares relative illumination layout
    instead of absolute exposure.
    """

    def __init__(
        self,
        lambda_sem: float = 1.0,
        lambda_lay: float = 0.2,
        lambda_state: float = None,
        eta: float = 0.5,
        epsilon: float = 0.05,
        iterations: int = 50,
        topk: int = 4,
        sample: bool = True,
        transport_mode: str = "sinkhorn",
        gibbs_temperature: float = 1.0,
    ) -> None:
        self.lambda_sem = float(lambda_sem)
        self.lambda_lay = float(lambda_lay if lambda_state is None else lambda_state)
        self.eta = float(eta)
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.topk = int(max(1, topk))
        self.sample = bool(sample)
        self.transport_mode = str(transport_mode or "sinkhorn").lower().strip()
        self.gibbs_temperature = float(gibbs_temperature)
        if self.transport_mode not in {"sinkhorn", "gibbs"}:
            raise ValueError(f"Unknown transport_mode={transport_mode!r}; expected 'sinkhorn' or 'gibbs'.")

    @staticmethod
    def _normalize_layout(layout: torch.Tensor) -> torch.Tensor:
        layout = layout.float()
        mean = layout.mean(dim=1, keepdim=True)
        std = layout.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (layout - mean) / std

    def layout_cost(self, state_low: torch.Tensor, state_high: torch.Tensor) -> torch.Tensor:
        low4 = self._normalize_layout(state_low[:, 48:64])
        high4 = self._normalize_layout(state_high[:, 48:64])
        low8 = self._normalize_layout(state_low[:, 64:128])
        high8 = self._normalize_layout(state_high[:, 64:128])
        c4 = torch.cdist(low4, high4, p=2).pow(2) / 16.0
        c8 = torch.cdist(low8, high8, p=2).pow(2) / 64.0
        return c4 + float(self.eta) * c8

    def pair(
        self,
        state_low: torch.Tensor,
        state_high: torch.Tensor,
        q_low: torch.Tensor,
        q_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_low = F.normalize(q_low.float(), p=2, dim=-1, eps=1e-6)
        q_high = F.normalize(q_high.float(), p=2, dim=-1, eps=1e-6)
        c_sem = 1.0 - q_low @ q_high.t()
        c_lay = self.layout_cost(state_low, state_high)
        cost = float(self.lambda_sem) * c_sem + float(self.lambda_lay) * c_lay
        if self.transport_mode == "sinkhorn":
            plan = sinkhorn_plan(cost, epsilon=self.epsilon, iterations=self.iterations)
            row_prob = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            temperature = float(max(1e-6, self.gibbs_temperature))
            row_prob = torch.softmax(-cost.float() / temperature, dim=1)
        if self.sample:
            k = int(min(self.topk, row_prob.shape[1]))
            top_prob, top_idx = torch.topk(row_prob, k=k, dim=1)
            top_prob = top_prob / top_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pick = torch.multinomial(top_prob, num_samples=1).view(-1, 1)
            idx = top_idx.gather(1, pick).view(-1)
        else:
            idx = row_prob.argmax(dim=1)
        return state_low, state_high[idx], q_low, idx
