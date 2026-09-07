import hashlib
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_image_files(root: str) -> List[str]:
    files: List[str] = []
    if not root or not os.path.isdir(root):
        return files
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def _dir_fingerprint(root: str) -> str:
    h = hashlib.sha1()
    for p in list_image_files(root):
        try:
            st = os.stat(p)
            rel = os.path.relpath(p, root)
            h.update(f"{rel}|{int(st.st_size)}|{int(st.st_mtime)}\n".encode("utf-8", errors="ignore"))
        except OSError:
            continue
    return h.hexdigest()


def _sha1_str(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def preprocess_for_clip(clip_encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    if int(x.shape[1]) == 1:
        x = x.repeat(1, 3, 1, 1)
    elif int(x.shape[1]) != 3:
        raise ValueError(f"CLIP input must have 1 or 3 channels, got {int(x.shape[1])}")

    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    gamma = float(getattr(clip_encoder, "gamma", 1.0) or 1.0)
    if gamma > 0:
        x = torch.pow(x, gamma)

    image_size = int(getattr(clip_encoder, "image_size", 224) or 224)
    if tuple(x.shape[-2:]) != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)

    mean = getattr(clip_encoder, "mean", None)
    std = getattr(clip_encoder, "std", None)
    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        x = (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype)
    return x


@dataclass
class ResidualDomainConfig:
    pos_root: str
    neg_root: str
    max_images: int = 2000
    cache_dir: str = ".cache_residual"
    force_recompute: bool = False
    show_progress: bool = True
    remove_content_bias: bool = True
    content_ridge: float = 1e-4
    content_tokens: Tuple[str, ...] = (
        "wall",
        "door",
        "ceiling",
        "window",
        "table",
        "chair",
        "floor",
        "room",
        "lamp",
        "furniture",
    )


@torch.no_grad()
def _compute_domain_mean(
    clip_encoder: nn.Module,
    paths: List[str],
    device: torch.device,
    max_images: int,
    show_progress: bool,
    desc: str,
) -> torch.Tensor:
    if not paths:
        raise ValueError(f"No images found for {desc}")
    if max_images > 0 and len(paths) > max_images:
        rng = random.Random(0)
        paths = rng.sample(paths, max_images)

    iterator = paths
    try:
        from tqdm import tqdm  # type: ignore

        if show_progress:
            iterator = tqdm(paths, desc=desc, total=len(paths), dynamic_ncols=True)
    except Exception:
        pass

    running: Optional[torch.Tensor] = None
    used = 0
    skipped = 0
    clip_encoder.eval()

    for path in iterator:
        try:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img).astype("float32") / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)
            x = preprocess_for_clip(clip_encoder, x)
            if hasattr(clip_encoder, "_encode_image_feature"):
                emb = clip_encoder._encode_image_feature(x)  # type: ignore[attr-defined]
            else:
                emb = clip_encoder(x)
            emb = F.normalize(emb.float(), p=2, dim=-1, eps=1e-12)
            running = emb[0].detach().clone() if running is None else running + emb[0].detach()
            used += 1
        except Exception:
            skipped += 1
            continue
        if hasattr(iterator, "set_postfix"):
            try:
                iterator.set_postfix({"used": used, "skipped": skipped})  # type: ignore[attr-defined]
            except Exception:
                pass

    if running is None or used == 0:
        raise ValueError(f"All images failed for {desc}")
    return F.normalize(running / float(used), p=2, dim=-1, eps=1e-12)


def load_or_compute_residual_domain(
    clip_encoder: nn.Module,
    cfg: ResidualDomainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    key = _sha1_str(
        f"pos={os.path.abspath(cfg.pos_root)}|neg={os.path.abspath(cfg.neg_root)}|"
        f"posfp={_dir_fingerprint(cfg.pos_root)}|negfp={_dir_fingerprint(cfg.neg_root)}|"
        f"max={int(cfg.max_images)}|clipdim={getattr(clip_encoder, 'out_dim', 'NA')}|"
        f"imagesz={getattr(clip_encoder, 'image_size', 'NA')}|"
        f"rcb={int(bool(cfg.remove_content_bias))}|ridge={float(cfg.content_ridge)}|"
        f"tokens={','.join(tuple(cfg.content_tokens or ())) }"
    )
    cache_path = os.path.join(cfg.cache_dir, f"residual_domain_{key}.pt")
    if os.path.isfile(cache_path) and not cfg.force_recompute:
        data = torch.load(cache_path, map_location="cpu")
        residual = data.get("residual")
        mean_pos = data.get("mean_pos")
        mean_neg = data.get("mean_neg")
        if all(isinstance(x, torch.Tensor) and x.ndim == 1 for x in (residual, mean_pos, mean_neg)):
            print(f"[ResidualDomain] Loaded cached residual: {cache_path}")
            return residual.to(device), mean_pos.to(device), mean_neg.to(device)

    pos_paths = list_image_files(cfg.pos_root)
    neg_paths = list_image_files(cfg.neg_root)
    print(f"[ResidualDomain] Computing r=mu_normal-mu_low: pos={len(pos_paths)} neg={len(neg_paths)}")
    mean_pos = _compute_domain_mean(clip_encoder, pos_paths, device, int(cfg.max_images), cfg.show_progress, "MeanEmb POS")
    mean_neg = _compute_domain_mean(clip_encoder, neg_paths, device, int(cfg.max_images), cfg.show_progress, "MeanEmb NEG")
    residual = F.normalize(mean_pos - mean_neg, p=2, dim=-1, eps=1e-12)
    residual = suppress_content_bias(
        clip_encoder,
        residual,
        device=device,
        enabled=bool(cfg.remove_content_bias),
        tokens=tuple(cfg.content_tokens or ()),
        ridge=float(cfg.content_ridge),
    )
    torch.save(
        {"residual": residual.detach().cpu(), "mean_pos": mean_pos.detach().cpu(), "mean_neg": mean_neg.detach().cpu()},
        cache_path,
    )
    print(f"[ResidualDomain] Saved residual cache: {cache_path}")
    return residual, mean_pos, mean_neg


@torch.no_grad()
def suppress_content_bias(
    clip_encoder: nn.Module,
    residual: torch.Tensor,
    device: torch.device,
    enabled: bool = True,
    tokens: Tuple[str, ...] = (),
    ridge: float = 1e-4,
) -> torch.Tensor:
    if not bool(enabled) or not tokens:
        return F.normalize(residual, p=2, dim=-1, eps=1e-12)
    if getattr(clip_encoder, "backend", None) != "open_clip" or not hasattr(clip_encoder, "clip_model"):
        print("[ResidualDomain] Content-bias suppression skipped: CLIP text encoder is unavailable.")
        return F.normalize(residual, p=2, dim=-1, eps=1e-12)
    try:
        import open_clip  # type: ignore

        clip_model = getattr(clip_encoder, "clip_model")
        text = open_clip.tokenize([str(t) for t in tokens]).to(device)
        emb = clip_model.encode_text(text).float()
        emb = F.normalize(emb, p=2, dim=-1, eps=1e-12).detach().cpu()
        bmat = emb.t().contiguous()
        rv = residual.detach().float().cpu().view(-1, 1)
        gram = bmat.t() @ bmat
        eye = torch.eye(int(gram.shape[0]), dtype=gram.dtype)
        coeff = torch.linalg.solve(gram + float(ridge) * eye, bmat.t() @ rv)
        cleaned = (rv - bmat @ coeff).view(-1)
        cleaned = F.normalize(cleaned.to(device=residual.device, dtype=residual.dtype), p=2, dim=-1, eps=1e-12)
        print(f"[ResidualDomain] Removed fixed content-token subspace: tokens={len(tokens)}")
        return cleaned
    except Exception as exc:
        print(f"[ResidualDomain] Content-bias suppression skipped: {exc}")
        return F.normalize(residual, p=2, dim=-1, eps=1e-12)


class SemanticDomainConditioner(nn.Module):
    """Build c=[Wd r, Ws z_l] or c=[Wd r, Ws z_l, Wi s_target]."""

    def __init__(self, clip_dim: int, cond_dim: int, state_dim: int = 0):
        super().__init__()
        self.clip_dim = int(clip_dim)
        self.cond_dim = int(cond_dim)
        self.state_dim = int(state_dim)
        self.domain_proj = nn.Sequential(nn.Linear(clip_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.semantic_proj = nn.Sequential(nn.Linear(clip_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.state_proj = None
        if self.state_dim > 0:
            self.state_proj = nn.Sequential(nn.Linear(self.state_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

    def forward(self, semantic: torch.Tensor, domain_prior: torch.Tensor, target_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = int(semantic.shape[0])
        if domain_prior.ndim == 1:
            domain_prior = domain_prior.view(1, -1).expand(b, -1)
        elif domain_prior.ndim == 2 and int(domain_prior.shape[0]) == 1:
            domain_prior = domain_prior.expand(b, -1)
        semantic = F.normalize(semantic.float(), p=2, dim=-1, eps=1e-6)
        domain_prior = F.normalize(domain_prior.float().to(device=semantic.device), p=2, dim=-1, eps=1e-6)
        parts = [self.domain_proj(domain_prior), self.semantic_proj(semantic)]
        if self.state_proj is not None:
            if target_state is None:
                raise ValueError("SemanticDomainConditioner was built with state_dim>0 but target_state is None")
            state = target_state.float().to(device=semantic.device)
            parts.append(self.state_proj(state))
        return torch.cat(parts, dim=-1)


def prepare_semantic_domain_condition(
    clip_encoder: nn.Module,
    conditioner: SemanticDomainConditioner,
    low_img: torch.Tensor,
    residual: torch.Tensor,
    clip_scale: float = 1.0,
    target_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with torch.no_grad():
        x = preprocess_for_clip(clip_encoder, low_img).float()
        if hasattr(clip_encoder, "_encode_image_feature"):
            z_low = clip_encoder._encode_image_feature(x)  # type: ignore[attr-defined]
        else:
            z_low = clip_encoder(x)
        z_low = F.normalize(z_low.float(), p=2, dim=-1, eps=1e-6)
    cond = conditioner(z_low, residual.to(device=low_img.device), target_state=target_state)
    cond = F.normalize(cond, p=2, dim=-1, eps=1e-6)
    return cond * float(clip_scale)


def clip_residual_domain_loss(
    clip_encoder: nn.Module,
    low_img: torch.Tensor,
    enhanced_img: torch.Tensor,
    residual: torch.Tensor,
    mean_pos: Optional[torch.Tensor],
    anchor_weight: float = 0.1,
    image_grad_clip: float = 0.05,
) -> torch.Tensor:
    if residual is None or not hasattr(clip_encoder, "_encode_image_feature"):
        return enhanced_img.new_tensor(0.0)

    with torch.no_grad():
        low_x = preprocess_for_clip(clip_encoder, low_img).float()
        low_phi = clip_encoder._encode_image_feature(low_x).float()  # type: ignore[attr-defined]
        low_phi = F.normalize(low_phi, p=2, dim=-1, eps=1e-6)

    enhanced_for_clip = torch.nan_to_num(enhanced_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if enhanced_for_clip.requires_grad:
        grad_clip = float(max(0.0, image_grad_clip))

        def _clean_clip_image_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            if grad_clip > 0.0:
                grad = grad.clamp(min=-grad_clip, max=grad_clip)
            return grad

        enhanced_for_clip.register_hook(_clean_clip_image_grad)

    enh_x = preprocess_for_clip(clip_encoder, enhanced_for_clip).float()
    if enh_x.is_cuda:
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=False)
    with autocast_ctx:
        enh_phi = clip_encoder._encode_image_feature(enh_x).float()  # type: ignore[attr-defined]
    enh_phi = F.normalize(torch.nan_to_num(enh_phi, nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6)

    r = F.normalize(residual.view(1, -1).to(device=enh_phi.device, dtype=enh_phi.dtype), p=2, dim=-1, eps=1e-6)
    delta = F.normalize(enh_phi - low_phi, p=2, dim=-1, eps=1e-6)
    l_dir = 1.0 - F.cosine_similarity(delta, r.expand_as(delta), dim=-1).mean()

    l_anchor = enhanced_img.new_tensor(0.0)
    if mean_pos is not None and float(anchor_weight) > 0.0:
        mu = F.normalize(mean_pos.view(1, -1).to(device=enh_phi.device, dtype=enh_phi.dtype), p=2, dim=-1, eps=1e-6)
        l_anchor = F.mse_loss(enh_phi, mu.expand_as(enh_phi))
    return l_dir + float(anchor_weight) * l_anchor
