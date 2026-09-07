import os
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.clip_domain import prepare_semantic_domain_condition
from utils.semantic_ot import content_feature_from_clip


try:
    import pyiqa  # type: ignore

    _HAS_PYIQA = True
except Exception:
    pyiqa = None  # type: ignore
    _HAS_PYIQA = False

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr  # type: ignore
    from skimage.metrics import structural_similarity as sk_ssim  # type: ignore

    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


_PSNR_METRIC = None
_SSIM_METRIC = None
_NIQE_METRIC = None
_MUSIQ_METRIC = None
_MUSIQ_WARNED = False


def _mean_or_none(vals):
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _fmt_metric(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "NA"


def _scist_noise_like(state: torch.Tensor, seed: int, batch_idx: int) -> torch.Tensor:
    gen = torch.Generator(device=state.device)
    gen.manual_seed(int(seed) + int(batch_idx) * 1009)
    return torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=gen)


def _scist_ablation_name(name: str) -> str:
    return str(name or "full").lower().strip()


def _scist_needs_target_state(name: str) -> bool:
    return _scist_ablation_name(name) not in {"clip_only"}


def _scist_condition_mask(cond: torch.Tensor, name: str) -> torch.Tensor:
    if int(cond.shape[-1]) % 3 != 0:
        return cond
    ab = _scist_ablation_name(name)
    md, ms, mi = 1.0, 1.0, 1.0
    if ab == "clip_only":
        mi = 0.0
    elif ab == "target_only":
        md, ms = 0.0, 0.0
    elif ab == "wo_zsem":
        ms = 0.0
    elif ab == "wo_r":
        md = 0.0
    elif ab == "wo_target_cond":
        mi = 0.0
    if md == 1.0 and ms == 1.0 and mi == 1.0:
        return cond
    branch = int(cond.shape[-1]) // 3
    mask = cond.new_tensor([md, ms, mi]).repeat_interleave(branch).view(1, -1)
    return cond * mask


@torch.no_grad()
def evaluate_illum_model(
    model_dict: Dict[str, nn.Module],
    eval_loader: Optional[DataLoader],
    device: torch.device,
    residual: torch.Tensor,
    eval_timestep: int,
    clip_scale: float = 1.0,
    max_eval_batches: int = 0,
    skip_no_ref_metrics: int = 0,
    compare_no_ref_baselines: int = 0,
    scist_modules: Optional[Dict[str, nn.Module]] = None,
    epoch: int = 1,
    scist_seed: int = 123,
    scist_ablation: str = "full",
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if eval_loader is None:
        return None, None, None, None

    global _PSNR_METRIC, _SSIM_METRIC, _NIQE_METRIC, _MUSIQ_METRIC, _MUSIQ_WARNED
    model = model_dict["model"].to(device)
    clip_encoder = model_dict["clip"].to(device)
    conditioner = model_dict["conditioner"].to(device)
    model.eval()
    clip_encoder.eval()
    conditioner.eval()

    psnr_vals = []
    ssim_vals = []
    niqe_vals = []
    musiq_vals = []
    low_niqe_vals = []
    low_musiq_vals = []
    gt_niqe_vals = []
    gt_musiq_vals = []
    compute_no_ref = int(skip_no_ref_metrics) <= 0 or (int(epoch) % int(skip_no_ref_metrics) == 0)
    compare_baselines = bool(int(compare_no_ref_baselines)) and compute_no_ref

    for batch_idx, (low, gt, _name) in enumerate(eval_loader):
        if int(max_eval_batches) > 0 and batch_idx >= int(max_eval_batches):
            break
        low = low.to(device)
        gt = gt.to(device)
        t = torch.full((int(low.shape[0]), 1), float(max(0, int(eval_timestep))), device=device)
        target_state = None
        if scist_modules is not None and _scist_needs_target_state(scist_ablation):
            extractor = scist_modules["state_extractor"]
            normalizer = scist_modules["state_normalizer"]
            imf = scist_modules["imf"]
            state_low = normalizer(extractor(low))
            from utils.clip_domain import preprocess_for_clip

            clip_x = preprocess_for_clip(clip_encoder, low).float()
            z = clip_encoder._encode_image_feature(clip_x).float() if hasattr(clip_encoder, "_encode_image_feature") else clip_encoder(low).float()
            q = content_feature_from_clip(z, residual.to(device))
            noise = _scist_noise_like(state_low, int(scist_seed), batch_idx)
            delta = imf.infer_correction(state_low, q, noise=noise)
            delta_clip = float(scist_modules.get("delta_clip", 0.0) or 0.0)
            if delta_clip > 0.0:
                delta = delta.clamp(min=-delta_clip, max=delta_clip)
            target_state = state_low + delta
            state_min = scist_modules.get("state_min")
            state_max = scist_modules.get("state_max")
            if isinstance(state_min, torch.Tensor) and isinstance(state_max, torch.Tensor):
                target_state = torch.maximum(
                    torch.minimum(target_state, state_max.to(device=target_state.device, dtype=target_state.dtype)),
                    state_min.to(device=target_state.device, dtype=target_state.dtype),
                )
            target_state = target_state.detach()
        if scist_modules is not None and target_state is None:
            target_state = low.new_zeros((int(low.shape[0]), 128))
        cond = prepare_semantic_domain_condition(clip_encoder, conditioner, low, residual, clip_scale=clip_scale, target_state=target_state)
        if scist_modules is not None:
            cond = _scist_condition_mask(cond, scist_ablation)
        enhanced = model(low, t, cond, return_illum=False)
        enhanced = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if int(enhanced.shape[1]) != int(gt.shape[1]):
            if int(enhanced.shape[1]) == 1 and int(gt.shape[1]) == 3:
                enhanced = enhanced.repeat(1, 3, 1, 1)
            elif int(enhanced.shape[1]) == 3 and int(gt.shape[1]) == 1:
                enhanced = enhanced.mean(dim=1, keepdim=True)

        if _HAS_PYIQA:
            try:
                if _PSNR_METRIC is None:
                    _PSNR_METRIC = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr", device=device)
                if _SSIM_METRIC is None:
                    _SSIM_METRIC = pyiqa.create_metric("ssim", device=device)
                psnr = _PSNR_METRIC(enhanced, gt)
                ssim = _SSIM_METRIC(enhanced, gt)
                psnr_vals.extend(psnr.detach().cpu().view(-1).tolist() if isinstance(psnr, torch.Tensor) else [float(psnr)])
                ssim_vals.extend(ssim.detach().cpu().view(-1).tolist() if isinstance(ssim, torch.Tensor) else [float(ssim)])
                if compute_no_ref:
                    if _NIQE_METRIC is None:
                        _NIQE_METRIC = pyiqa.create_metric("niqe", device=device)
                    niqe = _NIQE_METRIC(enhanced)
                    niqe_vals.extend(niqe.detach().cpu().view(-1).tolist() if isinstance(niqe, torch.Tensor) else [float(niqe)])
                    if compare_baselines:
                        low_niqe = _NIQE_METRIC(low)
                        gt_niqe = _NIQE_METRIC(gt)
                        low_niqe_vals.extend(low_niqe.detach().cpu().view(-1).tolist() if isinstance(low_niqe, torch.Tensor) else [float(low_niqe)])
                        gt_niqe_vals.extend(gt_niqe.detach().cpu().view(-1).tolist() if isinstance(gt_niqe, torch.Tensor) else [float(gt_niqe)])
                    try:
                        if _MUSIQ_METRIC is None:
                            _MUSIQ_METRIC = pyiqa.create_metric("musiq", device=device)
                        musiq = _MUSIQ_METRIC(enhanced)
                        musiq_vals.extend(musiq.detach().cpu().view(-1).tolist() if isinstance(musiq, torch.Tensor) else [float(musiq)])
                        if compare_baselines:
                            low_musiq = _MUSIQ_METRIC(low)
                            gt_musiq = _MUSIQ_METRIC(gt)
                            low_musiq_vals.extend(low_musiq.detach().cpu().view(-1).tolist() if isinstance(low_musiq, torch.Tensor) else [float(low_musiq)])
                            gt_musiq_vals.extend(gt_musiq.detach().cpu().view(-1).tolist() if isinstance(gt_musiq, torch.Tensor) else [float(gt_musiq)])
                    except Exception as exc:
                        if not _MUSIQ_WARNED:
                            print(f"[Warning] MUSIQ metric failed or is unavailable in this pyiqa build: {exc}")
                            _MUSIQ_WARNED = True
                continue
            except Exception as exc:
                print(f"[Warning] pyiqa metrics failed, fallback to skimage if available: {exc}")

        if _HAS_SKIMAGE:
            enh_np = enhanced.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            for i in range(enh_np.shape[0]):
                e = enh_np[i].transpose(1, 2, 0)
                g = gt_np[i].transpose(1, 2, 0)
                psnr_vals.append(float(sk_psnr(g, e, data_range=1.0)))
                try:
                    channel_axis = -1 if e.ndim == 3 and e.shape[-1] > 1 else None
                    ssim_vals.append(float(sk_ssim(g.squeeze(), e.squeeze(), data_range=1.0, channel_axis=channel_axis)))
                except Exception:
                    pass

    if compare_baselines:
        print(
            "[Eval][NoRefBaseline] "
            f"NIQE low/enh/gt={_fmt_metric(_mean_or_none(low_niqe_vals))}/"
            f"{_fmt_metric(_mean_or_none(niqe_vals))}/{_fmt_metric(_mean_or_none(gt_niqe_vals))} "
            f"MUSIQ low/enh/gt={_fmt_metric(_mean_or_none(low_musiq_vals))}/"
            f"{_fmt_metric(_mean_or_none(musiq_vals))}/{_fmt_metric(_mean_or_none(gt_musiq_vals))}"
        )

    return _mean_or_none(psnr_vals), _mean_or_none(ssim_vals), _mean_or_none(niqe_vals), _mean_or_none(musiq_vals)


@torch.no_grad()
def save_illum_samples(
    model_dict: Dict[str, nn.Module],
    eval_loader: Optional[DataLoader],
    device: torch.device,
    residual: torch.Tensor,
    save_dir: str,
    epoch: int,
    eval_timestep: int,
    clip_scale: float = 1.0,
    max_save: int = 4,
    seed: int = 0,
    save_gain_map: bool = True,
    scist_modules: Optional[Dict[str, nn.Module]] = None,
    scist_ablation: str = "full",
) -> None:
    if eval_loader is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    model = model_dict["model"].to(device)
    clip_encoder = model_dict["clip"].to(device)
    conditioner = model_dict["conditioner"].to(device)
    model.eval()
    clip_encoder.eval()
    conditioner.eval()

    indices = list(range(len(eval_loader.dataset))) if hasattr(eval_loader, "dataset") else []
    if indices:
        rng = random.Random(int(seed) + int(epoch))
        rng.shuffle(indices)
        keep = set(indices[: int(max_save)])
    else:
        keep = set()

    saved = 0
    for batch_idx, (low, gt, name) in enumerate(eval_loader):
        if indices and batch_idx not in keep:
            continue
        if saved >= int(max_save):
            break
        low = low.to(device)
        gt = gt.to(device)
        t = torch.full((int(low.shape[0]), 1), float(max(0, int(eval_timestep))), device=device)
        target_state = None
        if scist_modules is not None and _scist_needs_target_state(scist_ablation):
            extractor = scist_modules["state_extractor"]
            normalizer = scist_modules["state_normalizer"]
            imf = scist_modules["imf"]
            state_low = normalizer(extractor(low))
            from utils.clip_domain import preprocess_for_clip

            clip_x = preprocess_for_clip(clip_encoder, low).float()
            z = clip_encoder._encode_image_feature(clip_x).float() if hasattr(clip_encoder, "_encode_image_feature") else clip_encoder(low).float()
            q = content_feature_from_clip(z, residual.to(device))
            noise = _scist_noise_like(state_low, int(seed) + int(epoch) * 100003, batch_idx)
            delta = imf.infer_correction(state_low, q, noise=noise)
            delta_clip = float(scist_modules.get("delta_clip", 0.0) or 0.0)
            if delta_clip > 0.0:
                delta = delta.clamp(min=-delta_clip, max=delta_clip)
            target_state = state_low + delta
            state_min = scist_modules.get("state_min")
            state_max = scist_modules.get("state_max")
            if isinstance(state_min, torch.Tensor) and isinstance(state_max, torch.Tensor):
                target_state = torch.maximum(
                    torch.minimum(target_state, state_max.to(device=target_state.device, dtype=target_state.dtype)),
                    state_min.to(device=target_state.device, dtype=target_state.dtype),
                )
            target_state = target_state.detach()
        if scist_modules is not None and target_state is None:
            target_state = low.new_zeros((int(low.shape[0]), 128))
        cond = prepare_semantic_domain_condition(clip_encoder, conditioner, low, residual, clip_scale=clip_scale, target_state=target_state)
        if scist_modules is not None:
            cond = _scist_condition_mask(cond, scist_ablation)
        if save_gain_map:
            enhanced, gain = model(low, t, cond, return_illum=True)
            gain = torch.nan_to_num(gain, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            enhanced = model(low, t, cond, return_illum=False)
            gain = None
        enhanced = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if int(enhanced.shape[1]) != int(gt.shape[1]):
            if int(enhanced.shape[1]) == 1 and int(gt.shape[1]) == 3:
                enhanced = enhanced.repeat(1, 3, 1, 1)
            elif int(enhanced.shape[1]) == 3 and int(gt.shape[1]) == 1:
                enhanced = enhanced.mean(dim=1, keepdim=True)

        sample_name = str(name[0]) if isinstance(name, (list, tuple)) else str(name)
        stem = os.path.splitext(os.path.basename(sample_name))[0]
        save_image(torch.cat([low, enhanced, gt], dim=0), os.path.join(save_dir, f"epoch{epoch:03d}_{saved:03d}_{stem}.png"), nrow=3)
        if gain is not None:
            save_image(gain, os.path.join(save_dir, f"epoch{epoch:03d}_{saved:03d}_{stem}_gain.png"), nrow=1)
        saved += 1
