import argparse
import copy
import math
import os
import random
import shutil
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import LowLightEvalDataset, LowLightOnlyDataset, SyntheticLowLightDataset
from models.classifier import build_classifier
from models.generator import build_generator
from models.illumination_imf import IlluminationStateIMF
from models.illumination_state import IlluminationStateExtractor, IlluminationStateNormalizer, split_state_groups
from utils.clip_domain import (
    ResidualDomainConfig,
    SemanticDomainConditioner,
    clip_residual_domain_loss,
    load_or_compute_residual_domain,
    prepare_semantic_domain_condition,
)
from utils.semantic_ot import content_feature_from_clip
from utils.illum_eval import evaluate_illum_model, save_illum_samples


_MUSIQ_TRAIN_METRIC = None
_MUSIQ_TRAIN_WARNED = False


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-main training entry for CLIP-guided normal-light correction diffusion."
    )

    parser.add_argument("--low_root", type=str, default="/home/zhiqinkun/LLIM/datasets/data/LOLv1/Train/input")
    parser.add_argument("--high_root", type=str, default="/home/zhiqinkun/LLIM/datasets/data/DIV2K_384")
    parser.add_argument("--gt_root", type=str, default="/home/zhiqinkun/LLIM/datasets/data/LOLv1/Train/target")
    parser.add_argument("--save_dir", type=str, default="train_illum")
    parser.add_argument(
        "--stage",
        type=str,
        default="real_adapt",
        choices=("synthetic_pretrain", "real_adapt"),
        help="synthetic_pretrain learns Delta_A_ref from synthetic low/normal pairs; real_adapt is unpaired CLIP-guided adaptation.",
    )
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path from synthetic_pretrain or real_adapt.")
    parser.add_argument("--freeze_teacher", type=int, default=0, help="Freeze the zero-reference teacher; recommended for real_adapt after pretraining.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_num_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--eval_timestep", type=int, default=80)
    parser.add_argument("--diffusion_beta_start", type=float, default=1e-4)
    parser.add_argument("--diffusion_beta_end", type=float, default=2e-2)
    parser.add_argument("--g_embed_dim", type=int, default=96)
    parser.add_argument("--g_heads", type=int, default=4)
    parser.add_argument("--gain_min", type=float, default=1.1)
    parser.add_argument("--gain_max", type=float, default=10.0)
    parser.add_argument("--initial_gain", type=float, default=1.8)
    parser.add_argument("--exposure_target", type=float, default=0.62)

    parser.add_argument("--clip_model_name", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--clip_gamma", type=float, default=0.5)
    parser.add_argument("--clip_image_size", type=int, default=224)
    parser.add_argument("--clip_scale", type=float, default=1.0)
    parser.add_argument("--scist", type=int, default=0, help="Enable SCIST Stage-2 enhancer training with frozen illumination-state iMF.")
    parser.add_argument(
        "--scist_ablation",
        type=str,
        default="full",
        choices=(
            "full",
            "clip_only",
            "target_only",
            "wo_zsem",
            "wo_r",
            "wo_bias_suppression",
            "wo_target_cond",
            "wo_gsf",
            "wo_lstate",
        ),
        help="Strict SCIST main-ablation switch: masks condition branches and losses according to the paper ablation table.",
    )
    parser.add_argument("--scist_imf_ckpt", type=str, default="", help="Path to illumination_imf_latest.pt from train_state_imf.py.")
    parser.add_argument("--scist_priors", type=str, default="", help="Optional Stage-0 priors file containing state_mean/std and residual.")
    parser.add_argument("--scist_cond_dim", type=int, default=128, help="Projection width for each SCIST condition branch p_d/p_s/p_i.")
    parser.add_argument("--scist_domain_decay_start", type=int, default=8, help="SCIST epoch where CLIP direction loss starts decaying.")
    parser.add_argument("--scist_domain_decay_end", type=int, default=20, help="SCIST epoch where CLIP direction loss reaches its floor.")
    parser.add_argument("--scist_domain_min", type=float, default=0.05, help="Minimum SCIST CLIP direction loss weight after decay.")
    parser.add_argument("--scist_chroma_denoise", type=int, default=1, help="Apply deterministic low-SNR chroma attenuation in SCIST gain composition.")
    parser.add_argument("--scist_chroma_strength", type=float, default=0.45)
    parser.add_argument("--scist_chroma_luma_threshold", type=float, default=0.10)
    parser.add_argument("--scist_chroma_texture_threshold", type=float, default=0.018)
    parser.add_argument("--scist_restore", type=int, default=1, help="Enable SNR-gated reflectance/chroma restoration branch in the SCIST enhancer.")
    parser.add_argument("--scist_restore_scale", type=float, default=0.08, help="Maximum absolute RGB residual predicted by the SCIST restoration branch.")
    parser.add_argument("--scist_restore_gate_bias", type=float, default=-1.2, help="Initial bias for the learned restoration gate; lower is more conservative.")
    parser.add_argument("--scist_lambda_restore_reg", type=float, default=0.35, help="Regularize SCIST restoration residual magnitude.")
    parser.add_argument("--scist_lambda_restore_luma", type=float, default=1.0, help="Keep SCIST restoration branch from changing luminance already handled by gain.")
    parser.add_argument("--scist_lambda_restore_identity", type=float, default=0.5, help="Keep SCIST restoration branch inactive in high-SNR reliable regions.")
    parser.add_argument("--scist_lambda_restore_tv", type=float, default=0.15, help="Suppress spatially noisy restoration residuals.")
    parser.add_argument("--scist_color", type=int, default=1, help="Enable low-frequency log-chromaticity correction branch for SCIST.")
    parser.add_argument("--scist_color_scale", type=float, default=0.18, help="Maximum absolute log-color correction before low-pass smoothing.")
    parser.add_argument("--scist_color_smooth_kernel", type=int, default=15, help="Low-pass kernel for SCIST color correction field.")
    parser.add_argument("--scist_lambda_color_reg", type=float, default=0.25, help="Regularize SCIST log-color correction magnitude.")
    parser.add_argument("--scist_lambda_color_tv", type=float, default=0.3, help="Force SCIST color correction to stay low-frequency.")
    parser.add_argument("--scist_lambda_color_luma", type=float, default=1.0, help="Prevent SCIST color branch from changing luminance.")
    parser.add_argument("--scist_lambda_low_snr_chroma", type=float, default=0.8)
    parser.add_argument("--scist_lambda_neutral_color", type=float, default=0.35)
    parser.add_argument("--scist_neutral_color_margin", type=float, default=0.06)
    parser.add_argument("--scist_state_clip_quantile", type=float, default=0.02, help="Clamp iMF target state to normal-domain state quantiles. 0 disables.")
    parser.add_argument("--scist_delta_clip", type=float, default=3.0, help="Clamp normalized iMF correction magnitude per state dimension. 0 disables.")
    parser.add_argument("--scist_lambda_exposure_safety", type=float, default=0.8)
    parser.add_argument("--scist_exposure_delta", type=float, default=0.28)
    parser.add_argument("--scist_exposure_low", type=float, default=0.36)
    parser.add_argument("--scist_exposure_high", type=float, default=0.62)
    parser.add_argument("--scist_exposure_over_weight", type=float, default=4.0)
    parser.add_argument("--scist_lambda_highlight_safety", type=float, default=1.0)
    parser.add_argument("--scist_lambda_quality_stat", type=float, default=0.03)
    parser.add_argument("--scist_lambda_normal_stat", type=float, default=0.02)
    parser.add_argument("--lambda_state", type=float, default=1.0)
    parser.add_argument("--lambda_state_global", type=float, default=1.0)
    parser.add_argument("--lambda_state_hist", type=float, default=1.0)
    parser.add_argument("--lambda_state_spatial", type=float, default=1.0)

    parser.add_argument("--residual_pos_root", type=str, default=None)
    parser.add_argument("--residual_neg_root", type=str, default=None)
    parser.add_argument("--residual_max_images", type=int, default=2000)
    parser.add_argument("--residual_cache_dir", type=str, default=".cache_residual")
    parser.add_argument("--residual_recompute", action="store_true")

    parser.add_argument("--lambda_diffusion", type=float, default=1.0)
    parser.add_argument("--lambda_domain", type=float, default=0.2)
    parser.add_argument("--lambda_domain_anchor", type=float, default=0.1)
    parser.add_argument(
        "--clip_domain_grad_clip",
        type=float,
        default=0.05,
        help="Clamp CLIP residual-loss image gradients before they flow back to the illumination model.",
    )
    parser.add_argument("--lambda_teacher_zero_ref", type=float, default=1.0)
    parser.add_argument("--lambda_refiner_aux", type=float, default=0.5)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_correction", type=float, default=1.0)
    parser.add_argument("--lambda_exposure", type=float, default=1.0)
    parser.add_argument(
        "--real_adapt_exposure_mult",
        type=float,
        default=0.25,
        help="Multiply internal exposure loss in real_adapt. Lower values reduce overexposure drift.",
    )
    parser.add_argument(
        "--real_adapt_exposure_mode",
        type=str,
        default="band",
        choices=("symmetric", "band", "under"),
        help="Exposure loss used inside the model during real_adapt.",
    )
    parser.add_argument("--real_adapt_exposure_low", type=float, default=0.38)
    parser.add_argument("--real_adapt_exposure_high", type=float, default=0.62)
    parser.add_argument("--lambda_illum_smooth", type=float, default=0.1)
    parser.add_argument(
        "--lambda_noise",
        type=float,
        default=0.05,
        help="Weight for dark-region high-frequency noise suppression inside teacher/refiner zero-reference losses.",
    )
    parser.add_argument(
        "--lambda_artifact",
        type=float,
        default=0.0,
        help="Weight for flat/dark-region newly-created high-frequency artifact suppression.",
    )
    parser.add_argument("--artifact_kernel", type=int, default=5)
    parser.add_argument("--artifact_margin", type=float, default=0.01)
    parser.add_argument("--lambda_inverse", type=float, default=0.05)
    parser.add_argument(
        "--lambda_hf_anchor",
        type=float,
        default=0.0,
        help="Real-adapt only. Keep high-frequency details close to the frozen resumed model to suppress late-stage artifacts.",
    )
    parser.add_argument("--hf_anchor_kernel", type=int, default=7)
    parser.add_argument(
        "--lambda_brightness_floor",
        type=float,
        default=0.0,
        help="Real-adapt only. Penalize outputs darker than the resumed/coarse enhancement in dark regions.",
    )
    parser.add_argument("--brightness_floor_ratio", type=float, default=0.95)
    parser.add_argument(
        "--lambda_color_anchor",
        type=float,
        default=0.0,
        help="Real-adapt only. Preserve RGB chromaticity from the resumed/coarse enhancement to suppress color casts.",
    )
    parser.add_argument(
        "--lambda_lowfreq_anchor",
        type=float,
        default=0.0,
        help="Real-adapt only. Weakly preserve low-frequency illumination from the resumed model to prevent CLIP drift.",
    )
    parser.add_argument("--lowfreq_anchor_kernel", type=int, default=31)
    parser.add_argument(
        "--lambda_highlight",
        type=float,
        default=0.0,
        help="Real-adapt only. Penalize local overexposure and channel clipping.",
    )
    parser.add_argument("--highlight_threshold", type=float, default=0.92)
    parser.add_argument("--highlight_margin", type=float, default=0.08)
    parser.add_argument(
        "--highlight_compression",
        type=int,
        default=0,
        help="Enable soft-knee highlight compression in image composition for train/eval/inference.",
    )
    parser.add_argument("--highlight_knee", type=float, default=0.88)
    parser.add_argument("--highlight_ceiling", type=float, default=0.98)
    parser.add_argument(
        "--lambda_white_balance",
        type=float,
        default=0.0,
        help="Real-adapt only. Preserve global channel/chromaticity balance from the frozen first-stage output.",
    )
    parser.add_argument(
        "--lambda_adaptive_exposure",
        type=float,
        default=0.0,
        help="Real-adapt only. Input-adaptive local exposure loss to reduce mixed over/under enhancement.",
    )
    parser.add_argument("--adaptive_exposure_delta", type=float, default=0.24)
    parser.add_argument("--adaptive_exposure_low", type=float, default=0.34)
    parser.add_argument("--adaptive_exposure_high", type=float, default=0.60)
    parser.add_argument("--adaptive_exposure_over_weight", type=float, default=3.0)
    parser.add_argument(
        "--lambda_green_cast",
        type=float,
        default=0.0,
        help="Real-adapt only. Penalize green-channel dominance in neutral regions to suppress global green casts.",
    )
    parser.add_argument("--green_cast_margin", type=float, default=0.025)
    parser.add_argument(
        "--lambda_normal_stat",
        type=float,
        default=0.0,
        help="Real-adapt only. Match unpaired normal-light luminance/color statistics from high_root; does not use LOL target.",
    )
    parser.add_argument("--normal_stat_color_weight", type=float, default=0.3)
    parser.add_argument("--normal_stat_contrast_weight", type=float, default=0.5)
    parser.add_argument(
        "--lambda_quality_stat",
        type=float,
        default=0.0,
        help="Real-adapt only. Match unpaired normal-light local contrast, sharpness, and colorfulness statistics.",
    )
    parser.add_argument("--quality_contrast_weight", type=float, default=1.0)
    parser.add_argument("--quality_sharpness_weight", type=float, default=0.5)
    parser.add_argument("--quality_colorfulness_weight", type=float, default=0.3)
    parser.add_argument(
        "--lambda_low_snr_chroma",
        type=float,
        default=0.0,
        help="Real-adapt only. Suppress unreliable chroma amplified from extremely dark, flat low-SNR regions.",
    )
    parser.add_argument("--low_snr_luma_threshold", type=float, default=0.08)
    parser.add_argument("--low_snr_texture_threshold", type=float, default=0.015)
    parser.add_argument(
        "--lambda_musiq",
        type=float,
        default=0.0,
        help="Real-adapt only. Weak differentiable MUSIQ guidance; use only for short final fine-tuning.",
    )
    parser.add_argument("--musiq_loss_every", type=int, default=4)
    parser.add_argument("--musiq_loss_size", type=int, default=224)
    parser.add_argument("--illum_edge_alpha", type=float, default=10.0)
    parser.add_argument(
        "--real_adapt_curriculum",
        type=int,
        default=0,
        help="Enable staged real-adapt loss curriculum: illumination -> color -> artifact -> stabilize.",
    )
    parser.add_argument("--ra_illum_epochs", type=int, default=0)
    parser.add_argument("--ra_color_epochs", type=int, default=0)
    parser.add_argument("--ra_artifact_epochs", type=int, default=0)
    parser.add_argument("--ra_stable_epochs", type=int, default=0)
    parser.add_argument(
        "--ra_color_weight_mult",
        type=float,
        default=1.5,
        help="Multiplier for color-cast related losses during the real-adapt color phase.",
    )
    parser.add_argument(
        "--ra_final_domain",
        type=float,
        default=0.0,
        help="Optional tiny CLIP-domain weight in the final stabilize phase. Keep 0 for safest color/artifact behavior.",
    )
    parser.add_argument(
        "--ra_final_musiq",
        type=float,
        default=0.0,
        help="Optional weak MUSIQ weight used only in the real-adapt stabilize phase.",
    )
    parser.add_argument(
        "--ra_final_quality_stat",
        type=float,
        default=0.0,
        help="Optional normal-light quality-stat weight used only in the real-adapt stabilize phase.",
    )
    parser.add_argument("--ra_curriculum_verbose", type=int, default=1)

    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--best_from_epoch", type=int, default=1)
    parser.add_argument("--metric_decimals", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument(
        "--save_latest_every",
        type=int,
        default=0,
        help="Legacy latest-sample interval. 0 disables; use --force_save_interval for the paper-main save policy.",
    )
    parser.add_argument(
        "--force_save_interval",
        type=int,
        default=5,
        help="Save checkpoint and samples when no metric-best checkpoint has been saved for this many epochs.",
    )
    parser.add_argument("--save_gain_map", type=int, default=1)
    parser.add_argument("--save_random_seed", type=int, default=123)
    parser.add_argument("--seed", type=int, default=123, help="Training random seed for reproducible ablation runs.")
    parser.add_argument("--skip_no_ref_metrics", type=int, default=0)
    parser.add_argument(
        "--compare_no_ref_baselines",
        type=int,
        default=0,
        help="During eval, also print NIQE/MUSIQ for low input and eval GT to diagnose no-reference metric ceilings.",
    )
    parser.add_argument(
        "--use_niqe_for_best",
        type=int,
        default=0,
        help="0 disables NIQE-only best saves because NIQE may reward artifact-like high-frequency statistics.",
    )
    parser.add_argument("--log_gain_every", type=int, default=50)
    parser.add_argument("--progress_bar", type=int, default=1)

    return parser.parse_args(argv)


def select_device(args: argparse.Namespace) -> torch.device:
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{int(args.gpu)}")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers and int(num_workers) > 0),
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def _real_adapt_curriculum_lengths(args: argparse.Namespace) -> Dict[str, int]:
    total = int(max(1, int(args.epochs)))
    manual = {
        "illumination": int(max(0, int(args.ra_illum_epochs))),
        "color": int(max(0, int(args.ra_color_epochs))),
        "artifact": int(max(0, int(args.ra_artifact_epochs))),
        "stabilize": int(max(0, int(args.ra_stable_epochs))),
    }
    if sum(manual.values()) > 0:
        used = sum(manual.values())
        if used < total:
            manual["stabilize"] += total - used
        return {k: max(1, v) for k, v in manual.items() if v > 0}

    illum = max(1, int(round(total * 0.25)))
    color = max(1, int(round(total * 0.35)))
    artifact = max(1, int(round(total * 0.25)))
    stable = max(1, total - illum - color - artifact)
    if illum + color + artifact + stable > total:
        stable = max(1, total - illum - color - artifact)
    return {
        "illumination": illum,
        "color": color,
        "artifact": artifact,
        "stabilize": stable,
    }


def build_real_adapt_curriculum_args(args: argparse.Namespace, rel_epoch: int) -> Tuple[argparse.Namespace, str, Dict[str, int]]:
    if str(args.stage) != "real_adapt" or int(args.real_adapt_curriculum) <= 0:
        return args, "off", _real_adapt_curriculum_lengths(args)

    lengths = _real_adapt_curriculum_lengths(args)
    e = int(max(1, rel_epoch))
    boundary = 0
    phase = "stabilize"
    for name in ("illumination", "color", "artifact", "stabilize"):
        boundary += int(lengths.get(name, 0))
        if e <= boundary:
            phase = name
            break

    eff = copy.copy(args)

    # Real adaptation remains unsupervised: no diffusion target, no paired recon,
    # no paired correction. The curriculum only changes no-reference weights.
    eff.lambda_diffusion = 0.0
    eff.lambda_recon = 0.0
    eff.lambda_correction = 0.0
    eff.lambda_musiq = 0.0
    eff.lambda_quality_stat = 0.0
    eff.real_adapt_exposure_mode = "band"

    if phase == "illumination":
        # First make the image readable and keep highlight protection active.
        eff.lambda_domain = 0.0
        eff.lambda_domain_anchor = 0.0
        eff.lambda_refiner_aux = 0.55
        eff.lambda_exposure = 0.45
        eff.real_adapt_exposure_mult = 0.10
        eff.real_adapt_exposure_low = 0.34
        eff.real_adapt_exposure_high = 0.56
        eff.lambda_adaptive_exposure = 1.20
        eff.adaptive_exposure_delta = 0.24
        eff.adaptive_exposure_low = 0.34
        eff.adaptive_exposure_high = 0.58
        eff.adaptive_exposure_over_weight = 4.0
        eff.lambda_highlight = 8.0
        eff.highlight_threshold = 0.82
        eff.highlight_margin = 0.02
        eff.lambda_lowfreq_anchor = 0.30
        eff.lambda_hf_anchor = 0.50
        eff.lambda_color_anchor = 0.40
        eff.lambda_white_balance = 0.40
        eff.lambda_green_cast = 0.10
        eff.lambda_low_snr_chroma = 0.20
        eff.lambda_noise = 0.15
        eff.lambda_artifact = 0.20
        eff.lambda_inverse = 0.18
        eff.lambda_normal_stat = 0.0
    elif phase == "color":
        # After brightness is mostly stable, correct global red/green drift.
        color_mult = float(max(0.1, args.ra_color_weight_mult))
        eff.lambda_domain = 0.0
        eff.lambda_domain_anchor = 0.0
        eff.lambda_refiner_aux = 0.35
        eff.lambda_exposure = 0.20
        eff.real_adapt_exposure_mult = 0.06
        eff.real_adapt_exposure_low = 0.34
        eff.real_adapt_exposure_high = 0.56
        eff.lambda_adaptive_exposure = 0.45
        eff.adaptive_exposure_delta = 0.20
        eff.adaptive_exposure_low = 0.34
        eff.adaptive_exposure_high = 0.58
        eff.adaptive_exposure_over_weight = 4.5
        eff.lambda_color_anchor = 1.60 * color_mult
        eff.lambda_white_balance = 1.60 * color_mult
        eff.lambda_green_cast = 0.90 * color_mult
        eff.green_cast_margin = 0.02
        eff.lambda_low_snr_chroma = 1.00 * color_mult
        eff.lambda_normal_stat = 0.05 * color_mult
        eff.normal_stat_color_weight = 0.90 * color_mult
        eff.normal_stat_contrast_weight = 0.25
        eff.lambda_highlight = 7.0
        eff.lambda_lowfreq_anchor = 0.35
        eff.lambda_hf_anchor = 0.60
        eff.lambda_noise = 0.20
        eff.lambda_artifact = 0.35
        eff.lambda_inverse = 0.20
    elif phase == "artifact":
        # Then suppress dark/flat-region amplified noise and fake texture.
        eff.lambda_domain = 0.0
        eff.lambda_domain_anchor = 0.0
        eff.lambda_refiner_aux = 0.35
        eff.lambda_exposure = 0.16
        eff.real_adapt_exposure_mult = 0.05
        eff.lambda_adaptive_exposure = 0.35
        eff.adaptive_exposure_delta = 0.20
        eff.adaptive_exposure_over_weight = 5.0
        eff.lambda_noise = 0.55
        eff.lambda_artifact = 1.00
        eff.artifact_margin = 0.012
        eff.lambda_hf_anchor = 1.10
        eff.lambda_inverse = 0.28
        eff.lambda_color_anchor = 1.20
        eff.lambda_white_balance = 1.20
        eff.lambda_green_cast = 0.60
        eff.lambda_low_snr_chroma = 0.90
        eff.lambda_lowfreq_anchor = 0.35
        eff.lambda_highlight = 6.0
        eff.lambda_normal_stat = 0.03
    else:
        # Final phase is conservative stabilization; CLIP stays off unless
        # explicitly enabled with a tiny --ra_final_domain.
        eff.lambda_domain = float(max(0.0, args.ra_final_domain))
        eff.lambda_domain_anchor = 0.02 if eff.lambda_domain > 0.0 else 0.0
        eff.lambda_refiner_aux = 0.25
        eff.lambda_exposure = 0.12
        eff.real_adapt_exposure_mult = 0.04
        eff.lambda_adaptive_exposure = 0.25
        eff.adaptive_exposure_delta = 0.18
        eff.adaptive_exposure_over_weight = 5.0
        eff.lambda_noise = 0.45
        eff.lambda_artifact = 0.70
        eff.lambda_hf_anchor = 1.00
        eff.lambda_inverse = 0.30
        eff.lambda_color_anchor = 1.30
        eff.lambda_white_balance = 1.30
        eff.lambda_green_cast = 0.60
        eff.lambda_low_snr_chroma = 0.80
        eff.lambda_lowfreq_anchor = 0.40
        eff.lambda_highlight = 6.0
        eff.lambda_normal_stat = 0.03
        eff.lambda_musiq = float(max(float(args.lambda_musiq), float(args.ra_final_musiq)))
        eff.lambda_quality_stat = float(max(float(args.lambda_quality_stat), float(args.ra_final_quality_stat)))

    return eff, phase, lengths


def scist_ablation_name(args: argparse.Namespace) -> str:
    return str(getattr(args, "scist_ablation", "full") or "full").lower().strip()


def scist_uses_bias_suppression(args: argparse.Namespace) -> bool:
    """Whether the CLIP residual prior removes the fixed content-token subspace."""
    return scist_ablation_name(args) != "wo_bias_suppression"


def scist_condition_mask(args: argparse.Namespace) -> Tuple[float, float, float]:
    name = scist_ablation_name(args)
    if name == "clip_only":
        return 1.0, 1.0, 0.0
    if name == "target_only":
        return 0.0, 0.0, 1.0
    if name == "wo_zsem":
        return 1.0, 0.0, 1.0
    if name == "wo_r":
        return 0.0, 1.0, 1.0
    if name == "wo_target_cond":
        return 1.0, 1.0, 0.0
    return 1.0, 1.0, 1.0


def scist_needs_target_state(args: argparse.Namespace) -> bool:
    return scist_ablation_name(args) not in {"clip_only"}


def apply_scist_condition_mask(cond: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if int(getattr(args, "scist", 0)) <= 0 or int(cond.shape[-1]) % 3 != 0:
        return cond
    md, ms, mi = scist_condition_mask(args)
    if md == 1.0 and ms == 1.0 and mi == 1.0:
        return cond
    branch = int(cond.shape[-1]) // 3
    mask = cond.new_tensor([md, ms, mi]).repeat_interleave(branch).view(1, -1)
    return cond * mask


def apply_scist_ablation_to_args(args: argparse.Namespace) -> argparse.Namespace:
    eff = copy.copy(args)
    name = scist_ablation_name(eff)
    if name == "clip_only":
        eff.lambda_state = 0.0
    elif name == "target_only":
        eff.lambda_domain = 0.0
        eff.lambda_domain_anchor = 0.0
    elif name == "wo_lstate":
        eff.lambda_state = 0.0
    return eff


def apply_scist_epoch_schedule(args: argparse.Namespace, rel_epoch: int) -> argparse.Namespace:
    eff = copy.copy(args)
    start = int(max(1, int(getattr(args, "scist_domain_decay_start", 8))))
    end = int(max(start, int(getattr(args, "scist_domain_decay_end", 20))))
    floor = float(max(0.0, getattr(args, "scist_domain_min", 0.05)))
    base = float(getattr(args, "lambda_domain", 0.0))
    if int(rel_epoch) >= start:
        if end <= start:
            ratio = 1.0
        else:
            ratio = min(1.0, max(0.0, (float(rel_epoch) - float(start)) / float(end - start)))
        eff.lambda_domain = base + (floor - base) * ratio
        eff.lambda_domain_anchor = 0.0
    return eff


def build_models(args: argparse.Namespace, device: torch.device, in_channels: int) -> Dict[str, nn.Module]:
    clip = build_classifier(
        feature_dim=None,
        gamma=float(args.clip_gamma),
        image_size=int(args.clip_image_size),
        clip_model_name=str(args.clip_model_name),
        clip_pretrained=str(args.clip_pretrained),
    ).to(device)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False
    try:
        if hasattr(clip, "clip_model"):
            clip.clip_model.float()  # type: ignore[attr-defined]
    except Exception:
        pass

    clip_dim = int(getattr(clip, "out_dim"))
    scist_enabled = bool(int(getattr(args, "scist", 0)))
    cond_branch_dim = int(args.scist_cond_dim) if scist_enabled else clip_dim
    conditioner = SemanticDomainConditioner(
        clip_dim=clip_dim,
        cond_dim=cond_branch_dim,
        state_dim=128 if scist_enabled else 0,
    ).to(device)
    model = build_generator(
        timesteps=int(args.timesteps),
        in_channels=int(in_channels),
        out_channels=int(in_channels),
        cond_dim=int(cond_branch_dim * (3 if scist_enabled else 2)),
        generator_arch="gsf_gain_unet" if scist_enabled else "teacher_illum_diffusion",
        embed_dim=int(args.g_embed_dim),
        depth=0,
        num_heads=int(args.g_heads),
        gain_range=(float(args.gain_min), float(args.gain_max)),
        initial_gain=float(args.initial_gain),
        exposure_target=float(args.exposure_target),
        diffusion_beta_start=float(args.diffusion_beta_start),
        diffusion_beta_end=float(args.diffusion_beta_end),
        infer_timestep=int(args.eval_timestep),
        chroma_denoise=bool(int(args.scist_chroma_denoise)) if scist_enabled else False,
        chroma_strength=float(args.scist_chroma_strength),
        chroma_luma_threshold=float(args.scist_chroma_luma_threshold),
        chroma_texture_threshold=float(args.scist_chroma_texture_threshold),
        restore_enabled=bool(int(args.scist_restore)) if scist_enabled else False,
        restore_scale=float(args.scist_restore_scale),
        restore_gate_bias=float(args.scist_restore_gate_bias),
        color_enabled=bool(int(args.scist_color)) if scist_enabled else False,
        color_scale=float(args.scist_color_scale),
        color_smooth_kernel=int(args.scist_color_smooth_kernel),
        fusion_type="simple" if scist_enabled and scist_ablation_name(args) == "wo_gsf" else "gsf",
    ).to(device)
    core = getattr(model, "core", model)
    if hasattr(core, "highlight_compression"):
        setattr(core, "highlight_compression", bool(int(args.highlight_compression)))
        setattr(core, "highlight_knee", float(args.highlight_knee))
        setattr(core, "highlight_ceiling", float(args.highlight_ceiling))
    return {"clip": clip, "conditioner": conditioner, "model": model}


def load_scist_modules(args: argparse.Namespace, models: Dict[str, nn.Module], device: torch.device) -> Optional[Dict[str, nn.Module]]:
    if int(getattr(args, "scist", 0)) <= 0:
        return None
    ckpt_path = str(getattr(args, "scist_imf_ckpt", "") or "")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError("--scist 1 requires --scist_imf_ckpt pointing to train_state_imf.py output")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    clip_dim = int(ckpt.get("clip_dim", getattr(models["clip"], "out_dim")))
    imf_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    imf = IlluminationStateIMF(
        state_dim=128,
        clip_dim=clip_dim,
        hidden_dim=int(imf_args.get("hidden_dim", 256)),
        depth=int(imf_args.get("depth", 4)),
        time_embed_dim=int(imf_args.get("time_embed_dim", 64)),
    ).to(device)
    imf.load_state_dict(ckpt["model_state"], strict=True)
    imf.eval()
    for p in imf.parameters():
        p.requires_grad = False

    priors_path = str(getattr(args, "scist_priors", "") or "")
    stats = ckpt
    if priors_path:
        if not os.path.isfile(priors_path):
            raise FileNotFoundError(f"--scist_priors not found: {priors_path}")
        stats = torch.load(priors_path, map_location="cpu")
    normalizer = IlluminationStateNormalizer(
        mean=stats["state_mean"],
        std=stats["state_std"],
    ).to(device)
    state_min = None
    state_max = None
    q = float(max(0.0, min(0.25, float(getattr(args, "scist_state_clip_quantile", 0.02)))))
    high_blob = stats.get("high", None) if isinstance(stats, dict) else None
    if q > 0.0 and isinstance(high_blob, dict) and isinstance(high_blob.get("state"), torch.Tensor):
        high_state = high_blob["state"].float()
        if int(high_state.ndim) == 2 and int(high_state.shape[1]) == 128:
            state_min = torch.quantile(high_state, q, dim=0).view(1, -1).to(device)
            state_max = torch.quantile(high_state, 1.0 - q, dim=0).view(1, -1).to(device)
            print(f"[SCIST] Target state clamp from normal-domain quantiles q={q:.3g}")
    extractor = IlluminationStateExtractor().to(device).eval()
    for p in extractor.parameters():
        p.requires_grad = False
    print(f"[SCIST] Loaded frozen iMF from {ckpt_path}")
    return {
        "state_extractor": extractor,
        "state_normalizer": normalizer,
        "imf": imf,
        "state_min": state_min,
        "state_max": state_max,
        "delta_clip": float(getattr(args, "scist_delta_clip", 0.0)),
    }


def set_teacher_trainable(model: nn.Module, trainable: bool) -> None:
    core = getattr(model, "core", model)
    teacher = getattr(core, "teacher", None)
    if teacher is None:
        return
    for p in teacher.parameters():
        p.requires_grad = bool(trainable)
    print(f"[Teacher] trainable={bool(trainable)}")


def load_compatible_state_dict(module: nn.Module, state_dict: Dict[str, torch.Tensor], label: str) -> None:
    current = module.state_dict()
    compatible = {}
    skipped = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and k in current and tuple(current[k].shape) == tuple(v.shape):
            compatible[k] = v
        else:
            skipped.append(k)
    current.update(compatible)
    module.load_state_dict(current, strict=True)
    if skipped:
        print(f"[Resume] Skipped {label} params with incompatible shape: {len(skipped)}")


def load_resume_if_needed(args: argparse.Namespace, models: Dict[str, nn.Module], device: torch.device) -> int:
    if not str(args.resume):
        return 0
    ckpt_path = str(args.resume)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    load_compatible_state_dict(models["model"], ckpt.get("model_state", ckpt), "model")
    if "conditioner_state" in ckpt:
        load_compatible_state_dict(models["conditioner"], ckpt["conditioner_state"], "conditioner")
    start_epoch = int(ckpt.get("epoch", 0))
    print(f"[Resume] Loaded {ckpt_path} (epoch={start_epoch})")
    return start_epoch


def build_frozen_anchor(models: Dict[str, nn.Module], device: torch.device) -> Optional[Dict[str, nn.Module]]:
    anchor = {
        "model": copy.deepcopy(models["model"]).to(device).eval(),
        "conditioner": copy.deepcopy(models["conditioner"]).to(device).eval(),
    }
    for module in anchor.values():
        for p in module.parameters():
            p.requires_grad = False
    print("[Anchor] Frozen resumed model for high-frequency anti-artifact regularization.")
    return anchor


def high_frequency_anchor_loss(enhanced: torch.Tensor, anchor: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1

    def _highpass(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        pad = k // 2
        low = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
        return x - low

    return F.l1_loss(_highpass(enhanced), _highpass(anchor))


def _luma(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[1]) == 3:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x.mean(dim=1, keepdim=True)


def brightness_floor_loss(enhanced: torch.Tensor, reference: torch.Tensor, low_img: torch.Tensor, ratio: float = 0.95) -> torch.Tensor:
    enh_l = _luma(enhanced)
    ref_l = _luma(reference.detach())
    low_l = _luma(low_img.detach())
    dark_weight = (1.0 - low_l).clamp(0.0, 1.0)
    floor = ref_l * float(ratio)
    return (F.relu(floor - enh_l) * dark_weight).mean()


def color_anchor_loss(enhanced: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if int(enhanced.shape[1]) != 3 or int(reference.shape[1]) != 3:
        return enhanced.new_tensor(0.0)
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ref = torch.nan_to_num(reference.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    enh_chr = enh / (enh.sum(dim=1, keepdim=True) + 1e-4)
    ref_chr = ref / (ref.sum(dim=1, keepdim=True) + 1e-4)
    return F.smooth_l1_loss(enh_chr, ref_chr)


def low_frequency_anchor_loss(enhanced: torch.Tensor, reference: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ref = torch.nan_to_num(reference.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    enh_low = F.avg_pool2d(F.pad(_luma(enh), (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
    ref_low = F.avg_pool2d(F.pad(_luma(ref), (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
    return F.smooth_l1_loss(enh_low, ref_low)


def highlight_loss(enhanced: torch.Tensor, reference: torch.Tensor, threshold: float = 0.92, margin: float = 0.08) -> torch.Tensor:
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ref = torch.nan_to_num(reference.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    thr = float(max(0.5, min(0.99, threshold)))
    margin = float(max(0.0, min(0.3, margin)))
    enh_l = _luma(enh)
    ref_l = _luma(ref)
    adaptive_ceiling = torch.minimum(ref_l + margin, enh_l.new_full(enh_l.shape, thr))
    lum_over = F.relu(enh_l - adaptive_ceiling).pow(2).mean()
    channel_over = F.relu(enh - thr).pow(2).mean()
    sat_mask = (enh > 0.985).float()
    hard_clip = sat_mask.mean()
    near_clip = F.relu(enh - 0.97).pow(2).mean()
    return 2.0 * lum_over + channel_over + near_clip + 0.05 * hard_clip


def white_balance_anchor_loss(enhanced: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if int(enhanced.shape[1]) != 3 or int(reference.shape[1]) != 3:
        return enhanced.new_tensor(0.0)
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ref = torch.nan_to_num(reference.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    enh_l = _luma(enh)
    valid = ((enh_l > 0.05) & (enh_l < 0.92)).float()
    denom = valid.sum(dim=(2, 3), keepdim=False).clamp_min(1.0)
    enh_mean = (enh * valid).sum(dim=(2, 3)) / denom
    ref_mean = (ref * valid).sum(dim=(2, 3)) / denom
    enh_ratio = enh_mean / (enh_mean.sum(dim=1, keepdim=True) + 1e-4)
    ref_ratio = ref_mean / (ref_mean.sum(dim=1, keepdim=True) + 1e-4)
    gray = enh_ratio.new_full(enh_ratio.shape, 1.0 / 3.0)
    return F.smooth_l1_loss(enh_ratio, ref_ratio) + 0.2 * F.smooth_l1_loss(enh_ratio, gray)


def adaptive_exposure_loss(
    enhanced: torch.Tensor,
    low_img: torch.Tensor,
    delta: float = 0.24,
    low: float = 0.34,
    high: float = 0.60,
    over_weight: float = 3.0,
    patch_size: int = 16,
) -> torch.Tensor:
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    low_img = torch.nan_to_num(low_img.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    k = int(max(1, patch_size))
    enh_pool = F.avg_pool2d(_luma(enh), kernel_size=k, stride=k)
    low_pool = F.avg_pool2d(_luma(low_img), kernel_size=k, stride=k)
    target = low_pool + float(delta) * (1.0 - low_pool)
    target = target.clamp(float(low), float(max(low + 1e-3, high)))
    under = F.relu(target - enh_pool).pow(2)
    # The allowed ceiling is target-dependent, so bright or already-readable
    # patches are not pushed to a fixed high exposure.
    ceiling = torch.minimum(target + 0.08, enh_pool.new_full(enh_pool.shape, float(high)))
    over = F.relu(enh_pool - ceiling).pow(2)
    clip = F.relu(enh - 0.92).pow(2).mean()
    return under.mean() + float(over_weight) * over.mean() + 0.5 * clip


def green_cast_loss(enhanced: torch.Tensor, reference: torch.Tensor, low_img: torch.Tensor, margin: float = 0.025) -> torch.Tensor:
    if int(enhanced.shape[1]) != 3:
        return enhanced.new_tensor(0.0)
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ref = torch.nan_to_num(reference.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    low = torch.nan_to_num(low_img.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if int(ref.shape[1]) != 3:
        ref = ref.repeat(1, 3, 1, 1)

    low_max = low.amax(dim=1, keepdim=True)
    low_min = low.amin(dim=1, keepdim=True)
    low_sat = (low_max - low_min) / (low_max + 1e-4)
    neutral = (1.0 - low_sat / 0.35).clamp(0.0, 1.0)
    valid_lum = ((_luma(enh) > 0.06) & (_luma(enh) < 0.92)).float()
    weight = neutral * valid_lum

    r, g, b = enh[:, 0:1], enh[:, 1:2], enh[:, 2:3]
    green_excess = F.relu(g - 0.5 * (r + b) - float(max(0.0, margin)))
    local_loss = (green_excess * weight).sum() / (weight.sum() + 1e-6)

    enh_mean = enh.mean(dim=(2, 3))
    ref_mean = ref.mean(dim=(2, 3))
    enh_ratio = enh_mean / (enh_mean.sum(dim=1, keepdim=True) + 1e-4)
    ref_ratio = ref_mean / (ref_mean.sum(dim=1, keepdim=True) + 1e-4)
    green_ratio_excess = F.relu(enh_ratio[:, 1] - ref_ratio[:, 1] - float(max(0.0, margin)))
    gray_excess = F.relu(enh_ratio[:, 1] - 1.0 / 3.0 - float(max(0.0, margin)))
    return local_loss + green_ratio_excess.mean() + 0.5 * gray_excess.mean()


def low_snr_chroma_loss(
    enhanced: torch.Tensor,
    low_img: torch.Tensor,
    luma_threshold: float = 0.08,
    texture_threshold: float = 0.015,
) -> torch.Tensor:
    if int(enhanced.shape[1]) != 3 or int(low_img.shape[1]) != 3:
        return enhanced.new_tensor(0.0)
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    low = torch.nan_to_num(low_img.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    low_l = _luma(low)
    pad = 2
    low_blur = F.avg_pool2d(F.pad(low_l, (pad, pad, pad, pad), mode="reflect"), kernel_size=5, stride=1)
    local_texture = F.avg_pool2d(F.pad((low_l - low_blur).abs(), (pad, pad, pad, pad), mode="reflect"), kernel_size=5, stride=1)
    dark_weight = ((float(luma_threshold) - low_l) / max(float(luma_threshold), 1e-4)).clamp(0.0, 1.0)
    flat_weight = ((float(texture_threshold) - local_texture) / max(float(texture_threshold), 1e-4)).clamp(0.0, 1.0)
    enh_l = _luma(enh)
    valid_enh = ((enh_l > 0.06) & (enh_l < 0.92)).float()
    weight = dark_weight * flat_weight * valid_enh

    gray = enh_l.repeat(1, 3, 1, 1)
    chroma_dev = (enh - gray).abs().mean(dim=1, keepdim=True)
    local_loss = (chroma_dev * weight).sum() / (weight.sum() + 1e-6)

    masked_mean = (enh * weight).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)).clamp_min(1e-6))
    valid_img = (weight.mean(dim=(2, 3)) > 1e-4).float()
    ratio = masked_mean / (masked_mean.sum(dim=1, keepdim=True) + 1e-4)
    gray_ratio = ratio.new_full(ratio.shape, 1.0 / 3.0)
    global_loss = (F.smooth_l1_loss(ratio, gray_ratio, reduction="none").mean(dim=1) * valid_img.view(-1)).sum()
    global_loss = global_loss / (valid_img.sum() + 1e-6)
    return local_loss + 0.5 * global_loss


def scist_restoration_branch_losses(
    out: Dict[str, torch.Tensor],
    low_img: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    enhanced = out.get("enhanced")
    gain_enhanced = out.get("gain_enhanced")
    residual = out.get("restore_residual")
    gate = out.get("restore_gate")
    snr_mask = out.get("snr_mask")
    if enhanced is None or gain_enhanced is None or residual is None or gate is None or snr_mask is None:
        zero = low_img.new_tensor(0.0)
        return zero, zero, zero, zero

    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    gain_ref = torch.nan_to_num(gain_enhanced.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    res = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    g = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    snr = torch.nan_to_num(snr_mask.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    l_reg = (res.abs() * (0.25 + g)).mean()
    l_luma = F.smooth_l1_loss(_luma(enh), _luma(gain_ref))

    high_snr = (1.0 - snr).clamp(0.0, 1.0)
    l_identity = ((enh - gain_ref).abs().mean(dim=1, keepdim=True) * high_snr).sum() / (high_snr.sum() + 1e-6)

    dx = (res[:, :, :, 1:] - res[:, :, :, :-1]).abs()
    dy = (res[:, :, 1:, :] - res[:, :, :-1, :]).abs()
    gx = 0.5 * (g[:, :, :, 1:] + g[:, :, :, :-1])
    gy = 0.5 * (g[:, :, 1:, :] + g[:, :, :-1, :])
    l_tv = (dx * (0.25 + gx)).mean() + (dy * (0.25 + gy)).mean()
    return l_reg, l_luma, l_identity, l_tv


def scist_color_branch_losses(
    out: Dict[str, torch.Tensor],
    low_img: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    color_log = out.get("color_log")
    illum_enhanced = out.get("illum_enhanced")
    gain_enhanced = out.get("gain_enhanced")
    if color_log is None or illum_enhanced is None or gain_enhanced is None:
        zero = low_img.new_tensor(0.0)
        return zero, zero, zero

    c = torch.nan_to_num(color_log, nan=0.0, posinf=0.0, neginf=0.0)
    illum = torch.nan_to_num(illum_enhanced.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    colored = torch.nan_to_num(gain_enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    l_reg = c.abs().mean()
    dx = (c[:, :, :, 1:] - c[:, :, :, :-1]).abs().mean()
    dy = (c[:, :, 1:, :] - c[:, :, :-1, :]).abs().mean()
    l_tv = dx + dy
    l_luma = F.smooth_l1_loss(_luma(colored), _luma(illum))
    return l_reg, l_tv, l_luma


def neutral_color_cast_loss(
    enhanced: torch.Tensor,
    low_img: torch.Tensor,
    margin: float = 0.06,
) -> torch.Tensor:
    if int(enhanced.shape[1]) != 3 or int(low_img.shape[1]) != 3:
        return enhanced.new_tensor(0.0)
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    low = torch.nan_to_num(low_img.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    low_max = low.amax(dim=1, keepdim=True)
    low_min = low.amin(dim=1, keepdim=True)
    low_sat = (low_max - low_min) / (low_max + 1e-4)
    neutral = (1.0 - low_sat / 0.30).clamp(0.0, 1.0)
    enh_l = _luma(enh)
    valid = ((_luma(low) < 0.35) & (enh_l > 0.05) & (enh_l < 0.92)).float()
    weight = neutral * valid

    ratio = enh / (enh.sum(dim=1, keepdim=True) + 1e-4)
    excess = F.relu((ratio - 1.0 / 3.0).abs() - float(max(0.0, margin)))
    local_loss = (excess * weight).sum() / (weight.sum() * 3.0 + 1e-6)

    masked_mean = (enh * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1e-6)
    valid_img = (weight.mean(dim=(2, 3)) > 1e-4).float()
    mean_ratio = masked_mean / (masked_mean.sum(dim=1, keepdim=True) + 1e-4)
    global_excess = F.relu((mean_ratio - 1.0 / 3.0).abs() - 0.5 * float(max(0.0, margin)))
    global_loss = (global_excess.mean(dim=1) * valid_img.view(-1)).sum() / (valid_img.sum() + 1e-6)
    return local_loss + global_loss


def normal_light_stat_loss(
    enhanced: torch.Tensor,
    normal_ref: torch.Tensor,
    color_weight: float = 0.3,
    contrast_weight: float = 0.5,
) -> torch.Tensor:
    enhanced = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    normal_ref = torch.nan_to_num(normal_ref.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if tuple(enhanced.shape[-2:]) != tuple(normal_ref.shape[-2:]):
        normal_ref = F.interpolate(normal_ref, size=enhanced.shape[-2:], mode="bilinear", align_corners=False)
    if int(enhanced.shape[1]) != int(normal_ref.shape[1]):
        if int(enhanced.shape[1]) == 1 and int(normal_ref.shape[1]) == 3:
            normal_ref = _luma(normal_ref)
        elif int(enhanced.shape[1]) == 3 and int(normal_ref.shape[1]) == 1:
            normal_ref = normal_ref.repeat(1, 3, 1, 1)

    enh_l = _luma(enhanced)
    ref_l = _luma(normal_ref)
    enh_mean = enh_l.mean(dim=(2, 3))
    ref_mean = ref_l.mean(dim=(2, 3)).mean(dim=0, keepdim=True).expand_as(enh_mean)
    l_mean = F.smooth_l1_loss(enh_mean, ref_mean)

    enh_std = enh_l.flatten(2).std(dim=2, unbiased=False)
    ref_std = ref_l.flatten(2).std(dim=2, unbiased=False).mean(dim=0, keepdim=True).expand_as(enh_std)
    l_contrast = F.smooth_l1_loss(enh_std, ref_std)

    l_color = enhanced.new_tensor(0.0)
    if int(enhanced.shape[1]) == 3 and int(normal_ref.shape[1]) == 3:
        enh_rgb = enhanced.mean(dim=(2, 3))
        ref_rgb = normal_ref.mean(dim=(2, 3)).mean(dim=0, keepdim=True).expand_as(enh_rgb)
        enh_ratio = enh_rgb / (enh_rgb.sum(dim=1, keepdim=True) + 1e-4)
        ref_ratio = ref_rgb / (ref_rgb.sum(dim=1, keepdim=True) + 1e-4)
        l_color_ratio = F.smooth_l1_loss(enh_ratio, ref_ratio)

        enh_log_chr = torch.log(enhanced.clamp_min(1e-4)) - torch.log(enh_l.clamp_min(1e-4))
        ref_log_chr = torch.log(normal_ref.clamp_min(1e-4)) - torch.log(ref_l.clamp_min(1e-4))
        enh_valid = ((enh_l > 0.05) & (enh_l < 0.92)).float()
        ref_valid = ((ref_l > 0.05) & (ref_l < 0.92)).float()
        enh_den = enh_valid.sum(dim=(2, 3)).clamp_min(1.0)
        ref_den = ref_valid.sum(dim=(2, 3)).clamp_min(1.0)
        enh_chr_mean = (enh_log_chr * enh_valid).sum(dim=(2, 3)) / enh_den
        ref_chr_mean = (ref_log_chr * ref_valid).sum(dim=(2, 3)) / ref_den
        enh_chr_var = ((enh_log_chr - enh_chr_mean[:, :, None, None]).pow(2) * enh_valid).sum(dim=(2, 3)) / enh_den
        ref_chr_var = ((ref_log_chr - ref_chr_mean[:, :, None, None]).pow(2) * ref_valid).sum(dim=(2, 3)) / ref_den
        ref_chr_mean = ref_chr_mean.mean(dim=0, keepdim=True).expand_as(enh_chr_mean)
        ref_chr_std = ref_chr_var.clamp_min(0.0).sqrt().mean(dim=0, keepdim=True).expand_as(enh_chr_mean)
        enh_chr_std = enh_chr_var.clamp_min(0.0).sqrt()
        l_color_chr = F.smooth_l1_loss(enh_chr_mean, ref_chr_mean) + 0.5 * F.smooth_l1_loss(enh_chr_std, ref_chr_std)
        l_color = l_color_ratio + 0.5 * l_color_chr

    return l_mean + float(contrast_weight) * l_contrast + float(color_weight) * l_color


def _local_luma_std(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    l = _luma(x)
    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    mean = F.avg_pool2d(F.pad(l, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
    mean_sq = F.avg_pool2d(F.pad(l * l, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
    return (mean_sq - mean * mean).clamp_min(0.0).sqrt()


def _luma_gradient_strength(x: torch.Tensor) -> torch.Tensor:
    l = _luma(x)
    dx = (l[:, :, :, 1:] - l[:, :, :, :-1]).abs()
    dy = (l[:, :, 1:, :] - l[:, :, :-1, :]).abs()
    return 0.5 * (dx.mean(dim=(1, 2, 3)) + dy.mean(dim=(1, 2, 3)))


def _image_colorfulness(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[1]) != 3:
        return x.new_zeros((int(x.shape[0]),))
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = (rg.flatten(1).std(dim=1, unbiased=False).pow(2) + yb.flatten(1).std(dim=1, unbiased=False).pow(2)).sqrt()
    mean = (rg.flatten(1).mean(dim=1).pow(2) + yb.flatten(1).mean(dim=1).pow(2)).sqrt()
    return std + 0.3 * mean


def normal_light_quality_stat_loss(
    enhanced: torch.Tensor,
    normal_ref: torch.Tensor,
    contrast_weight: float = 1.0,
    sharpness_weight: float = 0.5,
    colorfulness_weight: float = 0.3,
) -> torch.Tensor:
    enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ref = torch.nan_to_num(normal_ref.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if tuple(enh.shape[-2:]) != tuple(ref.shape[-2:]):
        ref = F.interpolate(ref, size=enh.shape[-2:], mode="bilinear", align_corners=False)
    if int(enh.shape[1]) != int(ref.shape[1]):
        if int(enh.shape[1]) == 1 and int(ref.shape[1]) == 3:
            ref = _luma(ref)
        elif int(enh.shape[1]) == 3 and int(ref.shape[1]) == 1:
            ref = ref.repeat(1, 3, 1, 1)

    enh_contrast = _local_luma_std(enh).mean(dim=(1, 2, 3))
    ref_contrast = _local_luma_std(ref).mean(dim=(1, 2, 3)).mean().expand_as(enh_contrast)
    l_contrast = F.smooth_l1_loss(enh_contrast, ref_contrast)

    enh_sharp = _luma_gradient_strength(enh)
    ref_sharp = _luma_gradient_strength(ref).mean().expand_as(enh_sharp)
    l_sharp_under = F.relu(0.95 * ref_sharp - enh_sharp).pow(2).mean()
    l_sharp_over = F.relu(enh_sharp - 1.25 * ref_sharp).pow(2).mean()
    l_sharp = l_sharp_under + 0.25 * l_sharp_over

    l_colorful = enh.new_tensor(0.0)
    if int(enh.shape[1]) == 3 and int(ref.shape[1]) == 3:
        enh_color = _image_colorfulness(enh)
        ref_color = _image_colorfulness(ref).mean().expand_as(enh_color)
        l_color_under = F.relu(0.90 * ref_color - enh_color).pow(2).mean()
        l_color_over = F.relu(enh_color - 1.20 * ref_color).pow(2).mean()
        l_colorful = l_color_under + 0.5 * l_color_over

    return (
        float(contrast_weight) * l_contrast
        + float(sharpness_weight) * l_sharp
        + float(colorfulness_weight) * l_colorful
    )


def scist_target_state(
    low_img: torch.Tensor,
    clip_encoder: nn.Module,
    state_extractor: IlluminationStateExtractor,
    state_normalizer: IlluminationStateNormalizer,
    imf: IlluminationStateIMF,
    residual: torch.Tensor,
    state_min: Optional[torch.Tensor] = None,
    state_max: Optional[torch.Tensor] = None,
    delta_clip: float = 0.0,
) -> torch.Tensor:
    with torch.no_grad():
        state_low = state_normalizer(state_extractor(low_img))
        x = torch.nan_to_num(low_img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        from utils.clip_domain import preprocess_for_clip

        clip_x = preprocess_for_clip(clip_encoder, x).float()
        z = clip_encoder._encode_image_feature(clip_x).float() if hasattr(clip_encoder, "_encode_image_feature") else clip_encoder(x).float()
        q = content_feature_from_clip(z, residual.to(device=low_img.device))
        delta = imf.infer_correction(state_low, q)
        if float(delta_clip) > 0.0:
            delta = delta.clamp(min=-float(delta_clip), max=float(delta_clip))
        target = state_low + delta
        if isinstance(state_min, torch.Tensor) and isinstance(state_max, torch.Tensor):
            target = torch.maximum(torch.minimum(target, state_max.to(device=target.device, dtype=target.dtype)), state_min.to(device=target.device, dtype=target.dtype))
    return target.detach()


def illumination_state_consistency_loss(
    enhanced: torch.Tensor,
    target_state: torch.Tensor,
    state_extractor: IlluminationStateExtractor,
    state_normalizer: IlluminationStateNormalizer,
    lambda_global: float = 1.0,
    lambda_hist: float = 1.0,
    lambda_spatial: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state_e = state_normalizer(state_extractor(enhanced))
    eg, eh, esp = split_state_groups(state_e)
    tg, th, tsp = split_state_groups(target_state.detach())
    l_g = F.smooth_l1_loss(eg, tg)
    l_h = F.smooth_l1_loss(eh, th)
    l_sp = F.smooth_l1_loss(esp, tsp)
    total = float(lambda_global) * l_g + float(lambda_hist) * l_h + float(lambda_spatial) * l_sp
    return total, l_g.detach(), l_h.detach(), l_sp.detach()


def musiq_quality_loss(enhanced: torch.Tensor, device: torch.device, image_size: int = 224) -> torch.Tensor:
    global _MUSIQ_TRAIN_METRIC, _MUSIQ_TRAIN_WARNED
    try:
        import pyiqa  # type: ignore

        if _MUSIQ_TRAIN_METRIC is None:
            _MUSIQ_TRAIN_METRIC = pyiqa.create_metric("musiq", device=device)
            _MUSIQ_TRAIN_METRIC.eval()
            for p in _MUSIQ_TRAIN_METRIC.parameters():
                p.requires_grad = False
        enh = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        size = int(image_size)
        if size > 0 and tuple(enh.shape[-2:]) != (size, size):
            enh = F.interpolate(enh, size=(size, size), mode="bilinear", align_corners=False)
        score = _MUSIQ_TRAIN_METRIC(enh)
        score = score if isinstance(score, torch.Tensor) else enh.new_tensor(float(score))
        return -score.float().mean() / 100.0
    except Exception as exc:
        if not _MUSIQ_TRAIN_WARNED:
            print(f"[Warning] Training MUSIQ loss is disabled because pyiqa MUSIQ failed: {exc}")
            _MUSIQ_TRAIN_WARNED = True
        return enhanced.new_tensor(0.0)


def train_one_epoch(
    epoch: int,
    models: Dict[str, nn.Module],
    anchor_models: Optional[Dict[str, nn.Module]],
    optimizer: optim.Optimizer,
    loader: DataLoader,
    normal_ref_loader: Optional[DataLoader],
    device: torch.device,
    residual: torch.Tensor,
    mean_pos: torch.Tensor,
    args: argparse.Namespace,
    scaler: Optional[torch.amp.GradScaler],
    scist_modules: Optional[Dict[str, nn.Module]] = None,
) -> None:
    model = models["model"]
    clip = models["clip"]
    conditioner = models["conditioner"]
    model.train()
    conditioner.train()
    clip.eval()

    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(loader, desc=f"Train E{epoch}", dynamic_ncols=True) if int(args.progress_bar) else loader
    except Exception:
        iterator = loader

    sums = {
        "total": 0.0,
        "diff": 0.0,
        "dom": 0.0,
        "teacher": 0.0,
        "aux": 0.0,
        "ill": 0.0,
        "exp": 0.0,
        "noise": 0.0,
        "artifact": 0.0,
        "inv": 0.0,
        "recon": 0.0,
        "corr": 0.0,
        "low_lum": 0.0,
        "enh_lum": 0.0,
        "hf_anchor": 0.0,
        "bright_floor": 0.0,
        "color_anchor": 0.0,
        "lf_anchor": 0.0,
        "highlight": 0.0,
        "white_balance": 0.0,
        "adaptive_exp": 0.0,
        "scist_exp": 0.0,
        "scist_hi": 0.0,
        "green_cast": 0.0,
        "neutral_color": 0.0,
        "normal_stat": 0.0,
        "quality_stat": 0.0,
        "state": 0.0,
        "state_g": 0.0,
        "state_h": 0.0,
        "state_sp": 0.0,
        "low_snr_chroma": 0.0,
        "restore_reg": 0.0,
        "restore_luma": 0.0,
        "restore_id": 0.0,
        "restore_tv": 0.0,
        "color_reg": 0.0,
        "color_tv": 0.0,
        "color_luma": 0.0,
        "musiq": 0.0,
        "exposure_eff": 0.0,
    }
    steps = 0
    skipped = 0
    use_amp = bool(int(args.amp)) and device.type == "cuda"
    trainable_params = [p for p in list(model.parameters()) + list(conditioner.parameters()) if p.requires_grad]
    normal_ref_iter = iter(normal_ref_loader) if normal_ref_loader is not None else None

    for batch_idx, batch in enumerate(iterator):
        if int(args.max_train_batches) > 0 and batch_idx >= int(args.max_train_batches):
            break
        low = batch[0] if isinstance(batch, (tuple, list)) else batch
        low = low.to(device)
        target = None
        if str(args.stage) == "synthetic_pretrain" and isinstance(batch, (tuple, list)) and len(batch) > 1:
            target = batch[1].to(device)

        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else torch.amp.autocast(device_type=device.type, enabled=False)
        with autocast_ctx:
            lambda_exposure_eff = float(args.lambda_exposure)
            exposure_mode_eff = "symmetric"
            exposure_low_eff = float(args.real_adapt_exposure_low)
            exposure_high_eff = float(args.real_adapt_exposure_high)
            if str(args.stage) == "real_adapt":
                lambda_exposure_eff *= float(args.real_adapt_exposure_mult)
                exposure_mode_eff = str(args.real_adapt_exposure_mode)
            target_state = None
            if scist_modules is not None and scist_needs_target_state(args):
                target_state = scist_target_state(
                    low,
                    clip,
                    scist_modules["state_extractor"],  # type: ignore[arg-type]
                    scist_modules["state_normalizer"],  # type: ignore[arg-type]
                    scist_modules["imf"],  # type: ignore[arg-type]
                    residual,
                    state_min=scist_modules.get("state_min"),  # type: ignore[arg-type]
                    state_max=scist_modules.get("state_max"),  # type: ignore[arg-type]
                    delta_clip=float(getattr(args, "scist_delta_clip", 0.0)),
                )
            cond_target_state = target_state
            if scist_modules is not None and cond_target_state is None:
                cond_target_state = low.new_zeros((int(low.shape[0]), 128))
            cond = prepare_semantic_domain_condition(
                clip,
                conditioner,
                low,
                residual,
                clip_scale=float(args.clip_scale),
                target_state=cond_target_state,
            )
            cond = apply_scist_condition_mask(cond, args)
            lambda_diffusion_eff = 0.0 if scist_modules is not None else float(args.lambda_diffusion)
            lambda_recon_eff = 0.0 if scist_modules is not None else float(args.lambda_recon)
            lambda_correction_eff = 0.0 if scist_modules is not None else float(args.lambda_correction)
            target_for_model = None if scist_modules is not None else target
            out = model.training_step(
                low_img=low,
                cond=cond,
                lambda_diffusion=lambda_diffusion_eff,
                lambda_teacher_zero_ref=float(args.lambda_teacher_zero_ref),
                lambda_refiner_aux=float(args.lambda_refiner_aux),
                lambda_recon=lambda_recon_eff,
                lambda_correction=lambda_correction_eff,
                lambda_exposure=lambda_exposure_eff,
                exposure_mode=exposure_mode_eff,
                exposure_low=exposure_low_eff,
                exposure_high=exposure_high_eff,
                lambda_smooth=float(args.lambda_illum_smooth),
                lambda_noise=float(args.lambda_noise),
                lambda_artifact=float(args.lambda_artifact),
                artifact_kernel=int(args.artifact_kernel),
                artifact_margin=float(args.artifact_margin),
                lambda_inverse=float(args.lambda_inverse),
                illum_edge_alpha=float(args.illum_edge_alpha),
                target_img=target_for_model,
            )
            if float(args.lambda_domain) > 0.0 or float(args.lambda_domain_anchor) > 0.0:
                l_dom = clip_residual_domain_loss(
                    clip,
                    low,
                    out["enhanced"],
                    residual,
                    mean_pos,
                    anchor_weight=float(args.lambda_domain_anchor),
                    image_grad_clip=float(args.clip_domain_grad_clip),
                )
            else:
                l_dom = low.new_tensor(0.0)
            l_hf_anchor = low.new_tensor(0.0)
            l_bright_floor = low.new_tensor(0.0)
            l_color_anchor = low.new_tensor(0.0)
            l_lf_anchor = low.new_tensor(0.0)
            l_highlight = low.new_tensor(0.0)
            l_white_balance = low.new_tensor(0.0)
            l_adaptive_exp = low.new_tensor(0.0)
            l_scist_exp = low.new_tensor(0.0)
            l_scist_hi = low.new_tensor(0.0)
            l_green_cast = low.new_tensor(0.0)
            l_neutral_color = low.new_tensor(0.0)
            l_normal_stat = low.new_tensor(0.0)
            l_quality_stat = low.new_tensor(0.0)
            l_state = low.new_tensor(0.0)
            l_state_g = low.new_tensor(0.0)
            l_state_h = low.new_tensor(0.0)
            l_state_sp = low.new_tensor(0.0)
            l_low_snr_chroma = low.new_tensor(0.0)
            l_restore_reg = low.new_tensor(0.0)
            l_restore_luma = low.new_tensor(0.0)
            l_restore_id = low.new_tensor(0.0)
            l_restore_tv = low.new_tensor(0.0)
            l_color_reg = low.new_tensor(0.0)
            l_color_tv = low.new_tensor(0.0)
            l_color_luma = low.new_tensor(0.0)
            l_musiq = low.new_tensor(0.0)
            if scist_modules is not None and target_state is not None and float(args.lambda_state) > 0.0:
                l_state, l_state_g, l_state_h, l_state_sp = illumination_state_consistency_loss(
                    out["enhanced"],
                    target_state,
                    scist_modules["state_extractor"],  # type: ignore[arg-type]
                    scist_modules["state_normalizer"],  # type: ignore[arg-type]
                    lambda_global=float(args.lambda_state_global),
                    lambda_hist=float(args.lambda_state_hist),
                    lambda_spatial=float(args.lambda_state_spatial),
                )
            wants_real_anchor = (
                str(args.stage) == "real_adapt"
                and (
                    float(args.lambda_hf_anchor) > 0.0
                    or float(args.lambda_brightness_floor) > 0.0
                    or float(args.lambda_color_anchor) > 0.0
                    or float(args.lambda_lowfreq_anchor) > 0.0
                    or float(args.lambda_highlight) > 0.0
                    or float(args.lambda_white_balance) > 0.0
                    or float(args.lambda_adaptive_exposure) > 0.0
                    or float(args.lambda_green_cast) > 0.0
                )
            )
            if wants_real_anchor:
                anchor_ref = out.get("coarse_enhanced", low).detach()
                if anchor_models is not None:
                    with torch.no_grad():
                        anchor_cond = prepare_semantic_domain_condition(
                            clip,
                            anchor_models["conditioner"],
                            low,
                            residual,
                            clip_scale=float(args.clip_scale),
                            target_state=target_state,
                        )
                        t_anchor = torch.full((int(low.shape[0]), 1), float(max(0, int(args.eval_timestep))), device=device)
                        anchor_ref = anchor_models["model"](low, t_anchor, anchor_cond, return_illum=False)
                        anchor_ref = torch.nan_to_num(anchor_ref, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                enhanced_for_anchor = out["enhanced"]
                if float(args.lambda_hf_anchor) > 0.0:
                    l_hf_anchor = high_frequency_anchor_loss(
                        enhanced_for_anchor,
                        anchor_ref,
                        kernel_size=int(args.hf_anchor_kernel),
                    )
                if float(args.lambda_brightness_floor) > 0.0:
                    l_bright_floor = brightness_floor_loss(
                        enhanced_for_anchor,
                        anchor_ref,
                        low,
                        ratio=float(args.brightness_floor_ratio),
                    )
                if float(args.lambda_color_anchor) > 0.0:
                    l_color_anchor = color_anchor_loss(enhanced_for_anchor, anchor_ref)
                if float(args.lambda_lowfreq_anchor) > 0.0:
                    l_lf_anchor = low_frequency_anchor_loss(
                        enhanced_for_anchor,
                        anchor_ref,
                        kernel_size=int(args.lowfreq_anchor_kernel),
                    )
                if float(args.lambda_highlight) > 0.0:
                    l_highlight = highlight_loss(
                        enhanced_for_anchor,
                        anchor_ref,
                        threshold=float(args.highlight_threshold),
                        margin=float(args.highlight_margin),
                    )
                if float(args.lambda_white_balance) > 0.0:
                    l_white_balance = white_balance_anchor_loss(enhanced_for_anchor, anchor_ref)
                if float(args.lambda_adaptive_exposure) > 0.0:
                    l_adaptive_exp = adaptive_exposure_loss(
                        enhanced_for_anchor,
                        low,
                        delta=float(args.adaptive_exposure_delta),
                        low=float(args.adaptive_exposure_low),
                        high=float(args.adaptive_exposure_high),
                        over_weight=float(args.adaptive_exposure_over_weight),
                    )
                if float(args.lambda_green_cast) > 0.0:
                    l_green_cast = green_cast_loss(
                        enhanced_for_anchor,
                        anchor_ref,
                        low,
                        margin=float(args.green_cast_margin),
                    )
            if str(args.stage) == "real_adapt" and float(args.lambda_low_snr_chroma) > 0.0:
                l_low_snr_chroma = low_snr_chroma_loss(
                    out["enhanced"],
                    low,
                    luma_threshold=float(args.low_snr_luma_threshold),
                    texture_threshold=float(args.low_snr_texture_threshold),
                )
            if scist_modules is not None and float(getattr(args, "scist_lambda_neutral_color", 0.0)) > 0.0:
                l_neutral_color = neutral_color_cast_loss(
                    out["enhanced"],
                    low,
                    margin=float(getattr(args, "scist_neutral_color_margin", 0.06)),
                )
            if scist_modules is not None and bool(int(getattr(args, "scist_restore", 0))):
                l_restore_reg, l_restore_luma, l_restore_id, l_restore_tv = scist_restoration_branch_losses(out, low)
            if scist_modules is not None and bool(int(getattr(args, "scist_color", 0))):
                l_color_reg, l_color_tv, l_color_luma = scist_color_branch_losses(out, low)
            if scist_modules is not None and float(getattr(args, "scist_lambda_exposure_safety", 0.0)) > 0.0:
                l_scist_exp = adaptive_exposure_loss(
                    out["enhanced"],
                    low,
                    delta=float(getattr(args, "scist_exposure_delta", 0.28)),
                    low=float(getattr(args, "scist_exposure_low", 0.36)),
                    high=float(getattr(args, "scist_exposure_high", 0.62)),
                    over_weight=float(getattr(args, "scist_exposure_over_weight", 4.0)),
                )
            if scist_modules is not None and float(getattr(args, "scist_lambda_highlight_safety", 0.0)) > 0.0:
                l_scist_hi = F.relu(out["enhanced"] - 0.92).pow(2).mean() + 0.1 * (out["enhanced"] > 0.985).float().mean()
            wants_normal_ref = (
                str(args.stage) == "real_adapt"
                and normal_ref_iter is not None
                and (float(args.lambda_normal_stat) > 0.0 or float(args.lambda_quality_stat) > 0.0)
            )
            if wants_normal_ref:
                try:
                    ref_batch = next(normal_ref_iter)
                except StopIteration:
                    normal_ref_iter = iter(normal_ref_loader) if normal_ref_loader is not None else None
                    ref_batch = next(normal_ref_iter) if normal_ref_iter is not None else None
                if ref_batch is not None:
                    normal_ref = ref_batch[1] if isinstance(ref_batch, (tuple, list)) and len(ref_batch) > 1 else ref_batch
                    normal_ref = normal_ref.to(device)
                    if float(args.lambda_normal_stat) > 0.0:
                        l_normal_stat = normal_light_stat_loss(
                            out["enhanced"],
                            normal_ref,
                            color_weight=float(args.normal_stat_color_weight),
                            contrast_weight=float(args.normal_stat_contrast_weight),
                        )
                    if float(args.lambda_quality_stat) > 0.0:
                        l_quality_stat = normal_light_quality_stat_loss(
                            out["enhanced"],
                            normal_ref,
                            contrast_weight=float(args.quality_contrast_weight),
                            sharpness_weight=float(args.quality_sharpness_weight),
                            colorfulness_weight=float(args.quality_colorfulness_weight),
                        )
            musiq_every = int(max(1, int(args.musiq_loss_every)))
            if str(args.stage) == "real_adapt" and float(args.lambda_musiq) > 0.0 and (batch_idx % musiq_every == 0):
                l_musiq = musiq_quality_loss(
                    out["enhanced"],
                    device=device,
                    image_size=int(args.musiq_loss_size),
                )
            total = (
                out["loss"]
                + float(args.lambda_domain) * l_dom
                + float(args.lambda_hf_anchor) * l_hf_anchor
                + float(args.lambda_brightness_floor) * l_bright_floor
                + float(args.lambda_color_anchor) * l_color_anchor
                + float(args.lambda_lowfreq_anchor) * l_lf_anchor
                + float(args.lambda_highlight) * l_highlight
                + float(args.lambda_white_balance) * l_white_balance
                + float(args.lambda_adaptive_exposure) * l_adaptive_exp
                + float(getattr(args, "scist_lambda_exposure_safety", 0.0)) * l_scist_exp
                + float(getattr(args, "scist_lambda_highlight_safety", 0.0)) * l_scist_hi
                + float(args.lambda_green_cast) * l_green_cast
                + float(getattr(args, "scist_lambda_neutral_color", 0.0)) * l_neutral_color
                + float(getattr(args, "scist_lambda_restore_reg", 0.0)) * l_restore_reg
                + float(getattr(args, "scist_lambda_restore_luma", 0.0)) * l_restore_luma
                + float(getattr(args, "scist_lambda_restore_identity", 0.0)) * l_restore_id
                + float(getattr(args, "scist_lambda_restore_tv", 0.0)) * l_restore_tv
                + float(getattr(args, "scist_lambda_color_reg", 0.0)) * l_color_reg
                + float(getattr(args, "scist_lambda_color_tv", 0.0)) * l_color_tv
                + float(getattr(args, "scist_lambda_color_luma", 0.0)) * l_color_luma
                + float(args.lambda_normal_stat) * l_normal_stat
                + float(args.lambda_quality_stat) * l_quality_stat
                + float(args.lambda_state) * l_state
                + float(args.lambda_low_snr_chroma) * l_low_snr_chroma
                + float(args.lambda_musiq) * l_musiq
            )

        if not torch.isfinite(total).all():
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            print(f"[Train][IllumDiff] Skip non-finite loss at epoch={epoch} step={batch_idx + 1}")
            continue

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            bad_grad = False
            max_abs_grad = 0.0
            bad_grad_name = ""
            for name, p in list(model.named_parameters()) + [(f"conditioner.{n}", p) for n, p in conditioner.named_parameters()]:
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    bad_grad = True
                    bad_grad_name = name
                    break
                max_abs_grad = max(max_abs_grad, float(p.grad.detach().abs().max().item()))
            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                print(
                    f"[Train][IllumDiff] Skip non-finite gradient at epoch={epoch} "
                    f"step={batch_idx + 1} param={bad_grad_name or 'unknown'}"
                )
                scaler.update()
                continue
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            bad_grad = False
            max_abs_grad = 0.0
            bad_grad_name = ""
            for name, p in list(model.named_parameters()) + [(f"conditioner.{n}", p) for n, p in conditioner.named_parameters()]:
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    bad_grad = True
                    bad_grad_name = name
                    break
                max_abs_grad = max(max_abs_grad, float(p.grad.detach().abs().max().item()))
            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                print(
                    f"[Train][IllumDiff] Skip non-finite gradient at epoch={epoch} "
                    f"step={batch_idx + 1} param={bad_grad_name or 'unknown'}"
                )
                continue
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip_norm))
            optimizer.step()

        steps += 1
        sums["total"] += float(total.detach().float().item())
        sums["diff"] += float(out["l_diff"].detach().float().item())
        sums["dom"] += float(l_dom.detach().float().item())
        sums["teacher"] += float(out["l_teacher"].detach().float().item())
        sums["aux"] += float(out["l_refiner_aux"].detach().float().item())
        sums["ill"] += float(out["l_ill"].detach().float().item())
        sums["exp"] += float(out["l_exp"].detach().float().item())
        sums["noise"] += float(out["l_noise"].detach().float().item())
        sums["artifact"] += float(out.get("l_artifact", torch.tensor(0.0, device=device)).detach().float().item())
        sums["inv"] += float(out["l_inv"].detach().float().item())
        sums["recon"] += float(out.get("l_recon", torch.tensor(0.0, device=device)).detach().float().item())
        sums["corr"] += float(out.get("l_target_illum", torch.tensor(0.0, device=device)).detach().float().item())
        sums["hf_anchor"] += float(l_hf_anchor.detach().float().item())
        sums["bright_floor"] += float(l_bright_floor.detach().float().item())
        sums["color_anchor"] += float(l_color_anchor.detach().float().item())
        sums["lf_anchor"] += float(l_lf_anchor.detach().float().item())
        sums["highlight"] += float(l_highlight.detach().float().item())
        sums["white_balance"] += float(l_white_balance.detach().float().item())
        sums["adaptive_exp"] += float(l_adaptive_exp.detach().float().item())
        sums["scist_exp"] += float(l_scist_exp.detach().float().item())
        sums["scist_hi"] += float(l_scist_hi.detach().float().item())
        sums["green_cast"] += float(l_green_cast.detach().float().item())
        sums["neutral_color"] += float(l_neutral_color.detach().float().item())
        sums["normal_stat"] += float(l_normal_stat.detach().float().item())
        sums["quality_stat"] += float(l_quality_stat.detach().float().item())
        sums["state"] += float(l_state.detach().float().item())
        sums["state_g"] += float(l_state_g.detach().float().item())
        sums["state_h"] += float(l_state_h.detach().float().item())
        sums["state_sp"] += float(l_state_sp.detach().float().item())
        sums["low_snr_chroma"] += float(l_low_snr_chroma.detach().float().item())
        sums["restore_reg"] += float(l_restore_reg.detach().float().item())
        sums["restore_luma"] += float(l_restore_luma.detach().float().item())
        sums["restore_id"] += float(l_restore_id.detach().float().item())
        sums["restore_tv"] += float(l_restore_tv.detach().float().item())
        sums["color_reg"] += float(l_color_reg.detach().float().item())
        sums["color_tv"] += float(l_color_tv.detach().float().item())
        sums["color_luma"] += float(l_color_luma.detach().float().item())
        sums["musiq"] += float(l_musiq.detach().float().item())
        sums["exposure_eff"] += float(lambda_exposure_eff)
        with torch.no_grad():
            low_lum = 0.299 * low[:, 0:1] + 0.587 * low[:, 1:2] + 0.114 * low[:, 2:3] if int(low.shape[1]) == 3 else low.mean(dim=1, keepdim=True)
            enh = out["enhanced"].detach()
            enh_lum = 0.299 * enh[:, 0:1] + 0.587 * enh[:, 1:2] + 0.114 * enh[:, 2:3] if int(enh.shape[1]) == 3 else enh.mean(dim=1, keepdim=True)
            sums["low_lum"] += float(low_lum.mean().detach().float().item())
            sums["enh_lum"] += float(enh_lum.mean().detach().float().item())

        if int(args.log_gain_every) > 0 and (batch_idx == 0 or (batch_idx + 1) % int(args.log_gain_every) == 0):
            gain = out.get("gain", None)
            if isinstance(gain, torch.Tensor):
                g = gain.detach().float()
                print(
                    f"[Gain][IllumDiff] E{epoch} step{batch_idx + 1} "
                    f"gain[min,mean,max]=[{float(g.amin()):.3f},{float(g.mean()):.3f},{float(g.amax()):.3f}]"
                )

        if hasattr(iterator, "set_postfix"):
            try:
                iterator.set_postfix(
                    {
                        "tot": f"{float(total.detach()):.4f}",
                        "diff": f"{float(out['l_diff']):.4f}",
                        "dom": f"{float(l_dom.detach()):.4f}",
                        "aux": f"{float(out['l_refiner_aux']):.4f}",
                        "hf": f"{float(l_hf_anchor.detach()):.4f}",
                        "floor": f"{float(l_bright_floor.detach()):.4f}",
                        "hi": f"{float(l_highlight.detach()):.4f}",
                        "aexp": f"{float(l_adaptive_exp.detach()):.4f}",
                        "sexp": f"{float(l_scist_exp.detach()):.4f}",
                        "shi": f"{float(l_scist_hi.detach()):.4f}",
                        "green": f"{float(l_green_cast.detach()):.4f}",
                        "ncol": f"{float(l_neutral_color.detach()):.4f}",
                        "lsnr": f"{float(l_low_snr_chroma.detach()):.4f}",
                        "rreg": f"{float(l_restore_reg.detach()):.4f}",
                        "rlum": f"{float(l_restore_luma.detach()):.4f}",
                        "rid": f"{float(l_restore_id.detach()):.4f}",
                        "creg": f"{float(l_color_reg.detach()):.4f}",
                        "ctv": f"{float(l_color_tv.detach()):.4f}",
                        "clum": f"{float(l_color_luma.detach()):.4f}",
                        "musiq": f"{float(l_musiq.detach()):.4f}",
                        "qstat": f"{float(l_quality_stat.detach()):.4f}",
                        "state": f"{float(l_state.detach()):.4f}",
                        "stat": f"{float(l_normal_stat.detach()):.4f}",
                        "art": f"{float(out.get('l_artifact', torch.tensor(0.0, device=device)).detach()):.4f}",
                    }
                )
            except Exception:
                pass

    if steps == 0:
        print(f"[Train][IllumDiff] Epoch {epoch} produced 0 steps; skipped={skipped}")
        return

    print(
        f"[Train][IllumDiff] Epoch {epoch} "
        f"w(diff/dom/smooth/inv/teacher/aux)=("
        f"{float(args.lambda_diffusion):.3g}/{float(args.lambda_domain):.3g}/"
        f"{float(args.lambda_illum_smooth):.3g}/{float(args.lambda_inverse):.3g}/"
        f"{float(args.lambda_teacher_zero_ref):.3g}/{float(args.lambda_refiner_aux):.3g}) "
        f"mean total={sums['total']/steps:.4f} diff={sums['diff']/steps:.4f} "
        f"dom={sums['dom']/steps:.4f} teacher={sums['teacher']/steps:.4f} "
        f"aux={sums['aux']/steps:.4f} ill={sums['ill']/steps:.4f} "
        f"exp={sums['exp']/steps:.4f} noise={sums['noise']/steps:.4f} art={sums['artifact']/steps:.4f} "
        f"inv={sums['inv']/steps:.4f} recon={sums['recon']/steps:.4f} corr={sums['corr']/steps:.4f} "
        f"hf={sums['hf_anchor']/steps:.4f} floor={sums['bright_floor']/steps:.4f} "
        f"color={sums['color_anchor']/steps:.4f} lf={sums['lf_anchor']/steps:.4f} "
        f"hi={sums['highlight']/steps:.4f} wb={sums['white_balance']/steps:.4f} "
        f"aexp={sums['adaptive_exp']/steps:.4f} sexp={sums['scist_exp']/steps:.4f} shi={sums['scist_hi']/steps:.4f} "
        f"green={sums['green_cast']/steps:.4f} "
        f"ncol={sums['neutral_color']/steps:.4f} "
        f"lsnr={sums['low_snr_chroma']/steps:.4f} "
        f"restore(reg/lum/id/tv)={sums['restore_reg']/steps:.4f}/{sums['restore_luma']/steps:.4f}/"
        f"{sums['restore_id']/steps:.4f}/{sums['restore_tv']/steps:.4f} "
        f"color(reg/tv/lum)={sums['color_reg']/steps:.4f}/{sums['color_tv']/steps:.4f}/{sums['color_luma']/steps:.4f} "
        f"musiq={sums['musiq']/steps:.4f} "
        f"nstat={sums['normal_stat']/steps:.4f} qstat={sums['quality_stat']/steps:.4f} "
        f"state={sums['state']/steps:.4f} sg/h/sp={sums['state_g']/steps:.4f}/{sums['state_h']/steps:.4f}/{sums['state_sp']/steps:.4f} "
        f"exp_eff={sums['exposure_eff']/steps:.3g} "
        f"lum(low/enh)={sums['low_lum']/steps:.4f}/{sums['enh_lum']/steps:.4f}"
    )


def _fmt_metric(v: Optional[float], decimals: int) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{float(v):.{int(decimals)}f}"


def _state_dict(module: nn.Module):
    return module.module.state_dict() if hasattr(module, "module") else module.state_dict()


def save_checkpoint_and_samples(
    *,
    tag: str,
    epoch: int,
    args: argparse.Namespace,
    models: Dict[str, nn.Module],
    optimizer: optim.Optimizer,
    eval_loader: Optional[DataLoader],
    device: torch.device,
    residual: torch.Tensor,
    mean_pos: torch.Tensor,
    psnr: Optional[float],
    ssim: Optional[float],
    niqe: Optional[float],
    musiq: Optional[float],
    decimals: int,
    scist_modules: Optional[Dict[str, nn.Module]] = None,
) -> str:
    folder = (
        f"epoch{epoch:03d}_{tag}_PSNR{_fmt_metric(psnr, decimals)}_SSIM{_fmt_metric(ssim, decimals)}_"
        f"NIQE{_fmt_metric(niqe, decimals)}_MUSIQ{_fmt_metric(musiq, decimals)}"
    )
    epoch_dir = os.path.join(args.save_dir, folder)
    save_illum_samples(
        models,
        eval_loader,
        device,
        residual=residual,
        save_dir=epoch_dir,
        epoch=epoch,
        eval_timestep=int(args.eval_timestep),
        clip_scale=float(args.clip_scale),
        max_save=4,
        seed=int(args.save_random_seed),
        save_gain_map=bool(int(args.save_gain_map)),
        scist_modules=scist_modules,
        scist_ablation=scist_ablation_name(args),
    )
    ckpt_path = os.path.join(epoch_dir, f"illum_diff_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "save_tag": str(tag),
            "stage": str(args.stage),
            "objective": "normal_light_correction_diffusion_delta_A",
            "model_state": _state_dict(models["model"]),
            "conditioner_state": _state_dict(models["conditioner"]),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "residual": residual.detach().cpu(),
            "mean_pos": mean_pos.detach().cpu(),
            "scist": bool(scist_modules is not None),
        },
        ckpt_path,
    )
    return ckpt_path


def main() -> None:
    args = parse_args()
    set_random_seed(int(args.seed))
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        shutil.copy2(__file__, os.path.join(args.save_dir, "train_illum.py"))
    except Exception:
        pass

    device = select_device(args)
    print(f"Using device: {device}")
    print(f"Using training stage: {args.stage}")
    print(f"Using normal-light root: {args.high_root}")
    print(f"Using real low-light root: {args.low_root}")
    print(f"Using eval-only GT root: {args.gt_root}")

    if str(args.stage) == "synthetic_pretrain":
        train_dataset = SyntheticLowLightDataset(high_root=args.high_root)
        print(
            "[Data] synthetic_pretrain training input is high_root only: "
            "I_n from DIV2K_384 is degraded online to I_s; LOL target is not used for training."
        )
    else:
        train_dataset = LowLightOnlyDataset(low_root=args.low_root)
        print(
            "[Data] real_adapt training input is low_root only: "
            "LOLv1/Train/input is used without paired GT; LOL target is not used for training."
        )
    train_loader = build_loader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor),
    )
    print(
        f"[DataLoader] train batch_size={int(args.batch_size)} workers={int(args.num_workers)} "
        f"pin_memory={bool(args.pin_memory)} persistent_workers={bool(args.persistent_workers and args.num_workers > 0)}"
    )

    eval_loader = None
    if args.gt_root and os.path.isdir(args.gt_root):
        eval_dataset = LowLightEvalDataset(low_root=args.low_root, gt_root=args.gt_root)
        eval_loader = build_loader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(args.eval_num_workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
        )
        print(f"Using eval GT root: {args.gt_root} (metrics only: PSNR/SSIM/NIQE/MUSIQ logging and sample panels)")
        print(f"[DataLoader] eval batch_size=1 workers={int(args.eval_num_workers)} pin_memory={bool(args.pin_memory)}")

    normal_ref_loader = None
    if str(args.stage) == "real_adapt" and (
        float(args.lambda_normal_stat) > 0.0
        or float(args.lambda_quality_stat) > 0.0
        or (int(getattr(args, "scist", 0)) > 0 and float(getattr(args, "scist_lambda_normal_stat", 0.0)) > 0.0)
        or (int(getattr(args, "scist", 0)) > 0 and float(getattr(args, "scist_lambda_quality_stat", 0.0)) > 0.0)
        or float(args.ra_final_quality_stat) > 0.0
        or int(args.real_adapt_curriculum) > 0
    ):
        normal_ref_dataset = SyntheticLowLightDataset(high_root=args.high_root)
        normal_ref_loader = build_loader(
            normal_ref_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
        )
        print(
            f"[NormalStat] Enabled unpaired normal-light statistics from {args.high_root}; "
            "LOLv1 target is still metrics-only."
        )

    sample_low, _ = next(iter(train_loader))
    in_channels = int(sample_low.shape[1])
    models = build_models(args, device=device, in_channels=in_channels)
    start_epoch = load_resume_if_needed(args, models, device=device)
    scist_modules = load_scist_modules(args, models, device=device)
    anchor_models = None
    wants_anchor_model = (
        str(args.stage) == "real_adapt"
        and scist_modules is None
        and (
            float(args.lambda_hf_anchor) > 0.0
            or float(args.lambda_brightness_floor) > 0.0
            or float(args.lambda_color_anchor) > 0.0
            or float(args.lambda_lowfreq_anchor) > 0.0
            or float(args.lambda_highlight) > 0.0
            or float(args.lambda_white_balance) > 0.0
            or float(args.lambda_adaptive_exposure) > 0.0
            or float(args.lambda_green_cast) > 0.0
            or int(args.real_adapt_curriculum) > 0
        )
    )
    if wants_anchor_model:
        if not str(args.resume):
            print("[Anchor] Real-adapt anchor losses are enabled but --resume is empty; using coarse enhancement as fallback.")
        else:
            anchor_models = build_frozen_anchor(models, device=device)
    if int(args.freeze_teacher) > 0:
        set_teacher_trainable(models["model"], trainable=False)
    else:
        set_teacher_trainable(models["model"], trainable=True)

    pos_root = str(args.residual_pos_root or args.high_root)
    neg_root = str(args.residual_neg_root or args.low_root)
    residual, mean_pos, mean_neg = load_or_compute_residual_domain(
        models["clip"],
        ResidualDomainConfig(
            pos_root=pos_root,
            neg_root=neg_root,
            max_images=int(args.residual_max_images),
            cache_dir=str(args.residual_cache_dir),
            force_recompute=bool(args.residual_recompute),
            show_progress=bool(int(args.progress_bar)),
            remove_content_bias=scist_uses_bias_suppression(args),
        ),
        device=device,
    )
    _ = mean_neg
    if scist_modules is not None:
        print("[Prior] Using SCIST condition c=[Wd r, Ws z_l, Wi s_target].")
        print(f"[SCIST-Ablation] variant={scist_ablation_name(args)} condition_mask(p_d,p_s,p_i)={scist_condition_mask(args)}")
        print(f"[SCIST-Ablation] content_bias_suppression={scist_uses_bias_suppression(args)}")
        args.lambda_diffusion = 0.0
        args.lambda_recon = 0.0
        args.lambda_correction = 0.0
        args.lambda_teacher_zero_ref = 0.0
        args.lambda_refiner_aux = 0.0
        args.lambda_inverse = 0.0
        args.lambda_noise = 0.0
        args.lambda_artifact = 0.0
        args.lambda_hf_anchor = 0.0
        args.lambda_brightness_floor = 0.0
        args.lambda_color_anchor = 0.0
        args.lambda_lowfreq_anchor = 0.0
        args.lambda_highlight = 0.0
        args.lambda_white_balance = 0.0
        args.lambda_adaptive_exposure = 0.0
        args.lambda_green_cast = 0.0
        args.lambda_normal_stat = float(max(float(args.lambda_normal_stat), float(args.scist_lambda_normal_stat)))
        args.lambda_quality_stat = float(max(float(args.lambda_quality_stat), float(args.scist_lambda_quality_stat)))
        args.lambda_low_snr_chroma = float(max(float(args.lambda_low_snr_chroma), float(args.scist_lambda_low_snr_chroma)))
        args.lambda_musiq = 0.0
    else:
        print("[Prior] Using CLIP semantic-domain condition c=[Wd r, Ws z_l].")

    params = [p for p in list(models["model"].parameters()) + list(models["conditioner"].parameters()) if p.requires_grad]
    optimizer = optim.Adam(params, lr=float(args.lr), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=True) if bool(int(args.amp)) and device.type == "cuda" else None

    best_psnr = float("-inf")
    best_ssim = float("-inf")
    best_niqe = float("inf")
    best_musiq = float("-inf")
    decimals = int(max(2, min(10, int(args.metric_decimals))))
    last_checkpoint_save_epoch = 0
    if str(args.stage) == "real_adapt" and int(args.real_adapt_curriculum) > 0:
        lengths = _real_adapt_curriculum_lengths(args)
        print(
            "[RA-Curriculum] enabled "
            f"lengths={lengths} "
            "phases=illumination->color->artifact->stabilize; "
            "GT remains metrics-only."
        )

    for epoch in range(int(start_epoch) + 1, int(start_epoch) + int(args.epochs) + 1):
        start = time.perf_counter()
        rel_epoch = epoch - int(start_epoch)
        epoch_args, ra_phase, ra_lengths = build_real_adapt_curriculum_args(args, rel_epoch)
        if scist_modules is not None:
            epoch_args = apply_scist_epoch_schedule(epoch_args, rel_epoch)
            epoch_args = apply_scist_ablation_to_args(epoch_args)
        objective = (
            "SCIST: noise-to-correction iMF target-state + CLIP direction + edge-aware log-gain smoothness"
            if scist_modules is not None
            else "Delta_A correction diffusion + CLIP residual main + zero-reference auxiliaries"
        )
        print(
            f"[Stage] Epoch {epoch} -> {args.stage} "
            f"objective={objective}"
        )
        if scist_modules is not None:
            print(
                f"[SCIST-Schedule] rel_epoch={rel_epoch} "
                f"lambda_domain={float(epoch_args.lambda_domain):.4g} "
                f"lambda_state={float(epoch_args.lambda_state):.4g} "
                f"lambda_illum_smooth={float(epoch_args.lambda_illum_smooth):.4g}"
            )
        if str(args.stage) == "real_adapt" and int(args.real_adapt_curriculum) > 0 and int(args.ra_curriculum_verbose) > 0:
            print(
                f"[RA-Curriculum] epoch={epoch} rel={rel_epoch}/{int(args.epochs)} phase={ra_phase} lengths={ra_lengths} "
                f"cmul={float(getattr(epoch_args, 'ra_color_weight_mult', 1.0)):.3g} "
                f"w(dom/refaux/exp/aexp/color/wb/green/lsnr/mq/qstat/noise/art/hf/hi/nstat)="
                f"({float(epoch_args.lambda_domain):.3g}/{float(epoch_args.lambda_refiner_aux):.3g}/"
                f"{float(epoch_args.lambda_exposure) * float(epoch_args.real_adapt_exposure_mult):.3g}/"
                f"{float(epoch_args.lambda_adaptive_exposure):.3g}/{float(epoch_args.lambda_color_anchor):.3g}/"
                f"{float(epoch_args.lambda_white_balance):.3g}/{float(epoch_args.lambda_green_cast):.3g}/"
                f"{float(epoch_args.lambda_low_snr_chroma):.3g}/{float(epoch_args.lambda_musiq):.3g}/"
                f"{float(epoch_args.lambda_quality_stat):.3g}/"
                f"{float(epoch_args.lambda_noise):.3g}/{float(epoch_args.lambda_artifact):.3g}/"
                f"{float(epoch_args.lambda_hf_anchor):.3g}/{float(epoch_args.lambda_highlight):.3g}/"
                f"{float(epoch_args.lambda_normal_stat):.3g})"
            )
        train_one_epoch(
            epoch,
            models,
            anchor_models,
            optimizer,
            train_loader,
            normal_ref_loader,
            device,
            residual,
            mean_pos,
            epoch_args,
            scaler,
            scist_modules=scist_modules,
        )
        train_sec = time.perf_counter() - start

        eval_sec = 0.0
        save_sec = 0.0
        do_eval = eval_loader is not None and (int(args.eval_interval) <= 1 or epoch % int(args.eval_interval) == 0)
        if do_eval:
            eval_start = time.perf_counter()
            psnr, ssim, niqe, musiq = evaluate_illum_model(
                models,
                eval_loader,
                device,
                residual=residual,
                eval_timestep=int(args.eval_timestep),
                clip_scale=float(args.clip_scale),
                max_eval_batches=int(args.max_eval_batches),
                skip_no_ref_metrics=int(args.skip_no_ref_metrics),
                compare_no_ref_baselines=int(args.compare_no_ref_baselines),
                scist_modules=scist_modules,
                epoch=epoch,
                scist_seed=int(args.save_random_seed),
                scist_ablation=scist_ablation_name(epoch_args),
            )
            eval_sec = time.perf_counter() - eval_start
            print(
                f"[Eval] Epoch {epoch}: PSNR={_fmt_metric(psnr, decimals)} "
                f"SSIM={_fmt_metric(ssim, decimals)} NIQE={_fmt_metric(niqe, decimals)} MUSIQ={_fmt_metric(musiq, decimals)}"
            )

            improved_metrics = []
            if epoch >= int(args.best_from_epoch):
                if psnr is not None and round(float(psnr), decimals) > best_psnr:
                    best_psnr = round(float(psnr), decimals)
                    improved_metrics.append("PSNR")
                if ssim is not None and round(float(ssim), decimals) > best_ssim:
                    best_ssim = round(float(ssim), decimals)
                    improved_metrics.append("SSIM")
                if int(args.use_niqe_for_best) > 0 and niqe is not None and round(float(niqe), decimals) < best_niqe:
                    best_niqe = round(float(niqe), decimals)
                    improved_metrics.append("NIQE")
                if musiq is not None and round(float(musiq), decimals) > best_musiq:
                    best_musiq = round(float(musiq), decimals)
                    improved_metrics.append("MUSIQ")

            improved = len(improved_metrics) > 0
            force_interval = int(getattr(args, "force_save_interval", 5) or 0)
            force_save = (not improved) and force_interval > 0 and (epoch - int(last_checkpoint_save_epoch)) >= force_interval

            if improved or force_save:
                save_start = time.perf_counter()
                tag = "best" if improved else "periodic"
                ckpt_path = save_checkpoint_and_samples(
                    tag=tag,
                    epoch=epoch,
                    args=args,
                    models=models,
                    optimizer=optimizer,
                    eval_loader=eval_loader,
                    device=device,
                    residual=residual,
                    mean_pos=mean_pos,
                    psnr=psnr,
                    ssim=ssim,
                    niqe=niqe,
                    musiq=musiq,
                    decimals=decimals,
                    scist_modules=scist_modules,
                )
                last_checkpoint_save_epoch = epoch
                if improved:
                    print(f"[Best] Saved checkpoint and samples to {ckpt_path} (improved={','.join(improved_metrics)})")
                else:
                    print(f"[Periodic] Saved checkpoint and samples to {ckpt_path} (no checkpoint for {force_interval} epochs)")
                save_sec = time.perf_counter() - save_start
            elif int(args.save_latest_every) > 0 and epoch % int(args.save_latest_every) == 0:
                save_start = time.perf_counter()
                latest_dir = os.path.join(args.save_dir, "latest_samples", f"epoch{epoch:03d}")
                save_illum_samples(
                    models,
                    eval_loader,
                    device,
                    residual=residual,
                    save_dir=latest_dir,
                    epoch=epoch,
                    eval_timestep=int(args.eval_timestep),
                    clip_scale=float(args.clip_scale),
                    max_save=4,
                    seed=int(args.save_random_seed),
                    save_gain_map=bool(int(args.save_gain_map)),
                    scist_modules=scist_modules,
                    scist_ablation=scist_ablation_name(epoch_args),
                )
                print(f"[Latest] Saved non-best samples to {latest_dir}")
                save_sec = time.perf_counter() - save_start

        print(f"[EpochTime] epoch={epoch} train={train_sec:.1f}s eval={eval_sec:.1f}s save={save_sec:.1f}s")


if __name__ == "__main__":
    main()
