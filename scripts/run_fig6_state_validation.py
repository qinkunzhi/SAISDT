#!/usr/bin/env python3
"""Run the three fixed illumination-state diagnostics used by paper Fig. 6.

The script does not train SCIST.  It evaluates the deterministic 128-D state
extractor, exports five spatial-map candidates, measures controlled response,
and runs the exposure probe plus state/content linear CKA diagnostic.
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig6")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.classifier import build_classifier
from models.illumination_state import (
    IlluminationStateExtractor,
    IlluminationStateNormalizer,
    _separable_gaussian_blur,
)
from utils.clip_domain import preprocess_for_clip
from utils.semantic_ot import content_feature_from_clip


# Keep text editable/searchable in PDF editors instead of emitting Type-3 glyphs.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPOSURE_MAGNITUDES = (0.25, 0.5, 0.7, 0.9)
GAMMA_DEVIATIONS = (0.1, 0.18, 0.27, 0.34)
NOISE_LEVELS = (0.025, 0.03, 0.04, 0.06)
SHARPEN_LEVELS = (0.55, 0.9, 1.35, 1.8)
PROBE_EXPOSURES = (-1.5, -0.75, 0.0, 0.75, 1.5)
LOG_DISPLAY_MIN = 0.5
LOG_DISPLAY_MAX = 1.0
MAP_DISPLAY_MIN = 0.4
MAP_DISPLAY_MAX = 1.0


@dataclass
class Candidate:
    scene: str
    source: str
    path: str
    mean_log_luma: float
    layout_std: float
    backlit_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCIST Fig. 6 illumination-state validation")
    parser.add_argument("--out_dir", default="ablation_main/fig_6")
    parser.add_argument("--lol_root", default="/home/zhiqinkun/LLIM/datasets/data/LOLv1/Test/input")
    parser.add_argument("--backlit_root", default="/home/zhiqinkun/LLIM/datasets/data/Backlit300")
    parser.add_argument("--response_root", default="/home/zhiqinkun/LLIM/datasets/data/BAID_380/resize_gt")
    parser.add_argument("--priors", default="scist_priors.pt")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--candidate_count", type=int, default=5)
    parser.add_argument("--selected_count", type=int, default=3)
    parser.add_argument("--max_response_images", type=int, default=0)
    parser.add_argument("--probe_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--render_only", action="store_true", help="Regenerate Fig. 6(a)/combined figures from saved results without rerunning B/C.")
    return parser.parse_args()


def list_images(root: str) -> List[str]:
    paths = [str(p) for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def image_id(path: str) -> str:
    return Path(path).stem


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def image_tensor(path: str, size: int, device: torch.device) -> torch.Tensor:
    image = load_rgb(path).resize((int(size), int(size)), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def native_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load an image without resizing, cropping, or changing its aspect ratio."""
    array = np.asarray(load_rgb(path), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def load_batch(paths: Sequence[str], size: int, device: torch.device) -> torch.Tensor:
    tensors = [image_tensor(path, size, torch.device("cpu"))[0] for path in paths]
    return torch.stack(tensors, dim=0).to(device)


def normalized_log_luminance(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    log_y = torch.log(y + float(eps))
    lo = math.log(float(eps))
    hi = math.log(1.0 + float(eps))
    return ((log_y - lo) / (hi - lo)).clamp(0.0, 1.0)


def dhash(path: str, size: int = 16) -> int:
    image = load_rgb(path).convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    array = np.asarray(image)
    bits = array[:, 1:] > array[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    # Python 3.8 compatibility (int.bit_count was added later).
    return int(bin(a ^ b).count("1"))


@torch.inference_mode()
def score_map_sources(
    lol_paths: Sequence[str],
    backlit_paths: Sequence[str],
    extractor: IlluminationStateExtractor,
    size: int,
    device: torch.device,
) -> List[Candidate]:
    scored: List[Candidate] = []
    for source, paths in (("LOLv1-Test", lol_paths), ("Backlit300", backlit_paths)):
        for index, path in enumerate(paths):
            x = image_tensor(path, size, device)
            raw = extractor(x).float()
            ln = normalized_log_luminance(x, extractor.eps)
            h, w = ln.shape[-2:]
            center = ln[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
            border_mask = torch.ones_like(ln, dtype=torch.bool)
            border_mask[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = False
            border = ln[border_mask].mean()
            scored.append(
                Candidate(
                    scene="",
                    source=source,
                    path=path,
                    mean_log_luma=float(raw[0, 0].item()),
                    layout_std=float(raw[0, 64:128].std(unbiased=False).item()),
                    backlit_score=float((border - center).item()),
                )
            )
            if (index + 1) % 50 == 0 or index + 1 == len(paths):
                print(f"[Fig6-A] scored {source}: {index + 1}/{len(paths)}")
    return scored


def select_candidates(scored: Sequence[Candidate], count: int) -> List[Candidate]:
    lol = [item for item in scored if item.source == "LOLv1-Test"]
    backlit = [item for item in scored if item.source == "Backlit300"]
    pools: List[Tuple[str, List[Candidate]]] = [
        ("Overall dark", sorted(lol, key=lambda x: x.mean_log_luma)),
        ("Backlit", sorted(backlit, key=lambda x: x.backlit_score, reverse=True)),
        ("Spatially non-uniform", sorted(scored, key=lambda x: x.layout_std, reverse=True)),
        ("Overall dark (extra)", sorted(lol, key=lambda x: (x.mean_log_luma, -x.layout_std))),
        ("Backlit (extra)", sorted(backlit, key=lambda x: (x.backlit_score, x.layout_std), reverse=True)),
    ]
    selected: List[Candidate] = []
    hashes: List[int] = []
    for label, pool in pools:
        choice = None
        for item in pool:
            if any(existing.path == item.path for existing in selected):
                continue
            candidate_hash = dhash(item.path)
            if hashes and min(hamming(candidate_hash, old) for old in hashes) < 20:
                continue
            choice = item
            hashes.append(candidate_hash)
            break
        if choice is None:
            choice = next(item for item in pool if all(existing.path != item.path for existing in selected))
            hashes.append(dhash(choice.path))
        choice.scene = label
        selected.append(choice)
        if len(selected) >= int(count):
            break
    return selected


@torch.inference_mode()
def candidate_arrays(
    candidate: Candidate,
    extractor: IlluminationStateExtractor,
    size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = load_rgb(candidate.path)
    display = np.asarray(image, dtype=np.float32) / 255.0
    # Fig. 6(a) is a spatial correspondence diagnostic.  Use the native image
    # grid so Input, log-luminance, P4, and P8 refer to exactly the same field
    # of view.  The low-resolution P4/P8 values themselves stay 4x4/8x8.
    x = native_image_tensor(candidate.path, device)
    raw = extractor(x).float()[0]
    log_luma = normalized_log_luminance(x, extractor.eps)[0, 0].cpu().numpy()
    p4 = raw[48:64].reshape(4, 4).cpu().numpy()
    p8 = raw[64:128].reshape(8, 8).cpu().numpy()
    return display, log_luma, p4, p8


def render_map_grid(
    candidates: Sequence[Candidate],
    extractor: IlluminationStateExtractor,
    size: int,
    device: torch.device,
    output: Path,
) -> None:
    rows = len(candidates)
    fig, axes = plt.subplots(rows, 4, figsize=(12.0, 2.65 * rows), squeeze=False)
    headers = ("Input", "Log-luminance", "$P_4$ (4×4)", "$P_8$ (8×8)")
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=12, pad=8)
    heat = None
    for row, candidate in enumerate(candidates):
        display, log_luma, p4, p8 = candidate_arrays(candidate, extractor, size, device)
        axes[row, 0].imshow(display)
        axes[row, 1].imshow(log_luma, cmap="gray", vmin=LOG_DISPLAY_MIN, vmax=LOG_DISPLAY_MAX)
        vector_spatial_map(axes[row, 2], p4)
        heat = vector_spatial_map(axes[row, 3], p8)
        axes[row, 0].set_ylabel(candidate.scene, fontsize=9)
        for col in range(4):
            axes[row, col].set_box_aspect(display.shape[0] / display.shape[1])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col > 0:
                for spine in axes[row, col].spines.values():
                    spine.set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.91, top=0.96, bottom=0.02, wspace=0.04, hspace=0.16)
    if heat is not None:
        cax = fig.add_axes((0.93, 0.12, 0.012, 0.76))
        fig.colorbar(heat, cax=cax, label="$P_4/P_8$ normalized log-luminance (display range 0.4–1.0)")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def vector_spatial_map(ax: plt.Axes, values: np.ndarray):
    """Draw each P4/P8 cell as a PDF vector rectangle, with image-style origin."""
    height, width = values.shape
    mesh = ax.pcolormesh(
        np.arange(width + 1),
        np.arange(height + 1),
        values,
        cmap="gray",
        vmin=MAP_DISPLAY_MIN,
        vmax=MAP_DISPLAY_MAX,
        shading="flat",
        edgecolors="none",
        antialiased=False,
        rasterized=False,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    return mesh


def fit_rect_3x2(content_aspect: float) -> Tuple[float, float, float, float]:
    """Fit an unmodified aspect ratio inside a 3:2 canvas."""
    target_aspect = 1.5
    if float(content_aspect) <= target_aspect:
        height = 1.0
        width = float(content_aspect)
        return (0.5 * (target_aspect - width), 0.0, width, height)
    width = target_aspect
    height = target_aspect / float(content_aspect)
    return (0.0, 0.5 * (1.0 - height), width, height)


def show_padded_3x2(
    ax: plt.Axes,
    array: np.ndarray,
    content_aspect: float,
    **imshow_kwargs,
):
    """Show an image without stretching/cropping inside a white 3:2 panel."""
    x0, y0, width, height = fit_rect_3x2(content_aspect)
    image = ax.imshow(
        array,
        extent=(x0, x0 + width, y0 + height, y0),
        aspect="equal",
        **imshow_kwargs,
    )
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(1.0, 0.0)
    ax.set_facecolor("white")
    ax.set_box_aspect(2.0 / 3.0)
    return image


def vector_spatial_map_padded(ax: plt.Axes, values: np.ndarray, content_aspect: float):
    """Draw P4/P8 as vector cells inside the same padded 3:2 field."""
    height, width = values.shape
    x0, y0, rect_w, rect_h = fit_rect_3x2(content_aspect)
    mesh = ax.pcolormesh(
        np.linspace(x0, x0 + rect_w, width + 1),
        np.linspace(y0, y0 + rect_h, height + 1),
        values,
        cmap="gray",
        vmin=MAP_DISPLAY_MIN,
        vmax=MAP_DISPLAY_MAX,
        shading="flat",
        edgecolors="none",
        antialiased=False,
        rasterized=False,
    )
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(1.0, 0.0)
    ax.set_facecolor("white")
    ax.set_box_aspect(2.0 / 3.0)
    return mesh


def export_candidate_assets(
    candidates: Sequence[Candidate],
    extractor: IlluminationStateExtractor,
    size: int,
    device: torch.device,
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root.parent / "map_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(candidates[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in candidates)
    for rank, candidate in enumerate(candidates, start=1):
        item_root = root / f"{rank:02d}_{candidate.source}_{image_id(candidate.path)}"
        item_root.mkdir(parents=True, exist_ok=True)
        display, log_luma, p4, p8 = candidate_arrays(candidate, extractor, size, device)
        shutil.copy2(candidate.path, item_root / f"input{Path(candidate.path).suffix.lower()}")
        np.savetxt(item_root / "p4.csv", p4, delimiter=",")
        np.savetxt(item_root / "p8.csv", p8, delimiter=",")
        np.save(item_root / "log_luminance.npy", log_luma)
        height, width = display.shape[:2]
        log_display = np.clip((log_luma - LOG_DISPLAY_MIN) / (LOG_DISPLAY_MAX - LOG_DISPLAY_MIN), 0.0, 1.0)
        Image.fromarray(np.round(log_display * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(
            item_root / "log_luminance.png"
        )
        for name, layout in (("p4_visual.png", p4), ("p8_visual.png", p8)):
            layout_tensor = torch.from_numpy(layout).view(1, 1, *layout.shape).float()
            enlarged = F.interpolate(layout_tensor, size=(height, width), mode="nearest")[0, 0].numpy()
            enlarged = np.clip((enlarged - MAP_DISPLAY_MIN) / (MAP_DISPLAY_MAX - MAP_DISPLAY_MIN), 0.0, 1.0)
            Image.fromarray(np.round(enlarged * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(
                item_root / name
            )
        single = [candidate]
        render_map_grid(single, extractor, size, device, item_root / "panel.png")


def perturb(x: torch.Tensor, kind: str, value: float, generator: torch.Generator = None) -> torch.Tensor:
    if kind == "Exposure":
        return (x * (2.0 ** float(value))).clamp(0.0, 1.0)
    if kind == "Gamma":
        return x.clamp(0.0, 1.0).pow(float(value))
    if kind == "Gaussian noise":
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return (x + float(value) * noise).clamp(0.0, 1.0)
    if kind == "Sharpening":
        blurred = _separable_gaussian_blur(x, sigma=1.0)
        return (x + float(value) * (x - blurred)).clamp(0.0, 1.0)
    raise KeyError(kind)


@torch.inference_mode()
def controlled_response(
    paths: Sequence[str],
    extractor: IlluminationStateExtractor,
    normalizer: IlluminationStateNormalizer,
    size: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, float]]:
    protocols = {
        "Exposure": [(magnitude, (-magnitude, magnitude)) for magnitude in EXPOSURE_MAGNITUDES],
        "Gamma": [(deviation, (1.0 - deviation, 1.0 + deviation)) for deviation in GAMMA_DEVIATIONS],
        "Gaussian noise": [(sigma, (sigma,)) for sigma in NOISE_LEVELS],
        "Sharpening": [(amount, (amount,)) for amount in SHARPEN_LEVELS],
    }
    generators: Dict[Tuple[str, int], torch.Generator] = {}
    for level in range(4):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 1000 + level)
        generators[("Gaussian noise", level)] = gen
    rows: List[Dict[str, object]] = []
    for start in range(0, len(paths), int(batch_size)):
        batch_paths = paths[start : start + int(batch_size)]
        x = load_batch(batch_paths, size, device)
        state = normalizer(extractor(x)).float()
        for kind, levels in protocols.items():
            for level, (strength, applied_values) in enumerate(levels, start=1):
                directional_distances = []
                for applied_value in applied_values:
                    altered = perturb(x, kind, applied_value, generators.get((kind, level - 1)))
                    altered_state = normalizer(extractor(altered)).float()
                    directional_distances.append(
                        torch.linalg.vector_norm(altered_state - state, dim=1) / math.sqrt(128.0)
                    )
                # Exposure and Gamma use symmetric darkening/brightening pairs.
                # Averaging the two directions makes Level 1 -> 4 a true
                # monotonic increase in perturbation magnitude.
                distances = torch.stack(directional_distances, dim=0).mean(dim=0)
                for path, distance in zip(batch_paths, distances.cpu().tolist()):
                    rows.append(
                        {
                            "image": Path(path).name,
                            "perturbation": kind,
                            "level": level,
                            "parameter": float(strength),
                            "applied_values": "|".join(f"{float(value):.4g}" for value in applied_values),
                            "normalized_state_change": float(distance),
                        }
                    )
        print(f"[Fig6-B] {min(start + len(batch_paths), len(paths))}/{len(paths)}")
    summary: List[Dict[str, object]] = []
    for kind, levels in protocols.items():
        for level, (strength, applied_values) in enumerate(levels, start=1):
            data = np.asarray(
                [row["normalized_state_change"] for row in rows if row["perturbation"] == kind and row["level"] == level],
                dtype=np.float64,
            )
            summary.append(
                {
                    "perturbation": kind,
                    "level": level,
                    "parameter": float(strength),
                    "applied_values": "|".join(f"{float(value):.4g}" for value in applied_values),
                    "mean": float(data.mean()),
                    "standard_error": float(data.std(ddof=1) / math.sqrt(len(data))),
                    "num_images": int(len(data)),
                }
            )
    category_means = {
        kind: float(np.mean([row["normalized_state_change"] for row in rows if row["perturbation"] == kind]))
        for kind in protocols
    }
    aggregates = {
        "exposure_response": category_means["Exposure"],
        "gamma_response": category_means["Gamma"],
        "noise_response": category_means["Gaussian noise"],
        "sharpen_response": category_means["Sharpening"],
        "delta_photo": 0.5 * (category_means["Exposure"] + category_means["Gamma"]),
        "delta_hf": 0.5 * (category_means["Gaussian noise"] + category_means["Sharpening"]),
    }
    return rows, summary, aggregates


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_response(summary: Sequence[Dict[str, object]], output: Path) -> None:
    colors = {
        "Exposure": "#D55E00",
        "Gamma": "#CC79A7",
        "Gaussian noise": "#0072B2",
        "Sharpening": "#009E73",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    x = np.arange(1, 5)
    for kind in colors:
        selected = sorted((row for row in summary if row["perturbation"] == kind), key=lambda row: int(row["level"]))
        means = np.asarray([row["mean"] for row in selected])
        sem = np.asarray([row["standard_error"] for row in selected])
        ax.errorbar(x, means, yerr=sem, marker="o", markersize=7, linewidth=2.2, capsize=3, label=kind, color=colors[kind])
    ax.set_xticks(x, [f"Level {level}" for level in x], fontsize=13)
    ax.set_xlabel("Increasing perturbation magnitude", fontsize=14)
    ax.set_ylabel("Normalized state change", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def exposure_probe_states(
    paths: Sequence[str],
    extractor: IlluminationStateExtractor,
    normalizer: IlluminationStateNormalizer,
    size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    for start in range(0, len(paths), int(batch_size)):
        batch_paths = paths[start : start + int(batch_size)]
        x = load_batch(batch_paths, size, device)
        for label, exposure in enumerate(PROBE_EXPOSURES):
            state = normalizer(extractor(perturb(x, "Exposure", exposure))).float().cpu().numpy()
            features.append(state)
            labels.append(np.full((len(batch_paths),), label, dtype=np.int64))
            groups.append(np.arange(start, start + len(batch_paths), dtype=np.int64))
        print(f"[Fig6-C probe] {min(start + len(batch_paths), len(paths))}/{len(paths)}")
    return np.concatenate(features), np.concatenate(labels), np.concatenate(groups)


def run_exposure_probe(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    num_images: int,
    splits: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    image_indices = np.arange(num_images)
    splitter = ShuffleSplit(n_splits=int(splits), test_size=0.30, random_state=int(seed))
    rows: List[Dict[str, object]] = []
    for split, (train_images, test_images) in enumerate(splitter.split(image_indices), start=1):
        train_mask = np.isin(groups, train_images)
        test_mask = np.isin(groups, test_images)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, solver="lbfgs", multi_class="auto", random_state=int(seed) + split),
        )
        classifier.fit(features[train_mask], labels[train_mask])
        accuracy = classifier.score(features[test_mask], labels[test_mask])
        rows.append(
            {
                "split": split,
                "train_base_images": int(len(train_images)),
                "test_base_images": int(len(test_images)),
                "accuracy": float(accuracy),
            }
        )
    values = np.asarray([row["accuracy"] for row in rows], dtype=np.float64)
    return rows, {"exposure_probe_accuracy_mean": float(values.mean()), "exposure_probe_accuracy_std": float(values.std(ddof=1))}


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x.t() @ y).pow(2)
    denom = torch.linalg.matrix_norm(x.t() @ x) * torch.linalg.matrix_norm(y.t() @ y)
    return float((cross / denom.clamp_min(1e-12)).item())


@torch.inference_mode()
def compute_cka(
    paths: Sequence[str],
    extractor: IlluminationStateExtractor,
    normalizer: IlluminationStateNormalizer,
    priors: Dict[str, object],
    size: int,
    batch_size: int,
    device: torch.device,
) -> float:
    clip = build_classifier(
        feature_dim=None,
        gamma=float(priors.get("clip_gamma", 0.5)),
        image_size=int(priors.get("clip_image_size", 224)),
        clip_model_name=str(priors.get("clip_model_name", "ViT-B-32")),
        clip_pretrained=str(priors.get("clip_pretrained", "openai")),
    ).to(device).eval()
    for parameter in clip.parameters():
        parameter.requires_grad = False
    residual = priors["residual"].to(device).float()
    states: List[torch.Tensor] = []
    content: List[torch.Tensor] = []
    for start in range(0, len(paths), int(batch_size)):
        batch_paths = paths[start : start + int(batch_size)]
        x = load_batch(batch_paths, size, device)
        states.append(normalizer(extractor(x)).float().cpu())
        clip_x = preprocess_for_clip(clip, x).float()
        z = clip._encode_image_feature(clip_x).float() if hasattr(clip, "_encode_image_feature") else clip(x).float()
        q = content_feature_from_clip(z, residual)
        content.append(q.cpu())
        print(f"[Fig6-C CKA] {min(start + len(batch_paths), len(paths))}/{len(paths)}")
    return linear_cka(torch.cat(states), torch.cat(content))


def combined_figure(
    candidates: Sequence[Candidate],
    extractor: IlluminationStateExtractor,
    summary: Sequence[Dict[str, object]],
    size: int,
    device: torch.device,
    output: Path,
    layout_mode: str = "compact",
) -> None:
    rows = len(candidates)
    if layout_mode == "equal_width":
        # Extra row height makes column width, rather than row height, the
        # limiting dimension. Match each grid row to its native box aspect so
        # wide images do not leave unused vertical space inside equal rows.
        fig = plt.figure(figsize=(12.2, 11.3))
        hspace = 0.02
        row_height_ratios = []
        for candidate in candidates:
            native = load_rgb(candidate.path)
            row_height_ratios.append(native.height / native.width)
        # Keep the three image rows tight, but reserve a narrow independent
        # separator before panel (b) so its label cannot overlap row three.
        grid_height_ratios = row_height_ratios + [0.18, 1.55]
        grid_rows = rows + 2
        chart_row = rows + 1
    else:
        fig = plt.figure(figsize=(12.2, 9.7))
        hspace = 0.07
        row_height_ratios = [1.0] * rows
        grid_height_ratios = row_height_ratios + [1.55]
        grid_rows = rows + 1
        chart_row = rows
    grid = fig.add_gridspec(
        grid_rows,
        4,
        height_ratios=grid_height_ratios,
        hspace=hspace,
        wspace=0.05,
    )
    headers = ("Input", "Log-luminance", "$P_4$ (4×4)", "$P_8$ (8×8)")
    heat = None
    for row, candidate in enumerate(candidates):
        display, log_luma, p4, p8 = candidate_arrays(candidate, extractor, size, device)
        arrays = (display, log_luma, p4, p8)
        content_aspect = display.shape[1] / display.shape[0]
        for col, array in enumerate(arrays):
            ax = fig.add_subplot(grid[row, col])
            if row == 0:
                ax.set_title(headers[col], fontsize=12)
            if layout_mode == "padded_3x2":
                if col == 0:
                    show_padded_3x2(ax, array, content_aspect)
                    ax.set_ylabel(candidate.scene, fontsize=9)
                elif col == 1:
                    heat = show_padded_3x2(
                        ax,
                        array,
                        content_aspect,
                        cmap="gray",
                        vmin=LOG_DISPLAY_MIN,
                        vmax=LOG_DISPLAY_MAX,
                    )
                else:
                    heat = vector_spatial_map_padded(ax, array, content_aspect)
            else:
                if col == 0:
                    ax.imshow(array)
                    ax.set_ylabel(candidate.scene, fontsize=9)
                elif col == 1:
                    heat = ax.imshow(
                        array,
                        cmap="gray",
                        vmin=LOG_DISPLAY_MIN,
                        vmax=LOG_DISPLAY_MAX,
                        aspect="auto",
                    )
                else:
                    heat = vector_spatial_map(ax, array)
                ax.set_box_aspect(display.shape[0] / display.shape[1])
            ax.set_xticks([])
            ax.set_yticks([])
    ax = fig.add_subplot(grid[chart_row, :])
    colors = {"Exposure": "#D55E00", "Gamma": "#CC79A7", "Gaussian noise": "#0072B2", "Sharpening": "#009E73"}
    x = np.arange(1, 5)
    for kind, color in colors.items():
        selected = sorted((item for item in summary if item["perturbation"] == kind), key=lambda item: int(item["level"]))
        ax.errorbar(
            x,
            [item["mean"] for item in selected],
            yerr=[item["standard_error"] for item in selected],
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=3,
            color=color,
            label=kind,
        )
    ax.set_xticks(x, [f"Level {level}" for level in x], fontsize=13)
    ax.set_xlabel("Increasing perturbation magnitude", fontsize=14)
    ax.set_ylabel("Normalized state change", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=12.5)
    ax.text(-0.04, 1.04, "(b)", transform=ax.transAxes, fontsize=16, fontweight="bold")
    fig.text(0.01, 0.985, "(a)", fontsize=13, fontweight="bold", va="top")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_name(f"{output.stem}_vector.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(f"cuda:{int(args.gpu)}" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    priors = torch.load(args.priors, map_location="cpu")
    extractor = IlluminationStateExtractor().to(device).eval()
    normalizer = IlluminationStateNormalizer(priors["state_mean"], priors["state_std"]).to(device).eval()
    print(f"[Fig6] device={device} out={out}")

    lol_paths = list_images(args.lol_root)
    backlit_paths = list_images(args.backlit_root)
    response_paths = list_images(args.response_root)
    if int(args.max_response_images) > 0:
        response_paths = response_paths[: int(args.max_response_images)]
    if not lol_paths or not backlit_paths or not response_paths:
        raise RuntimeError(f"Missing images: LOL={len(lol_paths)} Backlit={len(backlit_paths)} response={len(response_paths)}")

    if bool(args.render_only):
        result_path = out / "fig6_results.json"
        response_path = out / "controlled_response_summary.csv"
        if not result_path.is_file() or not response_path.is_file():
            raise FileNotFoundError("--render_only requires existing fig6_results.json and controlled_response_summary.csv")
        with result_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        candidates = [Candidate(**item) for item in saved["map_candidates"]]
        with response_path.open("r", newline="", encoding="utf-8") as handle:
            response_summary = []
            for row in csv.DictReader(handle):
                response_summary.append(
                    {
                        "perturbation": row["perturbation"],
                        "level": int(row["level"]),
                        "parameter": float(row["parameter"]),
                        "mean": float(row["mean"]),
                        "standard_error": float(row["standard_error"]),
                        "num_images": int(row["num_images"]),
                    }
                )
        export_candidate_assets(candidates, extractor, int(args.image_size), device, out / "map_candidates")
        render_map_grid(candidates, extractor, int(args.image_size), device, out / "fig6a_all_5_candidates.png")
        selected = candidates[: int(args.selected_count)]
        render_map_grid(selected, extractor, int(args.image_size), device, out / "fig6a_selected_3.png")
        plot_response(response_summary, out / "fig6b_response_curves.png")
        combined_figure(selected, extractor, response_summary, int(args.image_size), device, out / "fig6_state_validation.png")
        combined_figure(
            selected,
            extractor,
            response_summary,
            int(args.image_size),
            device,
            out / "fig6_state_validation_equal_width.png",
            layout_mode="equal_width",
        )
        combined_figure(
            selected,
            extractor,
            response_summary,
            int(args.image_size),
            device,
            out / "fig6_state_validation_padded_3x2.png",
            layout_mode="padded_3x2",
        )
        print(f"[Fig6] Render-only update complete: {out}")
        return

    scored = score_map_sources(lol_paths, backlit_paths, extractor, int(args.image_size), device)
    write_csv(out / "all_map_candidate_scores.csv", [asdict(item) for item in scored])
    candidates = select_candidates(scored, int(args.candidate_count))
    export_candidate_assets(candidates, extractor, int(args.image_size), device, out / "map_candidates")
    render_map_grid(candidates, extractor, int(args.image_size), device, out / "fig6a_all_5_candidates.png")
    selected = candidates[: int(args.selected_count)]
    render_map_grid(selected, extractor, int(args.image_size), device, out / "fig6a_selected_3.png")

    response_rows, response_summary, aggregates = controlled_response(
        response_paths,
        extractor,
        normalizer,
        int(args.image_size),
        int(args.batch_size),
        device,
        int(args.seed),
    )
    write_csv(out / "controlled_response_per_image.csv", response_rows)
    write_csv(out / "controlled_response_summary.csv", response_summary)
    plot_response(response_summary, out / "fig6b_response_curves.png")

    probe_x, probe_y, probe_groups = exposure_probe_states(
        response_paths,
        extractor,
        normalizer,
        int(args.image_size),
        int(args.batch_size),
        device,
    )
    probe_rows, probe_summary = run_exposure_probe(
        probe_x,
        probe_y,
        probe_groups,
        len(response_paths),
        int(args.probe_splits),
        int(args.seed),
    )
    write_csv(out / "exposure_probe_splits.csv", probe_rows)
    cka = compute_cka(
        response_paths,
        extractor,
        normalizer,
        priors,
        int(args.image_size),
        int(args.batch_size),
        device,
    )

    combined_figure(selected, extractor, response_summary, int(args.image_size), device, out / "fig6_state_validation.png")
    combined_figure(
        selected,
        extractor,
        response_summary,
        int(args.image_size),
        device,
        out / "fig6_state_validation_equal_width.png",
        layout_mode="equal_width",
    )
    combined_figure(
        selected,
        extractor,
        response_summary,
        int(args.image_size),
        device,
        out / "fig6_state_validation_padded_3x2.png",
        layout_mode="padded_3x2",
    )
    results = {
        "protocol": {
            "state": "training-stat standardized fixed 128-D illumination state",
            "distance": "L2(s(I') - s(I)) / sqrt(128)",
            "map_sources": {"LOLv1-Test": args.lol_root, "Backlit300": args.backlit_root},
            "controlled_response_and_diagnostic_source": args.response_root,
            "controlled_source_note": "BAID-380 normal-light GT; independent of SCIST training",
            "num_response_images": len(response_paths),
            "image_size": int(args.image_size),
            "exposure_magnitudes_ev": EXPOSURE_MAGNITUDES,
            "exposure_applied_pairs_ev": [(-value, value) for value in EXPOSURE_MAGNITUDES],
            "gamma_deviations_from_one": GAMMA_DEVIATIONS,
            "gamma_applied_pairs": [(1.0 - value, 1.0 + value) for value in GAMMA_DEVIATIONS],
            "noise_sigmas": NOISE_LEVELS,
            "sharpen_amounts": SHARPEN_LEVELS,
            "probe_exposures": PROBE_EXPOSURES,
            "probe_split": "5 image-identity splits; 70% train / 30% test",
            "seed": int(args.seed),
        },
        "map_candidates": [asdict(item) for item in candidates],
        "controlled_response": aggregates,
        "representation_diagnostic": {**probe_summary, "cka_state_content": cka},
    }
    with (out / "fig6_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with (out / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Fig. 6 Illumination State Validation\n\n")
        handle.write(f"Controlled-response/diagnostic set: `{args.response_root}` ({len(response_paths)} independent normal-light GT images).\n\n")
        handle.write("## Main statistics\n\n")
        handle.write("| Exposure | Gamma | Noise | Sharpen | Δphoto | ΔHF | Probe accuracy | CKA(s,q) |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        handle.write(
            f"| {aggregates['exposure_response']:.4f} | {aggregates['gamma_response']:.4f} | "
            f"{aggregates['noise_response']:.4f} | {aggregates['sharpen_response']:.4f} | "
            f"{aggregates['delta_photo']:.4f} | {aggregates['delta_hf']:.4f} | "
            f"{probe_summary['exposure_probe_accuracy_mean'] * 100:.2f}% ± {probe_summary['exposure_probe_accuracy_std'] * 100:.2f}% | {cka:.4f} |\n\n"
        )
        handle.write("Five map candidates are in `map_candidates/`; the first three form the default paper preview.\n\n")
        handle.write("Log-luminance uses one fixed grayscale display window [0.5, 1.0] so dark inputs remain visually dark. P4/P8 use [0.4, 1.0] and nearest-neighbor display interpolation so their cells remain visible. Raw values are unchanged.\n\n")
        handle.write("## Candidate scenes\n\n")
        for rank, candidate in enumerate(candidates, start=1):
            handle.write(f"{rank}. `{candidate.source}/{Path(candidate.path).name}` — {candidate.scene};\n")
        handle.write("\n## Reproduction\n\n```bash\nconda activate dl_env\nGPU_ID=2 bash scripts/run_fig6_state_validation.sh\n```\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
