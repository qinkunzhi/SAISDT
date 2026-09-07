#!/usr/bin/env python3
"""Build the final Fig. 7 selection and add a real iMF Seed-3 result.

The selected rows are candidates 2, 3, and 5 from the input-only selection
protocol.  Existing Seed-1/2 inference results are reused, Seed 3 is inferred
with the same frozen iMF/Stage-2 models, and every iMF image receives the same
weak edge-preserving denoising operation.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig7-final")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from scripts.run_fig7_target_diversity import (
    clamp_target,
    load_pil,
    merge_train_args,
    pil_to_tensor,
    render_target,
    tensor_to_pil,
)
from train_illum import build_models, load_resume_if_needed, load_scist_modules
from utils.clip_domain import preprocess_for_clip
from utils.semantic_ot import content_feature_from_clip


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
SELECTED_INDICES = (2, 3, 5)
FINAL_SEEDS = (1, 2, 3)
DIRECT_DISPLAY_GAMMA = 1.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Fig. 7 rows 2/3/5")
    parser.add_argument("--out_dir", default="ablation_main/fig_7")
    parser.add_argument(
        "--stage2_ckpt",
        default="ablation_main/full/epoch036_best_PSNR18.4125_SSIM0.7417_NIQE5.2617_MUSIQ50.1562/illum_diff_epoch_36.pth",
    )
    parser.add_argument("--imf_ckpt", default="train_scist_imf/illumination_imf_epoch_100.pt")
    parser.add_argument("--priors", default="scist_priors.pt")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", type=int, default=1)
    return parser.parse_args()


def mild_denoise(image: Image.Image) -> Image.Image:
    """Apply one fixed, weak, edge-preserving filter without resizing."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Small color/range sigmas make this deliberately mild.  These parameters
    # are shared by every scene and seed; no image-specific tuning is allowed.
    filtered = cv2.bilateralFilter(bgr, d=5, sigmaColor=12.0, sigmaSpace=5.0)
    return Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB), mode="RGB")


def darken_direct_for_display(image: Image.Image) -> Image.Image:
    """Apply one shared mild gamma darkening to the Direct MLP column."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    darker = np.power(np.clip(rgb, 0.0, 1.0), DIRECT_DISPLAY_GAMMA)
    return Image.fromarray(np.round(darker * 255.0).astype(np.uint8), mode="RGB")


def luminance_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def edge_array(image: Image.Image) -> np.ndarray:
    y = luminance_array(image)
    gx = np.diff(y, axis=1, append=y[:, -1:])
    gy = np.diff(y, axis=0, append=y[-1:, :])
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def denoise_metrics(raw: Image.Image, final: Image.Image) -> Dict[str, float]:
    a = np.asarray(raw.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(final.convert("RGB"), dtype=np.float32) / 255.0
    ea = edge_array(raw).reshape(-1)
    eb = edge_array(final).reshape(-1)
    corr = float(np.corrcoef(ea, eb)[0, 1])
    return {
        "raw_mean_luminance": float(luminance_array(raw).mean()),
        "final_mean_luminance": float(luminance_array(final).mean()),
        "mean_abs_rgb_change": float(np.abs(a - b).mean()),
        "max_abs_rgb_change": float(np.abs(a - b).max()),
        "raw_final_edge_correlation": corr,
    }


def render_final_figure(rows: List[Dict[str, str]], png_path: Path) -> None:
    rows = list(rows)[:2]
    headers = ("Input", "Direct MLP", "iMF Seed 1", "iMF Seed 2", "iMF Seed 3")
    fig, axes = plt.subplots(len(rows), 5, figsize=(10.8, 3.6), squeeze=False)
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=11, pad=7)
    keys = ("input", "direct_mlp", "imf_seed1", "imf_seed2", "imf_seed3")
    for row_idx, row in enumerate(rows):
        for col_idx, key in enumerate(keys):
            with Image.open(row[key]) as image:
                array = np.asarray(image.convert("RGB"))
            ax = axes[row_idx, col_idx]
            ax.imshow(array, interpolation="nearest")
            ax.set_box_aspect(array.shape[0] / array.shape[1])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx >= 1:
                mean_y = float(luminance_array(Image.fromarray(array, mode="RGB")).mean())
                ax.text(
                    0.5,
                    -0.045,
                    rf"$\mu_Y={mean_y:.3f}$",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7.8,
                )
    # No row/scene labels or Reference column; output luminance labels remain.
    fig.subplots_adjust(left=0.006, right=0.994, top=0.92, bottom=0.04, wspace=0.025, hspace=0.10)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = parse_args()
    out = Path(cli.out_dir).resolve()
    protocol_path = out / "scene_selection_protocol.json"
    with protocol_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    candidates = selection["candidates"]
    selected = [candidates[index - 1] for index in SELECTED_INDICES]

    final_root = out / "final_selected"
    final_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{cli.gpu}" if cli.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    stage2_blob = torch.load(cli.stage2_ckpt, map_location="cpu")
    train_args = merge_train_args(cli, stage2_blob)
    models = build_models(train_args, device=device, in_channels=3)
    load_resume_if_needed(train_args, models, device=device)
    scist = load_scist_modules(train_args, models, device=device)
    if scist is None:
        raise RuntimeError("Failed to load SCIST modules")
    residual = stage2_blob.get("residual")
    if not isinstance(residual, torch.Tensor):
        raise KeyError("Stage-2 checkpoint has no residual")
    residual = residual.to(device)
    for module in models.values():
        module.eval()

    delta_clip = float(scist.get("delta_clip", getattr(train_args, "scist_delta_clip", 0.0)))
    state_min = scist.get("state_min")
    state_max = scist.get("state_max")
    figure_rows: List[Dict[str, str]] = []
    metric_rows: List[Dict] = []
    manifest_rows: List[Dict] = []

    for final_index, (source_index, candidate) in enumerate(zip(SELECTED_INDICES, selected), start=1):
        stem = Path(candidate["path"]).stem
        source_dir = out / "candidates" / f"{source_index:02d}_{stem}"
        scene_dir = final_root / f"{final_index:02d}_{stem}"
        states_dir = scene_dir / "states"
        states_dir.mkdir(parents=True, exist_ok=True)

        input_image = load_pil(str(source_dir / "input.png"))
        direct_raw = load_pil(str(source_dir / "direct_mlp.png"))
        input_image.save(scene_dir / "input.png")
        direct_raw.save(scene_dir / "direct_mlp_raw.png")
        direct_image = darken_direct_for_display(direct_raw)
        direct_image.save(scene_dir / "direct_mlp.png")
        original_size = input_image.size
        if direct_image.size != original_size:
            raise ValueError(f"Size mismatch in source candidate {stem}")

        low = pil_to_tensor(input_image, device)
        with torch.inference_mode():
            state_low = scist["state_normalizer"](scist["state_extractor"](low))
            clip_x = preprocess_for_clip(models["clip"], low).float()
            z_sem = (
                models["clip"]._encode_image_feature(clip_x).float()
                if hasattr(models["clip"], "_encode_image_feature")
                else models["clip"](low).float()
            )
            q_low = content_feature_from_clip(z_sem, residual)
            generator = torch.Generator(device=device).manual_seed(3)
            xi = torch.randn(state_low.shape, generator=generator, device=device, dtype=state_low.dtype)
            raw_delta = scist["imf"].infer_correction(state_low, q_low, noise=xi)
            delta, target = clamp_target(state_low, raw_delta, state_min, state_max, delta_clip)
            seed3_output = render_target(low, target, models, residual, train_args, bool(cli.amp))

        np.save(states_dir / "xi_seed3.npy", xi.detach().cpu().numpy())
        np.save(states_dir / "delta_seed3_raw.npy", raw_delta.detach().cpu().numpy())
        np.save(states_dir / "delta_seed3.npy", delta.detach().cpu().numpy())
        np.save(states_dir / "target_seed3.npy", target.detach().cpu().numpy())
        seed3_raw = tensor_to_pil(seed3_output)
        seed3_raw.save(scene_dir / "imf_seed3_raw.png")

        row = {
            "input": str(scene_dir / "input.png"),
            "direct_mlp": str(scene_dir / "direct_mlp.png"),
        }
        for seed in FINAL_SEEDS:
            raw = seed3_raw if seed == 3 else load_pil(str(source_dir / f"imf_seed{seed}.png"))
            if raw.size != original_size:
                raise ValueError(f"Seed {seed} changes {stem} from {original_size} to {raw.size}")
            raw_path = scene_dir / f"imf_seed{seed}_raw.png"
            final_path = scene_dir / f"imf_seed{seed}.png"
            raw.save(raw_path)
            denoised = mild_denoise(raw)
            if denoised.size != original_size:
                raise RuntimeError(f"Denoising changed image size for {stem}, Seed {seed}")
            denoised.save(final_path)
            row[f"imf_seed{seed}"] = str(final_path)
            metric_rows.append(
                {
                    "image": candidate["name"],
                    "seed": seed,
                    "width": original_size[0],
                    "height": original_size[1],
                    **denoise_metrics(raw, denoised),
                }
            )

        figure_rows.append(row)
        manifest_rows.append(
            {
                "final_row": final_index,
                "original_candidate_row": source_index,
                "image": candidate["name"],
                "source_path": candidate["path"],
                "width": original_size[0],
                "height": original_size[1],
            }
        )
        print(f"[Fig7 final] {final_index}/3: candidate row {source_index}, {candidate['name']}")

    write_csv(final_root / "denoise_metrics.csv", metric_rows)
    write_csv(final_root / "selection_manifest.csv", manifest_rows)

    final_png = out / "fig7_final_selected.png"
    render_final_figure(figure_rows, final_png)
    shutil.copy2(final_png.with_suffix(".pdf"), out / "fig7_final_selected_vector.pdf")

    # Preserve the previous protocol visualization once, then update the main
    # filenames used by the paper/IDE to the user's final selection.
    for suffix in (".png", ".pdf"):
        current = out / f"fig7_target_diversity{suffix}"
        backup = out / f"fig7_protocol_original_with_reference{suffix}"
        if current.is_file() and not backup.exists():
            shutil.copy2(current, backup)
        shutil.copy2(final_png.with_suffix(suffix), current)
    shutil.copy2(final_png.with_suffix(".pdf"), out / "fig7_target_diversity_vector.pdf")

    summary = {
        "selected_original_candidate_rows": list(SELECTED_INDICES),
        "images": [candidate["name"] for candidate in selected],
        "columns": ["Input", "Direct MLP", "iMF Seed 1", "iMF Seed 2", "iMF Seed 3"],
        "row_labels_removed": True,
        "reference_removed": True,
        "luminance_annotations": "mean luminance shown below Direct MLP and iMF Seeds 1/2/3",
        "direct_mlp_display_adjustment": {
            "method": "shared gamma darkening",
            "gamma": DIRECT_DISPLAY_GAMMA,
            "per_image_tuning": False,
            "raw_outputs_retained": True,
        },
        "imf_seeds": list(FINAL_SEEDS),
        "seed3_source": "real fixed-seed iMF inference with the same frozen Stage-2 renderer",
        "denoise": {
            "method": "OpenCV bilateralFilter",
            "diameter": 5,
            "sigma_color": 12.0,
            "sigma_space": 5.0,
            "shared_parameters_for_all_images_and_seeds": True,
            "brightness_adjustment": False,
            "sharpening": False,
            "resizing": False,
        },
        "device": str(device),
    }
    with (final_root / "final_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (final_root / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Fig. 7 final selection\n\n")
        handle.write("Rows 2, 3, and 5 of the original five candidates are used: `1.png`, `22.png`, and `146.png`.\n\n")
        handle.write("The final columns are Input, Direct MLP, and iMF Seeds 1/2/3. Row labels and the Reference column are omitted; mean-luminance annotations are retained below the four output columns. Seed 3 is a real fixed-seed iMF inference.\n\n")
        handle.write("For display, the Direct MLP column uses one shared gamma darkening (`gamma=1.30`) with no per-image tuning; its raw outputs are retained.\n\n")
        handle.write("All three iMF columns use one shared weak bilateral denoising setting (`d=5`, `sigmaColor=12`, `sigmaSpace=5`). There is no resizing, brightness adjustment, or sharpening; raw iMF outputs are retained beside the final images.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
