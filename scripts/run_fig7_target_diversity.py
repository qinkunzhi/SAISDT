#!/usr/bin/env python3
"""Generate paper Fig. 7: deterministic target versus fixed-seed iMF diversity."""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig7")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.direct_correction_mlp import DirectCorrectionMLP
from train_illum import (
    apply_scist_condition_mask,
    build_models,
    load_resume_if_needed,
    load_scist_modules,
    parse_args as parse_train_args,
)
from utils.clip_domain import prepare_semantic_domain_condition, preprocess_for_clip
from utils.semantic_ot import content_feature_from_clip


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEEDS = (0, 1, 2)


@dataclass
class CandidateScore:
    path: str
    name: str
    mean_luminance: float
    p4_std: float
    backlit_score: float
    dark_fraction: float
    scene: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCIST Fig. 7 conditional target diversity")
    parser.add_argument("--out_dir", default="ablation_main/fig_7")
    parser.add_argument("--low_root", default="/home/zhiqinkun/LLIM/datasets/data/LOLv1/Test/input")
    parser.add_argument("--gt_root", default="/home/zhiqinkun/LLIM/datasets/data/LOLv1/Test/target")
    parser.add_argument(
        "--stage2_ckpt",
        default="ablation_main/full/epoch036_best_PSNR18.4125_SSIM0.7417_NIQE5.2617_MUSIQ50.1562/illum_diff_epoch_36.pth",
    )
    parser.add_argument("--imf_ckpt", default="train_scist_imf/illumination_imf_epoch_100.pt")
    parser.add_argument("--direct_mlp_ckpt", default="ablation_main/fig_7/direct_mlp/direct_mlp_latest.pt")
    parser.add_argument("--priors", default="scist_priors.pt")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--selected_count", type=int, default=3)
    parser.add_argument("--candidate_count", type=int, default=5)
    parser.add_argument("--amp", type=int, default=1)
    return parser.parse_args()


def list_images(root: str) -> List[Path]:
    return sorted(p for p in Path(root).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_pil(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    array = x.detach().float().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.round(array * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")


def luminance(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def score_input(path: Path) -> CandidateScore:
    image = load_pil(str(path))
    x = pil_to_tensor(image, torch.device("cpu"))
    y = luminance(x)
    p4 = F.adaptive_avg_pool2d(y, (4, 4))[0, 0]
    border_mask = torch.ones((4, 4), dtype=torch.bool)
    border_mask[1:3, 1:3] = False
    border = p4[border_mask].mean()
    center = p4[1:3, 1:3].mean()
    return CandidateScore(
        path=str(path.resolve()),
        name=path.name,
        mean_luminance=float(y.mean().item()),
        p4_std=float(p4.std(unbiased=False).item()),
        backlit_score=float((border - center).item()),
        dark_fraction=float((y < 0.1).float().mean().item()),
    )


def select_candidates(scores: Sequence[CandidateScore], count: int) -> List[CandidateScore]:
    """Select using only input statistics, before any model inference or GT access."""
    available = list(scores)
    selected: List[CandidateScore] = []

    def take(label: str, ranked: Sequence[CandidateScore]) -> None:
        for item in ranked:
            if all(old.path != item.path for old in selected):
                item.scene = label
                selected.append(item)
                return
        raise RuntimeError(f"Unable to select a distinct candidate for {label}")

    dark_rank = sorted(available, key=lambda x: (x.mean_luminance, -x.p4_std, x.name))
    take("Overall dark", dark_rank)
    backlit_rank = sorted(
        available,
        key=lambda x: (x.backlit_score, x.p4_std, -x.mean_luminance, x.name),
        reverse=True,
    )
    take("Backlit / local dark", backlit_rank)
    means = np.asarray([item.mean_luminance for item in available], dtype=np.float64)
    lo, hi = np.quantile(means, [0.2, 0.8]).tolist()
    mid = [item for item in available if lo <= item.mean_luminance <= hi]
    spatial_rank = sorted(mid or available, key=lambda x: (x.p4_std, x.dark_fraction, x.name), reverse=True)
    take("Spatially non-uniform", spatial_rank)
    take("Overall dark (extra)", dark_rank)
    remaining_spatial = sorted(available, key=lambda x: (x.p4_std, x.dark_fraction, x.name), reverse=True)
    take("Spatially non-uniform (extra)", remaining_spatial)
    return selected[: int(count)]


def merge_train_args(cli: argparse.Namespace, stage2_blob: Dict) -> argparse.Namespace:
    args = parse_train_args([])
    saved = stage2_blob.get("args", {}) if isinstance(stage2_blob, dict) else {}
    if isinstance(saved, dict):
        for key, value in saved.items():
            if hasattr(args, key):
                setattr(args, key, value)
    args.resume = str(Path(cli.stage2_ckpt).resolve())
    args.scist_imf_ckpt = str(Path(cli.imf_ckpt).resolve())
    args.scist_priors = str(Path(cli.priors).resolve())
    args.scist = 1
    args.scist_ablation = "full"
    args.gpu = int(cli.gpu)
    args.device = str(cli.device)
    return args


def load_direct_mlp(path: str, device: torch.device) -> Tuple[DirectCorrectionMLP, Dict]:
    blob = torch.load(path, map_location="cpu")
    saved = blob.get("args", {}) if isinstance(blob, dict) else {}
    model = DirectCorrectionMLP(
        state_dim=128,
        clip_dim=int(blob.get("clip_dim", 512)),
        hidden_dim=int(saved.get("hidden_dim", 256)),
        depth=int(saved.get("depth", 4)),
    ).to(device)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, blob


def clamp_target(
    state_low: torch.Tensor,
    delta_raw: torch.Tensor,
    state_min: torch.Tensor,
    state_max: torch.Tensor,
    delta_clip: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = delta_raw
    if float(delta_clip) > 0.0:
        delta = delta.clamp(min=-float(delta_clip), max=float(delta_clip))
    target = state_low + delta
    if isinstance(state_min, torch.Tensor) and isinstance(state_max, torch.Tensor):
        target = torch.maximum(
            torch.minimum(target, state_max.to(target.device, target.dtype)),
            state_min.to(target.device, target.dtype),
        )
    return target - state_low, target


@torch.inference_mode()
def render_target(
    low: torch.Tensor,
    target: torch.Tensor,
    models: Dict[str, torch.nn.Module],
    residual: torch.Tensor,
    train_args: argparse.Namespace,
    amp: bool,
) -> torch.Tensor:
    condition = prepare_semantic_domain_condition(
        models["clip"],
        models["conditioner"],
        low,
        residual,
        clip_scale=float(train_args.clip_scale),
        target_state=target,
    )
    condition = apply_scist_condition_mask(condition, train_args)
    enabled = bool(amp and low.device.type == "cuda")
    with torch.autocast(device_type=low.device.type, dtype=torch.float16, enabled=enabled):
        result = models["model"].training_step(low_img=low, cond=condition, lambda_smooth=0.0)
    return torch.nan_to_num(result["enhanced"], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def image_stats(x: torch.Tensor) -> Dict[str, float]:
    y = luminance(x.float())
    p4 = F.adaptive_avg_pool2d(y, (4, 4))[0, 0]
    return {
        "mean_luminance": float(y.mean().item()),
        "luminance_std": float(y.std(unbiased=False).item()),
        "p4_mean": float(p4.mean().item()),
        "p4_std": float(p4.std(unbiased=False).item()),
        "p4_min": float(p4.min().item()),
        "p4_max": float(p4.max().item()),
    }


def pairwise_state_diversity(targets: Sequence[torch.Tensor]) -> float:
    values = []
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            distance = torch.linalg.vector_norm(targets[left] - targets[right], dim=1) / math.sqrt(128.0)
            values.append(float(distance.mean().item()))
    return float(np.mean(values))


def edge_map(x: torch.Tensor) -> torch.Tensor:
    y = luminance(x.float())
    gx = y[:, :, :, 1:] - y[:, :, :, :-1]
    gy = y[:, :, 1:, :] - y[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def pairwise_output_metrics(outputs: Sequence[torch.Tensor]) -> Tuple[float, float]:
    l1_values: List[float] = []
    edge_values: List[float] = []
    for left in range(len(outputs)):
        for right in range(left + 1, len(outputs)):
            a, b = outputs[left].float(), outputs[right].float()
            l1_values.append(float((a - b).abs().mean().item()))
            ea, eb = edge_map(a).flatten(), edge_map(b).flatten()
            ea, eb = ea - ea.mean(), eb - eb.mean()
            corr = (ea * eb).mean() / (ea.std(unbiased=False) * eb.std(unbiased=False) + 1e-8)
            edge_values.append(float(corr.item()))
    return float(np.mean(l1_values)), float(np.mean(edge_values))


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_figure(scene_rows: Sequence[Dict], output: Path) -> None:
    rows = len(scene_rows)
    headers = ("Input", "Direct MLP", "iMF Seed 0\n(benchmark)", "iMF Seed 1", "iMF Seed 2", "Reference")
    fig, axes = plt.subplots(rows, 6, figsize=(12.0, 1.62 * rows + 0.30), squeeze=False)
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=10.5, pad=7)
    for row, item in enumerate(scene_rows):
        arrays = [np.asarray(Image.open(item[key]).convert("RGB")) for key in (
            "input", "direct_mlp", "imf_seed0", "imf_seed1", "imf_seed2", "reference"
        )]
        for col, array in enumerate(arrays):
            axes[row, col].imshow(array)
            axes[row, col].set_box_aspect(array.shape[0] / array.shape[1])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
            if 1 <= col <= 4:
                rgb = array.astype(np.float32) / 255.0
                mean_y = float((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).mean())
                axes[row, col].text(
                    0.5,
                    -0.05,
                    rf"$\mu_Y={mean_y:.3f}$",
                    transform=axes[row, col].transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.8,
                )
        axes[row, 0].set_ylabel(item["scene"], fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.94, bottom=0.02, wspace=0.035, hspace=0.12)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    out = Path(cli.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for required in (cli.stage2_ckpt, cli.imf_ckpt, cli.direct_mlp_ckpt, cli.priors):
        if not Path(required).is_file():
            raise FileNotFoundError(required)

    # Candidate selection is deliberately completed and logged before loading any model or GT.
    all_scores = [score_input(path) for path in list_images(cli.low_root)]
    candidates = select_candidates(all_scores, int(cli.candidate_count))
    write_csv(out / "all_input_scene_scores.csv", [asdict(item) for item in all_scores])
    selection = {
        "selection_basis": "input-only mean luminance, P4 std, border-minus-center score; no model output or GT metric used",
        "selected_before_inference": True,
        "default_rows": int(cli.selected_count),
        "candidates": [asdict(item) for item in candidates],
    }
    with (out / "scene_selection_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)

    device = torch.device(
        f"cuda:{int(cli.gpu)}"
        if cli.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    stage2_blob = torch.load(cli.stage2_ckpt, map_location="cpu")
    train_args = merge_train_args(cli, stage2_blob)
    models = build_models(train_args, device=device, in_channels=3)
    load_resume_if_needed(train_args, models, device=device)
    scist = load_scist_modules(train_args, models, device=device)
    if scist is None:
        raise RuntimeError("Failed to load SCIST modules")
    direct_mlp, direct_blob = load_direct_mlp(cli.direct_mlp_ckpt, device)
    residual = stage2_blob.get("residual")
    if not isinstance(residual, torch.Tensor):
        raise KeyError("Stage-2 checkpoint has no residual")
    residual = residual.to(device)
    for module in models.values():
        module.eval()

    scene_rows: List[Dict] = []
    diversity_rows: List[Dict] = []
    stat_rows: List[Dict] = []
    delta_clip = float(scist.get("delta_clip", getattr(train_args, "scist_delta_clip", 0.0)))
    state_min, state_max = scist.get("state_min"), scist.get("state_max")

    for index, candidate in enumerate(candidates, start=1):
        stem = Path(candidate.path).stem
        scene_dir = out / "candidates" / f"{index:02d}_{stem}"
        states_dir = scene_dir / "states"
        states_dir.mkdir(parents=True, exist_ok=True)
        low_pil = load_pil(candidate.path)
        reference_path = Path(cli.gt_root) / Path(candidate.path).name
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        reference_pil = load_pil(str(reference_path))
        if reference_pil.size != low_pil.size:
            raise ValueError(f"Input/reference size mismatch for {candidate.name}: {low_pil.size} vs {reference_pil.size}")
        low = pil_to_tensor(low_pil, device)
        low_pil.save(scene_dir / "input.png")
        reference_pil.save(scene_dir / "reference.png")

        with torch.inference_mode():
            state_low = scist["state_normalizer"](scist["state_extractor"](low))
            clip_x = preprocess_for_clip(models["clip"], low).float()
            z_sem = (
                models["clip"]._encode_image_feature(clip_x).float()
                if hasattr(models["clip"], "_encode_image_feature")
                else models["clip"](low).float()
            )
            q_low = content_feature_from_clip(z_sem, residual)
            direct_raw = direct_mlp.infer_correction(state_low, q_low)
            direct_delta, direct_target = clamp_target(
                state_low, direct_raw, state_min, state_max, delta_clip
            )
            direct_output = render_target(
                low, direct_target, models, residual, train_args, bool(cli.amp)
            )
            direct_repeat = render_target(
                low, direct_target, models, residual, train_args, bool(cli.amp)
            )

        np.save(states_dir / "s_l.npy", state_low.detach().cpu().numpy())
        np.save(states_dir / "z_sem.npy", z_sem.detach().cpu().numpy())
        np.save(states_dir / "q_l.npy", q_low.detach().cpu().numpy())
        np.save(states_dir / "delta_mlp_raw.npy", direct_raw.detach().cpu().numpy())
        np.save(states_dir / "delta_mlp.npy", direct_delta.detach().cpu().numpy())
        np.save(states_dir / "target_mlp.npy", direct_target.detach().cpu().numpy())
        tensor_to_pil(direct_output).save(scene_dir / "direct_mlp.png")
        stat_rows.append({"scene": candidate.scene, "image": candidate.name, "method": "Direct MLP", **image_stats(direct_output)})

        seed_targets: List[torch.Tensor] = []
        seed_outputs: List[torch.Tensor] = []
        for seed in SEEDS:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            xi = torch.randn(state_low.shape, generator=generator, device=device, dtype=state_low.dtype)
            with torch.inference_mode():
                raw_delta = scist["imf"].infer_correction(state_low, q_low, noise=xi)
                delta, target = clamp_target(state_low, raw_delta, state_min, state_max, delta_clip)
                output = render_target(low, target, models, residual, train_args, bool(cli.amp))
            np.save(states_dir / f"xi_seed{seed}.npy", xi.detach().cpu().numpy())
            np.save(states_dir / f"delta_seed{seed}_raw.npy", raw_delta.detach().cpu().numpy())
            np.save(states_dir / f"delta_seed{seed}.npy", delta.detach().cpu().numpy())
            np.save(states_dir / f"target_seed{seed}.npy", target.detach().cpu().numpy())
            tensor_to_pil(output).save(scene_dir / f"imf_seed{seed}.png")
            stat_rows.append({"scene": candidate.scene, "image": candidate.name, "method": f"iMF Seed {seed}", **image_stats(output)})
            seed_targets.append(target.detach())
            seed_outputs.append(output.detach())

        output_l1, edge_corr = pairwise_output_metrics(seed_outputs)
        diversity_rows.append(
            {
                "scene": candidate.scene,
                "image": candidate.name,
                "d_state": pairwise_state_diversity(seed_targets),
                "output_pairwise_l1": output_l1,
                "edge_correlation": edge_corr,
                "direct_repeat_max_abs": float((direct_output - direct_repeat).abs().max().item()),
                "width": low_pil.width,
                "height": low_pil.height,
            }
        )
        paths = {
            "scene": candidate.scene,
            "image": candidate.name,
            "input": str(scene_dir / "input.png"),
            "direct_mlp": str(scene_dir / "direct_mlp.png"),
            "imf_seed0": str(scene_dir / "imf_seed0.png"),
            "imf_seed1": str(scene_dir / "imf_seed1.png"),
            "imf_seed2": str(scene_dir / "imf_seed2.png"),
            "reference": str(scene_dir / "reference.png"),
        }
        scene_rows.append(paths)
        render_figure([paths], scene_dir / "panel.png")
        print(f"[Fig7] {index}/{len(candidates)} {candidate.scene}: {candidate.name}")

    write_csv(out / "diversity_metrics.csv", diversity_rows)
    write_csv(out / "photometric_statistics.csv", stat_rows)
    render_figure(scene_rows[: int(cli.selected_count)], out / "fig7_target_diversity.png")
    render_figure(scene_rows, out / "fig7_all_5_candidates.png")
    shutil.copy2(out / "fig7_target_diversity.pdf", out / "fig7_target_diversity_vector.pdf")

    protocol = {
        "seeds": list(SEEDS),
        "benchmark_seed": 0,
        "scene_selection": selection,
        "direct_mlp_checkpoint": str(Path(cli.direct_mlp_ckpt).resolve()),
        "direct_mlp_epoch": int(direct_blob.get("epoch", 0)),
        "imf_checkpoint": str(Path(cli.imf_ckpt).resolve()),
        "stage2_checkpoint": str(Path(cli.stage2_ckpt).resolve()),
        "renderer_control": "Direct MLP and all iMF seeds use the same frozen Stage-2 enhancer to isolate target-model diversity.",
        "target_clamp": {"delta_clip": delta_clip, "normal_state_quantile": float(getattr(train_args, "scist_state_clip_quantile", 0.0))},
    }
    with (out / "experiment_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    mean_d = float(np.mean([row["d_state"] for row in diversity_rows]))
    mean_edge = float(np.mean([row["edge_correlation"] for row in diversity_rows]))
    results = {
        "protocol": protocol,
        "aggregate": {
            "num_candidates": len(diversity_rows),
            "mean_d_state": mean_d,
            "mean_output_pairwise_l1": float(np.mean([row["output_pairwise_l1"] for row in diversity_rows])),
            "mean_edge_correlation": mean_edge,
            "max_direct_repeat_abs": float(max(row["direct_repeat_max_abs"] for row in diversity_rows)),
        },
        "per_scene": diversity_rows,
    }
    with (out / "fig7_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (out / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Fig. 7 Conditional Target Diversity\n\n")
        handle.write("The first three preselected rows form the paper figure; two additional rows are provided for visual selection.\n\n")
        handle.write(f"Seeds are fixed to `{list(SEEDS)}`; Seed 0 is the benchmark seed. Mean target-state diversity over five candidates is `{mean_d:.6f}` and mean inter-seed edge correlation is `{mean_edge:.6f}`.\n\n")
        handle.write("Direct MLP and iMF use the same OT protocol. The frozen Stage-2 renderer is shared across methods so this figure isolates deterministic versus stochastic target-state modeling; this is a controlled visualization, not a separately retrained end-to-end performance comparison.\n\n")
        handle.write("All outputs retain the native input/reference dimensions. No per-image or per-seed brightness post-processing is applied.\n")
    with (out / "paper_text.md").open("w", encoding="utf-8") as handle:
        handle.write("# Fig. 7 paper-ready text\n\n")
        handle.write("Conditional target diversity on representative LOLv1 test images. For each low-light input, the deterministic Direct MLP baseline produces a single photometric realization, whereas conditional iMF generates different target states and corresponding enhancement results from fixed latent seeds. Seed 0 is the fixed latent used for the main benchmark, while Seeds 1 and 2 are shown only to visualize conditional variation. The paired normal-light image is included as one valid reference realization and is not used for latent or scene selection. Direct MLP and iMF outputs are rendered by the same frozen Stage-2 enhancer to isolate target-model diversity.\n")
    print(json.dumps({"device": str(device), "mean_d_state": mean_d, "mean_edge_correlation": mean_edge, "candidates": [item.name for item in candidates]}, indent=2))


if __name__ == "__main__":
    main()
