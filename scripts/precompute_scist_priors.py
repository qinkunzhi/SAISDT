import argparse
import os
import sys
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.classifier import build_classifier
from models.illumination_state import IlluminationStateExtractor, fit_state_normalizer
from utils.clip_domain import (
    ResidualDomainConfig,
    load_or_compute_residual_domain,
    preprocess_for_clip,
)
from utils.semantic_ot import content_feature_from_clip


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def load_batch(paths: List[str], size: int, device: torch.device) -> torch.Tensor:
    xs = []
    resize = transforms.Resize((int(size), int(size)))
    to_tensor = transforms.ToTensor()
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append(resize(to_tensor(img)))
    return torch.stack(xs, dim=0).to(device)


@torch.no_grad()
def encode_domain(
    paths: List[str],
    clip,
    state_extractor: IlluminationStateExtractor,
    residual: torch.Tensor,
    device: torch.device,
    batch_size: int,
    image_size: int,
    max_images: int,
    label: str,
) -> Dict[str, torch.Tensor]:
    if int(max_images) > 0:
        paths = paths[: int(max_images)]
    states = []
    z_list = []
    q_list = []
    names = []
    for start in range(0, len(paths), int(batch_size)):
        batch_paths = paths[start : start + int(batch_size)]
        x = load_batch(batch_paths, int(image_size), device)
        state = state_extractor(x).cpu()
        clip_x = preprocess_for_clip(clip, x).float()
        z = clip._encode_image_feature(clip_x).float() if hasattr(clip, "_encode_image_feature") else clip(x).float()
        z = F.normalize(z, p=2, dim=-1, eps=1e-6)
        q = content_feature_from_clip(z, residual.to(device))
        states.append(state)
        z_list.append(z.cpu())
        q_list.append(q.cpu())
        names.extend([os.path.basename(p) for p in batch_paths])
        print(f"[{label}] {min(start + len(batch_paths), len(paths))}/{len(paths)}")
    return {
        "paths": paths,
        "names": names,
        "state_raw": torch.cat(states, dim=0),
        "z_sem": torch.cat(z_list, dim=0),
        "q": torch.cat(q_list, dim=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCIST Stage 0 prior precomputation.")
    parser.add_argument("--low_root", type=str, default="/home/zhiqinkun/LLIM/datasets/data/LOLv1/Train/input")
    parser.add_argument("--high_root", type=str, default="/home/zhiqinkun/LLIM/datasets/data/DIV2K_384")
    parser.add_argument("--out", type=str, default="scist_priors.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_low_images", type=int, default=0)
    parser.add_argument("--max_high_images", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--clip_gamma", type=float, default=0.5)
    parser.add_argument("--clip_image_size", type=int, default=224)
    parser.add_argument("--residual_cache_dir", type=str, default=".cache_residual")
    parser.add_argument("--residual_max_images", type=int, default=2000)
    parser.add_argument("--residual_recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{int(args.gpu)}" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    clip = build_classifier(
        feature_dim=None,
        gamma=float(args.clip_gamma),
        image_size=int(args.clip_image_size),
        clip_model_name=str(args.clip_model_name),
        clip_pretrained=str(args.clip_pretrained),
    ).to(device).eval()
    for p in clip.parameters():
        p.requires_grad = False
    state_extractor = IlluminationStateExtractor().to(device).eval()

    residual, mean_pos, mean_neg = load_or_compute_residual_domain(
        clip,
        ResidualDomainConfig(
            pos_root=str(args.high_root),
            neg_root=str(args.low_root),
            max_images=int(args.residual_max_images),
            cache_dir=str(args.residual_cache_dir),
            force_recompute=bool(args.residual_recompute),
            show_progress=True,
        ),
        device=device,
    )

    low_paths = list_images(str(args.low_root))
    high_paths = list_images(str(args.high_root))
    if not low_paths:
        raise RuntimeError(f"No low-light images found: {args.low_root}")
    if not high_paths:
        raise RuntimeError(f"No normal-light images found: {args.high_root}")

    low = encode_domain(low_paths, clip, state_extractor, residual, device, int(args.batch_size), int(args.image_size), int(args.max_low_images), "LOW")
    high = encode_domain(high_paths, clip, state_extractor, residual, device, int(args.batch_size), int(args.image_size), int(args.max_high_images), "HIGH")

    stats = fit_state_normalizer(torch.cat([low["state_raw"], high["state_raw"]], dim=0))
    state_mean = stats["state_mean"]
    state_std = stats["state_std"]
    low["state"] = (low["state_raw"] - state_mean.view(1, -1)) / (state_std.view(1, -1) + 1e-6)
    high["state"] = (high["state_raw"] - state_mean.view(1, -1)) / (state_std.view(1, -1) + 1e-6)

    torch.save(
        {
            "low_root": str(args.low_root),
            "high_root": str(args.high_root),
            "state_mean": state_mean,
            "state_std": state_std,
            "low": low,
            "high": high,
            "residual": residual.detach().cpu(),
            "mean_pos": mean_pos.detach().cpu(),
            "mean_neg": mean_neg.detach().cpu(),
            "clip_dim": int(getattr(clip, "out_dim")),
            "clip_model_name": str(args.clip_model_name),
            "clip_pretrained": str(args.clip_pretrained),
            "clip_gamma": float(args.clip_gamma),
            "clip_image_size": int(args.clip_image_size),
        },
        str(args.out),
    )
    print(f"[SCIST] Saved priors to {args.out}")
    print(f"[SCIST] low={low['state'].shape} high={high['state'].shape} state_dim={state_mean.numel()} clip_dim={int(getattr(clip, 'out_dim'))}")


if __name__ == "__main__":
    main()
