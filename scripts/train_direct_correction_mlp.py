#!/usr/bin/env python3
"""Train the deterministic Fig. 7 baseline on the same OT protocol as SCIST iMF."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.direct_correction_mlp import DirectCorrectionMLP
from train_state_imf import SCISTPriorDataset
from utils.semantic_ot import SemanticOTCoupler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deterministic OT-correction MLP for Fig. 7.")
    parser.add_argument("--priors", default="scist_priors.pt")
    parser.add_argument("--imf_ckpt", default="train_scist_imf/illumination_imf_epoch_100.pt")
    parser.add_argument("--out_dir", default="ablation_main/fig_7/direct_mlp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{int(args.gpu)}"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    dataset = SCISTPriorDataset(args.priors)
    loader_generator = torch.Generator().manual_seed(int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=loader_generator,
    )
    imf_blob = torch.load(args.imf_ckpt, map_location="cpu")
    imf_args = imf_blob.get("args", {}) if isinstance(imf_blob, dict) else {}
    model = DirectCorrectionMLP(
        state_dim=128,
        clip_dim=int(dataset.clip_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=float(args.lr), betas=(0.9, 0.999), weight_decay=1e-4
    )
    coupler = SemanticOTCoupler(
        lambda_sem=float(imf_args.get("ot_lambda_sem", 1.0)),
        lambda_lay=float(imf_args.get("ot_lambda_lay", 0.2)),
        eta=float(imf_args.get("ot_eta", 0.5)),
        epsilon=float(imf_args.get("ot_epsilon", 0.05)),
        iterations=int(imf_args.get("ot_iters", 50)),
        topk=int(imf_args.get("ot_topk", 4)),
        sample=True,
        transport_mode=str(imf_args.get("ot_transport_mode", "sinkhorn")),
        gibbs_temperature=float(imf_args.get("ot_gibbs_temperature", 1.0)),
    )
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        steps = 0
        for state_low, q_low, state_high_pool, q_high_pool in loader:
            state_low = state_low.to(device)
            q_low = q_low.to(device)
            state_high_pool = state_high_pool.to(device)
            q_high_pool = q_high_pool.to(device)
            state_low, state_high, q_low, _idx = coupler.pair(
                state_low, state_high_pool, q_low, q_high_pool
            )
            target_delta = state_high - state_low
            predicted_delta = model(state_low, q_low)
            loss = F.mse_loss(predicted_delta, target_delta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().item())
            steps += 1
        mean_loss = loss_sum / max(1, steps)
        history.append({"epoch": epoch, "mse": mean_loss})
        print(
            f"[DirectMLP] epoch={epoch:03d}/{int(args.epochs)} "
            f"mse={mean_loss:.6f} time={time.perf_counter()-start:.2f}s"
        )
        if epoch == 1 or epoch % int(args.save_every) == 0 or epoch == int(args.epochs):
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "args": vars(args),
                "clip_dim": int(dataset.clip_dim),
                "state_mean": dataset.data["state_mean"],
                "state_std": dataset.data["state_std"],
                "ot_protocol": {
                    "lambda_sem": coupler.lambda_sem,
                    "lambda_lay": coupler.lambda_lay,
                    "eta": coupler.eta,
                    "epsilon": coupler.epsilon,
                    "iterations": coupler.iterations,
                    "topk": coupler.topk,
                    "transport_mode": coupler.transport_mode,
                    "gibbs_temperature": coupler.gibbs_temperature,
                },
            }
            torch.save(checkpoint, out / f"direct_mlp_epoch_{epoch}.pt")
            torch.save(checkpoint, out / "direct_mlp_latest.pt")
    with (out / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    summary = {
        "device": str(device),
        "num_low_states": len(dataset),
        "num_high_states": int(dataset.high_state.shape[0]),
        "epochs": int(args.epochs),
        "final_mse": history[-1]["mse"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "imf_parameter_count": sum(v.numel() for v in imf_blob["model_state"].values()),
        "shared_ot_protocol": checkpoint["ot_protocol"],
    }
    with (out / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
