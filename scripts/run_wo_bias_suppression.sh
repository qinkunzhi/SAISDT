#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhiqinkun/LLIM/comparision/RL-DiffNet-main"
PYTHON_BIN="${PYTHON_BIN:-/home/zhiqinkun/anaconda3/envs/dl_env/bin/python}"
GPU_ID="${GPU_ID:-0}"

cd "$REPO_ROOT"

exec "$PYTHON_BIN" train_illum.py \
  --stage real_adapt \
  --scist 1 \
  --scist_ablation wo_bias_suppression \
  --scist_imf_ckpt "$REPO_ROOT/train_scist_imf/illumination_imf_epoch_100.pt" \
  --scist_priors "$REPO_ROOT/scist_priors.pt" \
  --epochs 50 \
  --batch_size 4 \
  --lr 5e-6 \
  --gpu "$GPU_ID" \
  --seed 123 \
  --save_random_seed 123 \
  --save_dir "$REPO_ROOT/ablation_main/wo_bias_suppression" \
  --lambda_diffusion 0.0 \
  --lambda_recon 0.0 \
  --lambda_correction 0.0 \
  --lambda_teacher_zero_ref 0.0 \
  --lambda_refiner_aux 0.0 \
  --lambda_inverse 0.0 \
  --lambda_noise 0.0 \
  --lambda_low_snr_chroma 0.8 \
  --lambda_normal_stat 0.02 \
  --lambda_quality_stat 0.03
