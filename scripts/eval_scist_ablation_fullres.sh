#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhiqinkun/LLIM/comparision/RL-DiffNet-main"
PYTHON_BIN="${PYTHON_BIN:-/home/zhiqinkun/anaconda3/envs/dl_env/bin/python}"
GPU_ID="${GPU_ID:-1}"
SAMPLE_IDS="${SAMPLE_IDS:-219,64,344}"
WO_BIAS_CKPT="${WO_BIAS_CKPT:-}"

cd "$REPO_ROOT"

run_eval() {
  local variant="$1"
  local directory="$2"
  local checkpoint="$3"
  "$PYTHON_BIN" scripts/eval_scist_ablation.py \
    --ckpt "$checkpoint" \
    --out_dir "$REPO_ROOT/ablation_main/$directory" \
    --scist_ablation "$variant" \
    --scist_imf_ckpt "$REPO_ROOT/train_scist_imf/illumination_imf_epoch_100.pt" \
    --scist_priors "$REPO_ROOT/scist_priors.pt" \
    --sample_ids "$SAMPLE_IDS" \
    --eval_size 0 \
    --eval_amp 1 \
    --batch_size 1 \
    --gpu "$GPU_ID"
}

run_eval clip_only clip_only \
  "$REPO_ROOT/ablation_main/clip_only/epoch045_periodic_PSNR15.9049_SSIM0.6939_NIQE5.1505_MUSIQ49.9584/illum_diff_epoch_45.pth"
run_eval target_only target_state_only \
  "$REPO_ROOT/ablation_main/target_state_only/epoch048_periodic_PSNR17.8616_SSIM0.7386_NIQE5.2099_MUSIQ50.1156/illum_diff_epoch_48.pth"
run_eval wo_gsf wo_gsf \
  "$REPO_ROOT/ablation_main/wo_gsf/epoch044_best_PSNR18.1328_SSIM0.7483_NIQE5.2998_MUSIQ50.2061/illum_diff_epoch_44.pth"
run_eval wo_lstate wo_lstate \
  "$REPO_ROOT/ablation_main/wo_lstate/epoch045_periodic_PSNR15.9126_SSIM0.6943_NIQE5.1707_MUSIQ49.8224/illum_diff_epoch_45.pth"
run_eval full full \
  "$REPO_ROOT/ablation_main/full/epoch036_best_PSNR18.4125_SSIM0.7417_NIQE5.2617_MUSIQ50.1562/illum_diff_epoch_36.pth"

if [[ -n "$WO_BIAS_CKPT" ]]; then
  run_eval wo_bias_suppression wo_bias_suppression "$WO_BIAS_CKPT"
else
  echo "[EvalAblation] WO_BIAS_CKPT is empty; skipped w/o bias suppression."
fi
