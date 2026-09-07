#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"

python scripts/run_fig6_state_validation.py \
  --out_dir ablation_main/fig_6 \
  --gpu "${GPU_ID}" \
  --batch_size 16 \
  --candidate_count 5 \
  --selected_count 3
