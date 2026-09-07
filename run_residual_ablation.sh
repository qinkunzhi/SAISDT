#!/usr/bin/env bash

# Residual Guidance ablation runner for RL-DiffNet
# 使用方法：
#   1) 进入本目录：cd comparision/RL-DiffNet-main
#   2) 赋予执行权限：chmod +x run_residual_ablation.sh
#   3) 运行全部实验：./run_residual_ablation.sh all
#      只跑某一组： ./run_residual_ablation.sh E1
# 可通过环境变量覆盖默认参数，例如：
#   GPU=1 EPOCHS=40 BATCH_SIZE=4 SAVE_ROOT=train6_res ./run_residual_ablation.sh all

set -e

# ===== 通用可配置参数（可用环境变量覆盖） =====
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-4}
TIMESTEPS=${TIMESTEPS:-1000}
SAVE_ROOT=${SAVE_ROOT:-train6_residual}

# 数据路径（默认与 train.py 中保持一致，如有变动请在此修改）
LOW_ROOT=${LOW_ROOT:-/home/zhiqinkun/LLIM/datasets/data/LOLv1/Train/input}
HIGH_ROOT=${HIGH_ROOT:-/home/zhiqinkun/LLIM/datasets/data/DIV2K_384}
GT_ROOT=${GT_ROOT:-/home/zhiqinkun/LLIM/datasets/data/LOLv1/Train/target}

BASE_ARGS=(
  --gpu "${GPU}"
  --device cuda
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --timesteps "${TIMESTEPS}"
  --low_root "${LOW_ROOT}"
  --high_root "${HIGH_ROOT}"
  --gt_root "${GT_ROOT}"
  # 评估时保持与训练分布一致（可按需要关掉）
  --eval_match_train_noise
)

run_exp() {
  local id="$1"; shift
  local name="$1"; shift
  local save_dir="${SAVE_ROOT}/${id}_${name}"
  local extra_args=("$@")

  echo "=============================="
  echo "[RUN] Experiment ${id} - ${name}"
  echo "Save dir: ${save_dir}"
  echo "Extra args: ${extra_args[*]}"
  echo "=============================="

  python train.py "${BASE_ARGS[@]}" --save_dir "${save_dir}" "${extra_args[@]}"
}

exp_E0() {
  # E0: 无 Residual Guidance（对照组，仅使用 CLIP 低光语义特征）
  run_exp E0 noResidual
}

exp_E1() {
  # E1: Residual Guidance + residual projection loss（RGB 上），默认权重 0.1
  run_exp E1 res_rgb \
    --use_residual_guidance \
    --residual_loss_on rgb \
    --residual_loss_weight 0.1
}

exp_E2() {
  # E2: Residual Guidance（在 Luma 上计算投影 loss），抑制色偏
  run_exp E2 res_luma \
    --use_residual_guidance \
    --residual_loss_on luma \
    --residual_loss_weight 0.1
}

exp_E3() {
  # E3: Residual Guidance + 内容偏置去除（residual_remove_content_bias）
  run_exp E3 res_contentBiasOff \
    --use_residual_guidance \
    --residual_remove_content_bias \
    --residual_content_bias_topk 64 \
    --residual_loss_on luma \
    --residual_loss_weight 0.1
}

exp_E4() {
  # E4: Residual Guidance + RAVE 风格 token 极值移除（residual_remove_first_n_tokens）
  #    数值 400 参考典型 RAVE 设置，可按需调整。
  run_exp E4 res_tokenShift \
    --use_residual_guidance \
    --residual_remove_first_n_tokens 400 \
    --residual_loss_on luma \
    --residual_loss_weight 0.1
}

exp_E5() {
  # E5: Residual Guidance + 可学习 bias（ResidualGuidanceBias），以 mean_neg 初始化
  run_exp E5 res_learnableBias \
    --use_residual_guidance \
    --residual_use_learnable_bias \
    --residual_bias_init mean_neg \
    --residual_loss_on luma \
    --residual_loss_weight 0.1
}

exp_E6() {
  # E6: Residual Guidance，增大 residual projection 权重（0.3，对比 E2 的 0.1）
  run_exp E6 res_strongLoss \
    --use_residual_guidance \
    --residual_loss_on luma \
    --residual_loss_weight 0.3
}

run_all() {
  exp_E0
  exp_E1
  exp_E2
  exp_E3
  exp_E4
  exp_E5
  exp_E6
}

main() {
  local which="$1"
  case "${which}" in
    ""|all)
      run_all
      ;;
    E0)
      exp_E0 ;;
    E1)
      exp_E1 ;;
    E2)
      exp_E2 ;;
    E3)
      exp_E3 ;;
    E4)
      exp_E4 ;;
    E5)
      exp_E5 ;;
    E6)
      exp_E6 ;;
    *)
      echo "Unknown experiment id: ${which}"
      echo "Usage: $0 [all|E0|E1|E2|E3|E4|E5|E6]" >&2
      exit 1
      ;;
  esac
}

main "$@"
