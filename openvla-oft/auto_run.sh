#!/usr/bin/env bash
set -u  

RUN_ROOT="/home/ubuntu/runs/openvla"
# adapter 专用：列举/保存都在这里，和 pretrained base 分开
ADAPTER_RUN_ROOT="/home/ubuntu/runs/openvla_adapters"
DATA_ROOT="/home/ubuntu/16831pro_fine_tune/openvla-oft/datasets/masked_libero_rlds"

DATASET_NAME="libero_goal_no_noops"

# pretrained base：官方 openvla-7b（已下载）；如需用之前 OFT 2200 可改回下一行
BASE_VLA_PATH="${RUN_ROOT}/openvla-7b"
# BASE_VLA_PATH="${RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--2200_chkpt"

# 单卡：只有一张 GPU 时用 0，多卡时指定卡号
GPU_ID=0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
SAVE_FREQ=750
VAL_FREQ=100000
SLEEP_SECS=10

# resume 时用的学习率（finetune.py 不加载 optimizer 状态，每次都用当前 --learning_rate 新建 optimizer，所以这里改了就生效）
LEARNING_RATE=5e-5

# 训练时每隔 N 步保存 dataset 图片到 openvla-oft/dataset_debug/
export DATASET_DEBUG_SAVE=1
export DATASET_DEBUG_SAVE_EVERY=5000

KEEP_LAST_N=20   # 只保留最近N个ckpt，防止爆盘（建议3~5）

# ---- --fast_model：LoRA r=8、只 LoRA attention（qkv+proj）、action head + proprio，proprio 用更低 LR
LORA_RANK=32
LORA_TARGET_MODULES=""
PROPRIO_PROJECTOR_LR=""
for a in "$@"; do
  if [[ "$a" == "--fast_model" ]]; then
    LORA_RANK=8
    LORA_TARGET_MODULES="attn-only"
    PROPRIO_PROJECTOR_LR="2e-5"
    break
  fi
done

# ---- 完整性判定：仅存 adapter 不 merge 时，只要求 lora_adapter + action_head + processor ----
# 若失败会向 stderr 打印缺了哪些文件（方便排查）
is_complete_ckpt_dir () {
  local d="$1"
  local step
  step=$(echo "$d" | sed -n 's/.*--\([0-9]\+\)_chkpt/\1/p')
  if [[ -z "${step}" ]]; then
    echo "  [incomplete] no step in dir name" >&2
    return 1
  fi
  if [[ ! -s "${d}/config.json" ]]; then
    echo "  [incomplete] missing or empty: config.json" >&2
    return 1
  fi
  local ah="${d}/action_head--${step}_checkpoint.pt"
  if [[ ! -s "${ah}" ]]; then
    echo "  [incomplete] missing or empty: action_head--${step}_checkpoint.pt (checked ${ah})" >&2
    return 1
  fi
  if [[ ! -s "${d}/lora_adapter/adapter_config.json" ]]; then
    echo "  [incomplete] missing or empty: lora_adapter/adapter_config.json" >&2
    return 1
  fi
  if [[ -s "${d}/lora_adapter/adapter_model.safetensors" ]]; then
    :
  elif [[ -s "${d}/lora_adapter/adapter_model.bin" ]]; then
    :
  else
    echo "  [incomplete] missing: lora_adapter/adapter_model.safetensors or adapter_model.bin" >&2
    return 1
  fi
  return 0
}

extract_step () {
  local d="$1"
  echo "$d" | sed -n 's/.*--\([0-9]\+\)_chkpt/\1/p'
}

list_ckpts_sorted () {
  ls -d "${ADAPTER_RUN_ROOT}"/*_chkpt 2>/dev/null | sort -V || true
}

# 按 step 降序选最新且完整的 ckpt（只扫 ADAPTER_RUN_ROOT）
latest_complete_ckpt () {
  local d step
  while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    step=$(extract_step "$d")
    [[ -z "$step" ]] && continue
    if is_complete_ckpt_dir "$d"; then
      echo "$d"
      return 0
    else
      echo "===== $(date) : Incomplete ckpt, skip: ${d} (step=${step}) =====" >&2
    fi
  done < <(list_ckpts_sorted | while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    step=$(extract_step "$d")
    printf "%d\t%s\n" "${step:-0}" "$d"
  done | sort -t$'\t' -k1 -n -r | cut -f2-)
  echo ""
}

delete_old_ckpts_keep_last_n () {
  local keep_n="$1"
  local dirs
  dirs=$(list_ckpts_sorted)
  [[ -z "${dirs}" ]] && return 0

  local count
  count=$(echo "${dirs}" | wc -l | tr -d ' ')
  if (( count <= keep_n )); then
    return 0
  fi

  local to_delete
  to_delete=$(echo "${dirs}" | head -n $((count - keep_n)))

  echo "${to_delete}" | while read -r d; do
    [[ -z "${d}" ]] && continue
    echo "===== $(date) : Deleting old ckpt: ${d} ====="
    rm -rf "${d}"
  done
}

while true; do
  echo "===== $(date) : Selecting latest COMPLETE checkpoint ====="

  # 补救：adapter 目录有 lora_adapter 但缺 config.json 时，从 base 拷一份（同 run 的 config 一致）
  for d in "${ADAPTER_RUN_ROOT}"/*_chkpt; do
    [[ -d "$d" ]] || continue
    if [[ -d "${d}/lora_adapter" && ! -s "${d}/config.json" && -s "${BASE_VLA_PATH}/config.json" ]]; then
      echo "===== $(date) : Copying config.json from base into ${d} ====="
      cp -f "${BASE_VLA_PATH}/config.json" "${d}/"
      [[ -s "${BASE_VLA_PATH}/preprocessor_config.json" ]] && cp -f "${BASE_VLA_PATH}/preprocessor_config.json" "${d}/" 2>/dev/null || true
    fi
  done

  CKPT_DIR=$(latest_complete_ckpt)
  RESUME_STEP=""
  VLA_PATH="${BASE_VLA_PATH}"
  RESUME_ARG="--resume False"

  if [[ -n "${CKPT_DIR}" ]]; then
    RESUME_STEP=$(extract_step "${CKPT_DIR}")
    if [[ -n "${RESUME_STEP}" ]]; then
      VLA_PATH="${CKPT_DIR}"
      RESUME_ARG="--resume True"
      echo "===== $(date) : Using ckpt=${CKPT_DIR} (resume_step=${RESUME_STEP}) ====="
      delete_old_ckpts_keep_last_n "${KEEP_LAST_N}"
    fi
  fi

  if [[ -z "${RESUME_STEP}" ]]; then
    echo "===== $(date) : No COMPLETE checkpoint found under ${ADAPTER_RUN_ROOT} -> COLD START from base ${BASE_VLA_PATH} ====="
  fi

  if [[ -n "${PROPRIO_PROJECTOR_LR:-}" ]]; then
    echo "===== $(date) : [fast_model] lora_rank=${LORA_RANK}, lora_target=${LORA_TARGET_MODULES}, proprio_projector_lr=${PROPRIO_PROJECTOR_LR} ====="
  fi

  echo "===== $(date) : Starting training (learning_rate=${LEARNING_RATE}) ${RESUME_ARG} ====="

  EXTRA_OPTS=()
  [[ -n "${LORA_TARGET_MODULES:-}" ]] && EXTRA_OPTS+=(--lora_target_modules "${LORA_TARGET_MODULES}")
  [[ -n "${PROPRIO_PROJECTOR_LR:-}" ]] && EXTRA_OPTS+=(--proprio_projector_lr "${PROPRIO_PROJECTOR_LR}")

  RESUME_STEP_OPTS=()
  [[ -n "${RESUME_STEP:-}" ]] && RESUME_STEP_OPTS=(--resume_step "${RESUME_STEP}")

  PYTHONUNBUFFERED=1 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "${VLA_PATH}" \
    --base_vla_path "${BASE_VLA_PATH}" \
    --data_root_dir "${DATA_ROOT}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${ADAPTER_RUN_ROOT}" \
    --use_lora True \
    --lora_rank "${LORA_RANK}" \
    "${EXTRA_OPTS[@]}" \
    --merge_lora_during_training False \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --learning_rate "${LEARNING_RATE}" \
    --image_aug False \
    --wandb_project "openvla_gripper_proprio_fast" \
    --wandb_entity "maggiesh-carnegie-mellon-university" \
    --save_freq "${SAVE_FREQ}" \
    ${RESUME_ARG} \
    "${RESUME_STEP_OPTS[@]}" \
    --use_proprio True \
    --use_l1_regression True \
    --resume True
 
 
  


  code=$?
  echo "===== $(date) : Training exited with code ${code}. Restarting in ${SLEEP_SECS}s... ====="
  sleep "${SLEEP_SECS}"
done
