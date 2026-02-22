#!/usr/bin/env bash
set -u  # 不 set -e：训练崩了要继续循环

RUN_ROOT="/home/ubuntu/runs/openvla"
DATA_ROOT="/home/ubuntu/16831pro_fine_tune/openvla-oft/datasets/modified_libero_rlds"

DATASET_NAME="libero_goal_no_noops"

GPU_ID=1
SAVE_FREQ=100
VAL_FREQ=100000
SLEEP_SECS=10

KEEP_LAST_N=3   # 只保留最近N个ckpt，防止爆盘（建议3~5）

# ---- 完整性判定：按你截图的 ckpt 结构（4 shards + index + action_head + lora_adapter）----
is_complete_ckpt_dir () {
  local d="$1"

  [[ -s "${d}/config.json" ]] || return 1
  [[ -s "${d}/model.safetensors.index.json" ]] || return 1

  for i in 1 2 3 4; do
    local shard
    shard=$(printf "model-%05d-of-00004.safetensors" "$i")
    [[ -s "${d}/${shard}" ]] || return 1
  done

  local step
  step=$(echo "$d" | sed -n 's/.*--\([0-9]\+\)_chkpt/\1/p')
  [[ -n "${step}" ]] || return 1
  [[ -s "${d}/action_head--${step}_checkpoint.pt" ]] || return 1

  [[ -s "${d}/lora_adapter/adapter_config.json" ]] || return 1
  if [[ -s "${d}/lora_adapter/adapter_model.safetensors" ]]; then
    :
  elif [[ -s "${d}/lora_adapter/adapter_model.bin" ]]; then
    :
  else
    return 1
  fi

  return 0
}

extract_step () {
  local d="$1"
  echo "$d" | sed -n 's/.*--\([0-9]\+\)_chkpt/\1/p'
}

list_ckpts_sorted () {
  ls -d "${RUN_ROOT}"/*_chkpt 2>/dev/null | sort -V || true
}

latest_complete_ckpt () {
  local dirs
  dirs=$(list_ckpts_sorted)
  [[ -z "${dirs}" ]] && echo "" && return 0

  for d in $(echo "${dirs}" | tac); do
    if is_complete_ckpt_dir "$d"; then
      echo "$d"
      return 0
    else
      echo "===== $(date) : Incomplete ckpt, skip: ${d}" >&2
    fi
  done

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

  CKPT_DIR=$(latest_complete_ckpt)
  if [[ -z "${CKPT_DIR}" ]]; then
    echo "===== $(date) : No COMPLETE checkpoint found under ${RUN_ROOT}. ====="
    exit 1
  fi

  RESUME_STEP=$(extract_step "${CKPT_DIR}")
  if [[ -z "${RESUME_STEP}" ]]; then
    echo "===== $(date) : Failed to parse step from ${CKPT_DIR} ====="
    exit 1
  fi

  echo "===== $(date) : Using ckpt=${CKPT_DIR} (resume_step=${RESUME_STEP}) ====="

  
  delete_old_ckpts_keep_last_n "${KEEP_LAST_N}"

  echo "===== $(date) : Starting training ====="

 
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "${CKPT_DIR}" \
    --data_root_dir "${DATA_ROOT}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${RUN_ROOT}" \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 1 \
    --grad_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project "openvla" \
    --wandb_entity "maggiesh-carnegie-mellon-university" \
    --save_freq "${SAVE_FREQ}" \
    --resume True \
    --resume_step "${RESUME_STEP}"
  


  code=$?
  echo "===== $(date) : Training exited with code ${code}. Restarting in ${SLEEP_SECS}s... ====="
  sleep "${SLEEP_SECS}"
done
