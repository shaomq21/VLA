#!/bin/bash
# grpo_lite: Ultra-light GRPO, batch-mean baseline, minimal memory. Separate from SimpleVLA-RL.
# Usage: cd /home/ubuntu/16831pro_fine_tune/grpo_lite_rl && bash run_libero_push_perturb.sh
set -x

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export RAY_memory_usage_threshold=0.99
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=LIBERO

GRPO_LITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${GRPO_LITE_ROOT}/.." && pwd)"
export PYTHONPATH="${GRPO_LITE_ROOT}:${BASE_DIR}/LIBERO:${BASE_DIR}/openvla-oft:${PYTHONPATH:-}"

CKPT_DIR="/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt"
BASE_VLA_PATH="${GRPO_LITE_ROOT}/../openvla-oft/checkpoints/openvla-7b"
export CKPT_PATH="/home/ubuntu/runs/grpo_lite_rl_push_perturb/ckpt"
export ROLLOUT_DIR="/home/ubuntu/runs/grpo_lite_rl_push_perturb/rollout"

PROJECT_NAME="RL"
EXPERIMENT_NAME="grpo_lite_libero_push_perturb"
DATASET_NAME="libero_goal"
VLA_NAME="openvla-oft"
NUM_GPUS=1
NUM_NODES=1

SCRIPT_DIR="${GRPO_LITE_ROOT}/../SimpleVLA-RL"
ALIGN_PATH="${SCRIPT_DIR}/align.json"
[ ! -f "$ALIGN_PATH" ] && ALIGN_PATH="${SCRIPT_DIR}/verl/trainer/runtime_env.yaml"
bash "${SCRIPT_DIR}/examples/overwrite_vla_ckpt_utils.sh" "$CKPT_DIR"

LOG_DIR="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_$(date +%Y-%m-%d_%H-%M-%S).log"

# grpo_lite: minimal memory - smaller batches, lower vLLM mem
HYDRA_FULL_ERROR=1 python -u "${GRPO_LITE_ROOT}/launcher.py" \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=1 \
    +data.libero_single_task_id=5 \
    data.n_samples=1 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$CKPT_DIR \
    +actor_rollout_ref.model.base_vla_path=$BASE_VLA_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.model.lora_rank=4 \
    actor_rollout_ref.model.target_modules=attn-only \
    +actor_rollout_ref.model.train_only_action_head=True \
    +actor_rollout_ref.model.load_in_8bit=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=True \
    actor_rollout_ref.rollout.val_micro_batch_size=1 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=libero_goal_no_noops \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$CKPT_DIR \
    +actor_rollout_ref.rollout.base_vla_path=$BASE_VLA_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    +actor_rollout_ref.rollout.perturb_colors=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.rollout.use_prompt_seg_text=False \
    +actor_rollout_ref.rollout.use_mask_image=True \
    +actor_rollout_ref.rollout.use_mask_from_env=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=1 \
    trainer.test_freq=4 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo_lite \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=online \
    trainer.val_before_train=True \
    2>&1 | tee "$LOGFILE"
