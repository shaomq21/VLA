CUDA_VISIBLE_DEVICES=1 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "/home/hongyi/16831pro_fine_tune/openvla-oft/openvla-7b-oft-finetuned-libero-goal" \
  --data_root_dir "/home/hongyi/16831pro_fine_tune/openvla-oft/datasets/modified_libero_rlds" \
  --dataset_name "libero_goal_no_noops" \
  --run_root_dir "/home/hongyi/runs/openvla" \
  --use_lora True \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla" \
  --wandb_entity "maggiesh" \
  --save_freq 50
  --resume True





python ../tools/mask_processor.py


deattatch: Ctrl + B 然后 D
tmux attach -t vla-oft

阅读：Ctrl + B 然后 [



   CUDA_VISIBLE_DEVICES=1 \
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "/home/hongyi/runs/openvla/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--400_chkpt" \
    --data_root_dir "/home/hongyi/16831pro_fine_tune/openvla-oft/datasets/modified_libero_rlds" \
    --dataset_name "libero_goal_no_noops" \
    --run_root_dir "/home/hongyi/runs/openvla" \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 4 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project "openvla" \
    --wandb_entity "maggiesh" \
    --save_freq 50 \
    --resume True \
    --resume_step 400


chmod +x auto_run.sh
./auto_run.sh


CUDA_VISIBLE_DEVICES=1 \

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "/home/hongyi/runs/openvla/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--450_chkpt" \
  --task_suite_name libero_goal


PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH \
python experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "/home/ubuntu/runs/openvla/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--1700_chkpt" \
  --task_suite_name libero_goal

git pull

git add .
git commit -m "add_rlds_processor"
git push


python tools/rlds_mask.py --data_mix libero_goal_no_noops --debug_dir rlds_mask_debug --debug_every 200 --no_mask_wrist 

./auto_run.sh --fast_model



  export ROBOFLOW_API_KEY="

CKPT_DIR="/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt"

PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH \
python experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --base_vla_path "/home/ubuntu/runs/openvla/openvla-7b" \
  --use_proprio True \
  --task_suite_name libero_goal







cd /home/ubuntu/16831pro_fine_tune/openvla-oft
PYTHONPATH=. /home/ubuntu/miniconda3/envs/vla-preprocess/bin/python test_roboflow_gripper.py