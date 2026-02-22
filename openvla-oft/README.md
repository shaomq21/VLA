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

 wandb: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

PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH \
python experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "/home/ubuntu/runs/openvla/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--1700_chkpt" \
  --task_suite_name libero_goal

git pull

git add .
git commit -m "today progress"
git push