# grpo_lite_rl

独立于 SimpleVLA-RL 的简易版 GRPO 实现，显存占用更低。

## 原理

- **grpo_lite**：batch-mean baseline，无 per-prompt group，纯 tensor 运算
- 适用于 `n_samples=1`（每 prompt 一次 rollout）
- 不加载 ref policy，不创建 critic

## 结构

```
grpo_lite_rl/
├── grpo_lite/
│   ├── __init__.py
│   ├── advantage.py    # compute_grpo_lite_advantage
│   └── patch_verl.py   # 运行时 patch SimpleVLA-RL
├── launcher.py         # 入口：patch 后调用 main_ppo
├── run_libero_push_perturb.sh
└── README.md
```

## 使用

```bash
cd /home/ubuntu/16831pro_fine_tune/grpo_lite_rl
bash run_libero_push_perturb.sh
```

依赖：同目录下的 `SimpleVLA-RL`、`LIBERO`、`openvla-oft`。不修改 SimpleVLA-RL 源码，通过 patch 注入 grpo_lite。

## 显存优化（相比原 run 脚本）

- `ppo_mini_batch_size=1`, `ppo_micro_batch_size=1`
- `traj_mini_batch_size=2`
- `log_prob_micro_batch_size=2`
- `gpu_memory_utilization=0.5`
