"""
run_single_chunk_inference.py

单次推理脚本：根据用户传入的图片、proprio、task description 生成一个 action chunk。
不做环境交互，只做一次前向推理。

用法示例：
  python experiments/robot/libero/run_single_chunk_inference.py \\
    --image_path /path/to/image.png \\
    --task_description "put the bowl on the plate" \\
    --pretrained_checkpoint /path/to/checkpoint \\
    [--proprio "0.1,0.2,0.3,0,0,0,0,0"] \\
    [--apply_mask true] \\
    [--output_path actions.npy]
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional, Union

import draccus

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 兼容自定义 ckpt：避免 weights_only 报错（与 run_libero_eval_mask 一致）
_EVAL_FORCE_CPU = os.environ.get("EVAL_FORCE_CPU", "0") == "1"
_old_torch_load = torch.load

def _torch_load_safe(*args, **kwargs):
    if _EVAL_FORCE_CPU:
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
    return _old_torch_load(*args, **kwargs)

torch.load = _torch_load_safe

import numpy as np
from PIL import Image

# Append openvla-oft root for imports (libero -> robot -> experiments -> openvla-oft)
_openvla_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_openvla_root))

from prismatic.vla.datasets.datasets import (
    language_mask_processor,
    mask_image_via_other_env,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)


@draccus.wrap()
def run_inference(
    # 用户必须传入
    image_path: str,
    task_description: str,
    # 模型相关
    pretrained_checkpoint: Union[str, Path] = "",
    base_vla_path: Optional[str] = None,
    model_family: str = "openvla",
    use_l1_regression: bool = True,
    use_diffusion: bool = False,
    use_proprio: bool = False,
    num_images_in_input: int = 1,
    center_crop: bool = True,
    lora_rank: int = 32,
    unnorm_key: str = "libero_goal",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    # 用户可选传入
    proprio: Optional[str] = None,
    wrist_image_path: Optional[str] = None,
    # mask 相关（与 run_libero_eval_mask 一致）
    apply_mask: bool = True,
    mask_prompt: Optional[str] = None,
    mask_output_path: Optional[str] = None,
    # 输出
    output_path: Optional[str] = None,
    seed: int = 7,
) -> list:
    """
    单次推理：根据 image_path, task_description, proprio 生成一个 action chunk。

    Args:
        image_path: 主视角图片路径
        task_description: 任务描述，如 "put the bowl on the plate"
        proprio: 可选，逗号分隔的 8 维 proprio 向量，如 "0.1,0.2,..."
        wrist_image_path: 可选，手腕相机图片路径（num_images_in_input>1 时）
        apply_mask: 是否对图片做 mask（与 run_libero_eval_mask 一致）
        mask_prompt: mask 时用的文本，默认与 task_description 相同
        mask_output_path: mask 图片保存路径，默认自动生成
        output_path: 输出 action chunk 的路径（npy），不指定则只打印

    Returns:
        list: 一个 action chunk，即 NUM_ACTIONS_CHUNK 个 (7,) 的 action 数组
    """
    assert pretrained_checkpoint, "pretrained_checkpoint 必须指定"
    assert os.path.isfile(image_path), f"图片不存在: {image_path}"

    set_seed_everywhere(seed)

    # 加载模型
    class Cfg:
        pretrained_checkpoint = pretrained_checkpoint
        base_vla_path = base_vla_path
        model_family = model_family
        use_l1_regression = use_l1_regression
        use_diffusion = use_diffusion
        use_proprio = use_proprio
        num_images_in_input = num_images_in_input
        center_crop = center_crop
        lora_rank = lora_rank
        unnorm_key = unnorm_key
        load_in_8bit = load_in_8bit
        load_in_4bit = load_in_4bit
        use_film = False
        num_diffusion_steps_train = 50
        num_diffusion_steps_inference = 50

    cfg = Cfg()
    model = get_model(cfg)
    resize_size = get_image_resize_size(cfg)
    processor = get_processor(cfg)

    # unnorm_key 校验与设置
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{unnorm_key}_no_noops"
    else:
        cfg.unnorm_key = unnorm_key
    assert cfg.unnorm_key in model.norm_stats, (
        f"Action un-norm key {cfg.unnorm_key} not found in model.norm_stats!"
    )

    action_head = get_action_head(cfg, model.llm_dim) if (use_l1_regression or use_diffusion) else None
    proprio_projector = (
        get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if use_proprio else None
    )
    noisy_action_projector = None

    # 加载主视角图片
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)

    # 可选：对图片做 mask
    if apply_mask:
        prompt_for_mask = mask_prompt if mask_prompt is not None else task_description
        if mask_output_path is None:
            mask_output_path = image_path.replace(".png", "_masked.png").replace(".jpg", "_masked.jpg")
        masked = mask_image_via_other_env(img_pil, prompt_for_mask, mask_output_path)
        img_for_policy = resize_image_for_policy(np.array(masked), resize_size)
    else:
        img_for_policy = resize_image_for_policy(img_np, resize_size)

    # 构建 observation
    observation = {"full_image": img_for_policy}

    if num_images_in_input > 1 and wrist_image_path:
        wrist_np = np.array(Image.open(wrist_image_path).convert("RGB"))
        observation["wrist_image"] = resize_image_for_policy(wrist_np, resize_size)

    if use_proprio:
        if proprio is None:
            # 使用零向量作为占位
            observation["state"] = np.zeros(8, dtype=np.float32)
        else:
            observation["state"] = np.array([float(x.strip()) for x in proprio.split(",")], dtype=np.float32)
            assert observation["state"].size == 8, f"LIBERO proprio 应为 8 维，当前 {observation['state'].size}"

    # 处理 task description（language_mask_processor）
    proc_task = language_mask_processor(task_description)

    # 单次推理
    from experiments.robot.robot_utils import get_action

    actions = get_action(
        cfg,
        model,
        observation,
        proc_task,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=cfg.use_film,
    )

    # 输出
    actions_np = np.array([np.asarray(a) for a in actions])
    print("Action chunk shape:", actions_np.shape)
    print("Actions:\n", actions_np)

    if output_path:
        np.save(output_path, actions_np)
        print(f"Saved action chunk to {output_path}")

    return actions


if __name__ == "__main__":
    run_inference()
