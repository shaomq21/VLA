#!/usr/bin/env python3
"""
提取 OpenVLA LLAMA 进 action head 前的 hidden state。

用法:
  1. 在某文件夹下放一张 test 图 (如 test.png)
  2. 激活含 transformers/torch 的 conda 环境，例如: conda activate simplevla
  3. 运行:
     cd /home/ubuntu/16831pro_fine_tune/openvla-oft
     PYTHONPATH=. python scripts/extract_hidden_states.py \\
       --image_path /path/to/your/test.png \\
       [--output_dir ./output_hidden_states] \\
       [--load_in_8bit]  # 14GB 显存建议加此参数

  两个 prompt（下划线已转为空格）:
    - pick up the book in the middle and place it on the cabinet shelf
    - pick up the book on the right and place it on the cabinet shelf

  模型: https://huggingface.co/openvla/openvla-7b-finetuned-libero-10
  首次运行会自动从 HF 下载。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 添加项目根目录
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))

# 确保使用 LIBERO 常量
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 两个 prompt（下划线已去掉）
PROMPTS = [
    "pick up the book in the middle and place it on the cabinet shelf",
    "pick up the book on the right and place it on the cabinet shelf",
]

MODEL_ID = "openvla/openvla-7b-finetuned-libero-10"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_and_processor(load_in_8bit: bool = False):
    """从 HuggingFace 加载 openvla-7b-finetuned-libero-10 和 processor。"""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    # 注册自定义类（HF 模型可能已包含 auto_map）
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading model and processor from {MODEL_ID} (first run may download)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if load_in_8bit:
        load_kwargs.update(load_in_8bit=True, device_map="auto")
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **load_kwargs)

    if not load_in_8bit:
        model = model.to(DEVICE)
    model.eval()

    # 检测可用 unnorm_key
    if hasattr(model, "norm_stats") and model.norm_stats:
        available = list(model.norm_stats.keys())
        print(f"Available unnorm_key: {available}")

    return model, processor


def get_hidden_states(model, processor, image: Image.Image, prompt: str, unnorm_key: str = "libero_long"):
    """
    对单张图和 prompt 做 forward，返回 LLAMA 进 action head 前的 hidden state。

    Returns:
        actions: (NUM_ACTIONS_CHUNK, ACTION_DIM) 预测动作
        hidden_states: (1, act_chunk_len, llm_dim) 进 action head 前的 hidden state
    """
    # 构建与 OpenVLA 一致的 prompt
    full_prompt = f"In: What action should the robot take to {prompt}?\nOut:"

    inputs = processor(full_prompt, image, return_tensors="pt")
    inputs = {k: v.to(DEVICE, dtype=torch.bfloat16) if v.dtype in (torch.float32, torch.float64) else v.to(DEVICE)
              for k, v in inputs.items()}

    with torch.inference_mode():
        actions, hidden_states = model.predict_action(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            unnorm_key=unnorm_key,
        )

    return actions, hidden_states


def main():
    parser = argparse.ArgumentParser(description="提取 OpenVLA 进 action head 前的 hidden state")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="测试图片路径（如 /path/to/test.png）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_hidden_states",
        help="输出目录，存放 hidden states 和 actions",
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default=None,
        help="Action unnormalization 的 dataset key；不指定则用 model.norm_stats 里第一个",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="8bit 量化加载，14GB 显存建议开启",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载图片
    image = Image.open(image_path).convert("RGB")

    # 加载模型
    model, processor = load_model_and_processor(load_in_8bit=args.load_in_8bit)

    unnorm_key = args.unnorm_key
    if unnorm_key is None and hasattr(model, "norm_stats") and model.norm_stats:
        unnorm_key = next(iter(model.norm_stats.keys()))
        print(f"Using unnorm_key: {unnorm_key}")

    results = {}
    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i + 1}: {prompt} ---")
        actions, hidden_states = get_hidden_states(
            model, processor, image, prompt, unnorm_key=unnorm_key
        )

        # hidden_states: (1, act_chunk_len, llm_dim)
        hs = hidden_states.cpu().float().numpy()
        acts = np.asarray(actions)

        key = f"prompt_{i + 1}"
        results[key] = {
            "prompt": prompt,
            "actions": acts,
            "hidden_states": hs,
        }

        np.save(output_dir / f"{key}_hidden_states.npy", hs)
        np.save(output_dir / f"{key}_actions.npy", acts)

        print(f"  Hidden states shape: {hs.shape}")
        print(f"  Actions shape: {acts.shape}")

        # 释放显存，避免下一个 prompt 时 OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存元信息
    with open(output_dir / "meta.txt", "w") as f:
        f.write(f"model_id: {MODEL_ID}\n")
        f.write(f"image_path: {image_path}\n")
        f.write(f"unnorm_key: {unnorm_key}\n")
        f.write("prompts:\n")
        for i, p in enumerate(PROMPTS):
            f.write(f"  {i + 1}: {p}\n")

    print(f"\nDone. Outputs saved to {output_dir}")
    print(f"  - prompt_1_hidden_states.npy, prompt_1_actions.npy")
    print(f"  - prompt_2_hidden_states.npy, prompt_2_actions.npy")
    print(f"  - meta.txt")


if __name__ == "__main__":
    main()
