import os
import cv2
import argparse
import numpy as np
from termcolor import colored

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark


def render_task(task, bddl_file, init_states=None):
    """Render a single task scene and save the image."""
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    env = OffScreenRenderEnv(**env_args)
    env.reset()

    # 如果提供初始状态，就用它；否则默认重置环境
    if init_states is not None:
        obs = env.set_init_state(init_states[0])
    else:
        obs = env.reset()

    # 模拟几步，让环境稳定下来
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)

    image = obs["agentview_image"]
    os.makedirs("benchmark_tasks", exist_ok=True)

    # 输出文件名
    #image_name = f"{task.problem.replace(' ', '_')}.png"
    image_name = f"test.png"
    out_path = os.path.join("benchmark_tasks", image_name)
    cv2.imwrite(out_path, image[::-1, :, ::-1])

    env.close()
    print(colored(f"✅ Saved scene image to {out_path}", "green"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--bddl_file", type=str, required=True)
    args = parser.parse_args()

    benchmark_name = args.benchmark_name
    task_id = args.task_id
    bddl_file = args.bddl_file

    # 初始化 benchmark 实例
    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    task = benchmark_instance.get_task(task_id)

    # 如果存在初始状态，则读取，否则为 None
    try:
        init_states = benchmark_instance.get_task_init_states(task_id)
    except Exception:
        init_states = None

    render_task(task, bddl_file, init_states)

if __name__ == "__main__":
    main()
