"""
run_libero_generalization_eval.py

Generalization evaluation: only two tasks with perturb_colors (3 variants),
background perturbation (3 variants), image jitter (3 variants).
Mild perturbation intensity, similar to current perturb_colors.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from PIL import Image
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import draccus
import numpy as np
import tqdm

from libero.libero import benchmark
from prismatic.vla.datasets.datasets import language_mask_processor

# Append project root for experiments.robot imports (run from openvla-oft root)
_openvla_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_openvla_root))
import imageio
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_EVAL_FORCE_CPU = os.environ.get("EVAL_FORCE_CPU", "0") == "1"
_old_torch_load = torch.load


def _torch_load_safe(*args, **kwargs):
    if _EVAL_FORCE_CPU:
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
    return _old_torch_load(*args, **kwargs)


torch.load = _torch_load_safe


# Only these two tasks
GENERALIZATION_TASKS = [
    "push the plate to the front of the stove",
    "put the bowl on the plate",
]

# Task suite for LIBERO_GOAL (contains these tasks)
TASK_SUITE_NAME = "libero_goal"
TASK_MAX_STEPS = 150


class PerturbType(str, Enum):
    NONE = "none"
    COLOR = "color"
    BACKGROUND = "background"
    JITTER = "jitter"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _apply_color_perturbation(
    img: Union[np.ndarray, Image.Image], variant: int
) -> Union[np.ndarray, Image.Image]:
    """
    Mild color perturbation. variant 0|1|2.
    Original (variant 0): white→yellow, red→light blue.
    Variant 1: white→pale green, red→light pink.
    Variant 2: white→pale cyan, red→light orange.
    """
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    white_mask = (r > 200) & (g > 200) & (b > 200)
    red_mask = (r > 95) & (r >= g) & (r >= b) & ((r - np.minimum(g, b)) >= 10)
    alpha = 0.5  # same as original

    if variant == 0:
        # Original: white→yellow, red→light blue
        arr[white_mask, :] = [255, 255, 0]
        overlay = np.array([200.0, 220.0, 255.0], dtype=np.float32)
    elif variant == 1:
        # white→pale green, red→light pink
        arr[white_mask, :] = [220, 255, 220]
        overlay = np.array([255.0, 230.0, 240.0], dtype=np.float32)
    else:  # variant == 2
        # white→pale cyan, red→light orange
        arr[white_mask, :] = [220, 255, 255]
        overlay = np.array([255.0, 235.0, 210.0], dtype=np.float32)

    arr[red_mask, :] = (1 - alpha) * arr[red_mask, :] + alpha * overlay

    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


def _apply_background_perturbation(
    img: Union[np.ndarray, Image.Image], variant: int, rng: np.random.Generator
) -> Union[np.ndarray, Image.Image]:
    """
    KTV-style: light color patches over background. Mild intensity.
    variant 0: warm (pink/orange), 1: cool (blue/purple), 2: mixed (green/cyan).
    """
    arr = np.asarray(img).astype(np.float32).copy()
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img

    h, w = arr.shape[:2]
    blend_alpha = 0.18  # light overlay

    if variant == 0:
        colors = [
            [255, 220, 230],  # pink
            [255, 235, 210],  # peach
        ]
    elif variant == 1:
        colors = [
            [220, 225, 255],  # light blue
            [235, 220, 255],  # lavender
        ]
    else:  # variant == 2
        colors = [
            [220, 255, 235],  # mint
            [220, 248, 255],  # light cyan
        ]

    # 3–4 soft elliptical patches
    n_patches = 3 + rng.integers(0, 2)
    for _ in range(n_patches):
        cx = rng.integers(w // 4, 3 * w // 4)
        cy = rng.integers(h // 4, 3 * h // 4)
        rx = rng.integers(w // 6, w // 3)
        ry = rng.integers(h // 6, h // 3)
        color = np.array(colors[rng.integers(0, len(colors))], dtype=np.float32)

        yy, xx = np.ogrid[:h, :w]
        dist = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
        mask = np.exp(-dist * 0.5)
        mask = np.clip(mask, 0, 1)
        for c in range(3):
            arr[:, :, c] = (1 - blend_alpha * mask) * arr[:, :, c] + blend_alpha * mask * color[c]

    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


def _apply_image_jitter(
    img: Union[np.ndarray, Image.Image], variant: int, rng: np.random.Generator
) -> Union[np.ndarray, Image.Image]:
    """
    Mild image jitter. variant 0: translation, 1: rotation, 2: scale.
    """
    pil = Image.fromarray(np.asarray(img)) if isinstance(img, np.ndarray) else img
    pil = pil.convert("RGB")
    arr = np.asarray(pil)
    h, w = arr.shape[:2]

    if variant == 0:
        # Translation ±4 px
        dx = int(rng.integers(-4, 5))
        dy = int(rng.integers(-4, 5))
        out = np.roll(arr, (dy, dx), axis=(0, 1))
    elif variant == 1:
        # Rotation ±1.5 deg
        angle = float(rng.uniform(-1.5, 1.5))
        out_pil = pil.rotate(angle, resample=Image.BICUBIC, expand=False, fill=(128, 128, 128))
        out = np.asarray(out_pil)
    else:  # variant == 2
        # Scale 0.96–1.04, crop center
        s = float(rng.uniform(0.97, 1.03))
        new_h, new_w = int(h * s), int(w * s)
        resized = np.asarray(pil.resize((new_w, new_h), Image.Resampling.LANCZOS))
        top = max(0, (new_h - h) // 2)
        left = max(0, (new_w - w) // 2)
        out = resized[top : top + h, left : left + w]
        if out.shape[0] < h or out.shape[1] < w:
            pad = np.full((h, w, 3), 128, dtype=np.uint8)
            pad[: out.shape[0], : out.shape[1]] = out
            out = pad

    return Image.fromarray(out) if isinstance(img, Image.Image) else out


@dataclass
class GeneralizationConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    base_vla_path: Optional[str] = None

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = TASK_SUITE_NAME
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/info"
    use_mask_from_env: bool = False  # False = use mask_processor (Grounded-SAM)
    dino_config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_ckpt_path: str = "groundingdino_swint_ogc.pth"
    sam_ckpt_path: str = "sam_vit_b_01ec64.pth"
    sam_type: str = "vit_b"
    mask_device: str = "cuda"
    use_wandb: bool = False
    wandb_entity: str = "maggiesh-carnegie-mellon-university"
    wandb_project: str = "validation"
    seed: int = 7


def _resolve_base_vla_path(cfg: GeneralizationConfig) -> None:
    base = getattr(cfg, "base_vla_path", None)
    if not base or not isinstance(base, str) or not base.strip():
        return
    base = base.strip()
    if (base.startswith("/") or base.startswith(".")) and not os.path.isdir(base):
        root = Path(__file__).resolve().parent.parent.parent.parent
        fallback = root / "checkpoints" / "openvla-7b"
        if fallback.is_dir():
            cfg.base_vla_path = str(fallback)
            logger.info("base_vla_path %s not found; using %s", base, cfg.base_vla_path)


def _validate_config(cfg: GeneralizationConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must be set!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit!"


def _initialize_model(cfg: GeneralizationConfig):
    model = get_model(cfg)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        unnorm_key = cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.norm_stats, f"unnorm_key {unnorm_key} not found!"
        cfg.unnorm_key = unnorm_key

    return model, action_head, proprio_projector, noisy_action_projector, processor


def _prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def _process_action(action, model_family: str):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


_mask_processor_masker = None
_use_mask_subprocess = False

# Mask subprocess config (same as datasets.py, but env forces CPU)
_VLA_PREPROCESS_PY = "/home/ubuntu/miniconda3/envs/vla-preprocess/bin/python"
_MASK_ONE_SCRIPT = Path(__file__).resolve().parents[3].parent / "tools" / "mask_one.py"
_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_DINO_CKPT = "groundingdino_swint_ogc.pth"
_SAM_CKPT = "sam_vit_b_01ec64.pth"
_SAM_TYPE = "vit_b"


def _mask_via_subprocess_cpu(img_pil: Image.Image, lang: str) -> Image.Image:
    """Run mask_one.py in subprocess with CUDA hidden; mask uses CPU only."""
    import subprocess
    import tempfile
    import shutil
    # Free GPU cache in main process before mask (leaves more room for VLA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["NVIDIA_VISIBLE_DEVICES"] = ""
    for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    tmp_dir = tempfile.mkdtemp(prefix="gen_mask_tmp_")
    try:
        in_path = os.path.join(tmp_dir, "in.png")
        out_path = os.path.join(tmp_dir, "out.png")
        img_pil.save(in_path)
        # Use shell so CUDA_VISIBLE_DEVICES is set before Python/imports run
        import shlex
        lang_safe = shlex.quote(lang)
        cmd_str = (
            'CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" '
            f'"{_VLA_PREPROCESS_PY}" -u "{_MASK_ONE_SCRIPT}" '
            f'--image_in "{in_path}" --image_out "{out_path}" --lang {lang_safe} '
            f'--dino_config "{_DINO_CONFIG}" --dino_ckpt "{_DINO_CKPT}" '
            f'--sam_ckpt "{_SAM_CKPT}" --sam_type "{_SAM_TYPE}" --device cpu'
        )
        r = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"mask_one.py failed: {r.stderr}")
        return Image.open(out_path).convert("RGB")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_mask_processor_masker(cfg: GeneralizationConfig):
    """Lazy-load GroundedSAMMasker from mask_processor. Falls back to subprocess (vla-preprocess) if groundingdino not in current env."""
    global _mask_processor_masker, _use_mask_subprocess
    if _mask_processor_masker is not None:
        return _mask_processor_masker
    if _use_mask_subprocess:
        return None
    try:
        from mask_processor import GroundedSAMMasker, GroundedSAMConfig
        mask_cfg = GroundedSAMConfig(
            dino_config_path=cfg.dino_config_path,
            dino_checkpoint_path=cfg.dino_ckpt_path,
            sam_checkpoint_path=cfg.sam_ckpt_path,
            sam_type=cfg.sam_type,
            device=cfg.mask_device,
        )
        _mask_processor_masker = GroundedSAMMasker(mask_cfg)
        return _mask_processor_masker
    except ModuleNotFoundError as e:
        logger.info("mask_processor in-process not available (%s), using subprocess (vla-preprocess)", e)
        _use_mask_subprocess = True
        return None


def _save_sidebyside_video(
    raw_images: List[np.ndarray],
    masked_images: List[np.ndarray],
    idx: int,
    success: bool,
    task_description: str,
    suffix: str,
    log_file=None,
    fps: int = 30,
) -> str:
    """Save video with raw (left) and masked (right) side by side."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    extra = f"--{suffix}" if suffix else ""
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed}{extra}.mp4"
    writer = imageio.get_writer(mp4_path, fps=fps)
    for raw, mask in zip(raw_images, masked_images):
        raw_np = np.asarray(raw)
        mask_np = np.asarray(mask)
        if raw_np.shape != mask_np.shape:
            mask_np = np.asarray(
                Image.fromarray(mask_np).resize((raw_np.shape[1], raw_np.shape[0]), Image.Resampling.LANCZOS)
            )
        sidebyside = np.concatenate([raw_np, mask_np], axis=1)
        writer.append_data(sidebyside)
    writer.close()
    logger.info("Saved side-by-side video: %s", mp4_path)
    if log_file:
        log_file.write(f"Saved side-by-side video: {mp4_path}\n")
        log_file.flush()
    return mp4_path


def run_episode(
    cfg: GeneralizationConfig,
    env,
    raw_task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_state,
    log_file,
    perturb_type: PerturbType,
    perturb_variant: int,
    rng: np.random.Generator,
) -> Tuple[bool, List, List]:
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    replay_masked_images = []
    max_steps = TASK_MAX_STEPS
    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            if len(action_queue) == 0:
                task_description = language_mask_processor(raw_task_description)
                observation, img = _prepare_observation(obs, resize_size)

                img_np = np.asarray(img) if not isinstance(img, np.ndarray) else img
                if img_np.ndim == 2:
                    img_np = np.stack([img_np] * 3, axis=-1)
                img_pil = Image.fromarray(img_np)

                # Apply perturbation
                if perturb_type == PerturbType.COLOR:
                    img_pil = _apply_color_perturbation(img_pil, perturb_variant)
                elif perturb_type == PerturbType.BACKGROUND:
                    img_pil = _apply_background_perturbation(img_pil, perturb_variant, rng)
                elif perturb_type == PerturbType.JITTER:
                    img_pil = _apply_image_jitter(img_pil, perturb_variant, rng)

                img_for_replay = np.asarray(img_pil)
                replay_images.append(img_for_replay)

                if cfg.use_mask_from_env:
                    from experiments.robot.libero.libero_utils import mask_image_from_libero_seg
                    seg_key = "agentview_segmentation_instance"
                    if seg_key in obs:
                        try:
                            masked = mask_image_from_libero_seg(
                                img_for_replay, obs[seg_key], env, alpha=0.5
                            )
                        except (TypeError, AttributeError):
                            masked = img_for_replay
                    else:
                        masked = img_for_replay
                else:
                    masker = _get_mask_processor_masker(cfg)
                    img_pil_rgb = Image.fromarray(img_for_replay).convert("RGB")
                    if masker is not None:
                        masked_pil = masker.mask_image_from_lang(
                            img_pil_rgb,
                            raw_task_description,
                            alpha=0.35,
                        )
                        masked = np.asarray(masked_pil)
                    else:
                        masked_pil = _mask_via_subprocess_cpu(
                            img_pil_rgb, raw_task_description
                        )
                        masked = np.asarray(masked_pil)
                replay_masked_images.append(np.asarray(masked))

                observation["full_image"] = resize_image_for_policy(masked, resize_size)

                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = _process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        logger.exception("Episode error: %s", e)

    return success, replay_images, replay_masked_images


def run_generalization_eval(cfg: GeneralizationConfig) -> float:
    _resolve_base_vla_path(cfg)
    _validate_config(cfg)

    if getattr(cfg, "load_in_8bit", False) or getattr(cfg, "load_in_4bit", False):
        from accelerate import big_modeling as _acc_bm
        import transformers.modeling_utils as _tf_mu
        _orig = _acc_bm.dispatch_model
        def _patched(m, *a, force_hooks=False, **kw):
            return _orig(m, *a, force_hooks=True, **kw)
        _acc_bm.dispatch_model = _tf_mu.dispatch_model = _patched

    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = _initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    run_id = f"GEN-{TASK_SUITE_NAME}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(log_path, "w")
    logger.info("Logging to %s", log_path)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Find task IDs for our two tasks
    task_ids_to_run = []
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        desc = (task.language or "").strip().lower()
        if desc in [t.strip().lower() for t in GENERALIZATION_TASKS]:
            task_ids_to_run.append((task_id, task.language))

    if not task_ids_to_run:
        log_file.write("No matching tasks found. Check GENERALIZATION_TASKS.\n")
        log_file.close()
        return 0.0

    log_file.write(f"Running {len(task_ids_to_run)} tasks: {[d for _, d in task_ids_to_run]}\n")
    log_file.write("Perturbations: color(3) + background(3) + jitter(3) + baseline, 10 per task\n")
    log_file.flush()

    # Perturbation schedule: perturbations first, baseline (no perturbation) last
    schedule = []
    for v in range(3):
        schedule.append((PerturbType.COLOR, v))
    for v in range(3):
        schedule.append((PerturbType.BACKGROUND, v))
    for v in range(3):
        schedule.append((PerturbType.JITTER, v))
    schedule.append((PerturbType.NONE, 0))  # baseline at end

    total_episodes = 0
    total_successes = 0
    rng = np.random.default_rng(cfg.seed)

    for task_id, task_description in task_ids_to_run:
        task = task_suite.get_task(task_id)
        env, _ = get_libero_env(
            task, cfg.model_family, resolution=cfg.env_img_res,
            use_segmentation_env=cfg.use_mask_from_env,
        )
        raw_desc = task_description
        initial_states = task_suite.get_task_init_states(task_id)

        for sched_idx, (ptype, pvar) in enumerate(schedule):
            label = "baseline" if ptype == PerturbType.NONE else f"{ptype.value}_{pvar}"
            log_file.write(f"\n--- Task: {task_description} | Perturb: {label} ---\n")
            log_file.flush()

            initial_state = initial_states[0] if initial_states is not None else None
            if initial_state is None and hasattr(task_suite, "get_task_init_states"):
                inits = task_suite.get_task_init_states(task_id)
                initial_state = inits[0] if inits is not None else None

            success, repl_raw, repl_masked = run_episode(
                cfg, env, raw_desc, model, resize_size,
                processor, action_head, proprio_projector, noisy_action_projector,
                initial_state, log_file, ptype, pvar, rng,
            )

            total_episodes += 1
            if success:
                total_successes += 1

            suffix = label if ptype != PerturbType.NONE else "baseline"
            save_rollout_video(
                repl_raw, total_episodes, success=success,
                task_description=task_description, log_file=log_file, suffix=f"raw_{suffix}",
            )
            save_rollout_video(
                repl_masked, total_episodes, success=success,
                task_description=task_description, log_file=log_file,
                suffix=f"masked_{suffix}",
            )
            _save_sidebyside_video(
                repl_raw, repl_masked, total_episodes, success,
                task_description, f"sidebyside_{suffix}", log_file=log_file,
            )

            log_file.write(f"Success: {success} | Total: {total_successes}/{total_episodes}\n")
            log_file.flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    log_file.write(f"\n=== Final: {total_successes}/{total_episodes} = {rate:.2%} ===\n")
    log_file.close()
    return rate


if __name__ == "__main__":
    @draccus.wrap()
    def main(cfg: GeneralizationConfig):
        run_generalization_eval(cfg)

    main()
