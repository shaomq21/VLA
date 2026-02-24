"""
RLDS mask preprocessing: apply Grounded-SAM masking to RLDS dataset images.

- Loads raw RLDS episodes via tfds, applies mask to images, writes back to TFRecord format
- Output format: {out_root}/{data_mix}/1.0.0/{tfrecord_prefix}-train.tfrecord-XXXXX-of-YYYYY
- Preserves all fields (actions, state, language, etc.); only images are masked
- Output is directly usable for training (same structure as openvla/modified_libero_rlds)
"""

from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image

# Resolve paths: run from VLA repo root, openvla-oft is sibling of tools/
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
if _OPENVLA_ROOT.exists():
    sys.path.insert(0, str(_REPO_ROOT))  # for prismatic
    sys.path.insert(0, str(_OPENVLA_ROOT))  # for mask_processor
else:
    _OPENVLA_ROOT = Path("/home/ubuntu/16831pro_fine_tune/openvla-oft")
    sys.path.insert(0, str(_OPENVLA_ROOT.parent))
    sys.path.insert(0, str(_OPENVLA_ROOT))

print("[rlds_mask] Importing mask_processor...", flush=True)
from mask_processor import GroundedSAMConfig, GroundedSAMMasker
print("[rlds_mask] Imports done.", flush=True)

# Default paths (override via args or env)
DEFAULT_DATA_ROOT = os.environ.get("RLDS_DATA_ROOT", str(_REPO_ROOT / "openvla-oft/datasets/modified_libero_rlds"))
DEFAULT_OUT_ROOT = os.environ.get("RLDS_OUT_ROOT", str(_REPO_ROOT / "openvla-oft/datasets/masked_libero_rlds"))
RESUME_FILE = ".rlds_resume.json"
SAVE_PROGRESS_EVERY = 10  # episodes
NUM_SHARDS = 16
# Map data_mix name to tfrecord filename prefix (from dataset_info.json "name")
TFRECORD_PREFIX = {
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
    "libero_10_no_noops": "libero_10",
}


def _ensure_tf_cpu():
    tf.config.set_visible_devices([], "GPU")


def _to_numpy(x):
    """Convert tf.Tensor to numpy if needed."""
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x


def _decode_str(s):
    if hasattr(s, "decode"):
        return s.decode("utf-8")
    return str(s)


def _mask_episode_steps(episode, masker, mask_wrist=True):
    """Apply mask to image and wrist_image in each step. Returns modified steps list."""
    steps_ds = episode["steps"]
    if hasattr(steps_ds, "as_numpy_iterator"):
        steps_list = list(steps_ds.as_numpy_iterator())
    else:
        steps_list = [s for s in steps_ds]

    lang_raw = steps_list[0].get("language_instruction", b"")
    lang = _decode_str(lang_raw).lower() if lang_raw is not None else ""

    modified_steps = []
    for step in steps_list:
        obs = dict(step["observation"])
        # Mask primary image
        img_arr = _to_numpy(obs["image"])
        if img_arr is not None and img_arr.size > 0:
            img = Image.fromarray(img_arr).convert("RGB")
            masked = masker.mask_image_from_lang(img, lang)
            obs["image"] = np.array(masked)
        # Mask wrist image
        if mask_wrist and "wrist_image" in obs:
            wrist_arr = _to_numpy(obs["wrist_image"])
            if wrist_arr is not None and wrist_arr.size > 0:
                wrist_img = Image.fromarray(wrist_arr).convert("RGB")
                masked_wrist = masker.mask_image_from_lang(wrist_img, lang)
                obs["wrist_image"] = np.array(masked_wrist)
        step_copy = dict(step)
        step_copy["observation"] = obs
        modified_steps.append(step_copy)
    return modified_steps


def main():
    import traceback
    try:
        _main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


def _main():
    print("[rlds_mask] _main() started.", flush=True)
    parser = argparse.ArgumentParser(description="Apply Grounded-SAM masks to RLDS dataset; output TFRecord for training")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="RLDS data root (input)")
    parser.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT, help="Output root for masked TFRecord")
    parser.add_argument("--data_mix", type=str, default="libero_goal_no_noops", help="Dataset name (e.g. libero_goal_no_noops)")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed episode index")
    parser.add_argument("--no_mask_wrist", action="store_true", help="Skip masking wrist camera (only mask primary)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to process (for testing)")
    parser.add_argument("--num_shards", type=int, default=NUM_SHARDS, help="Number of output TFRecord shards")
    parser.add_argument("--dino_config", type=str, default=None)
    parser.add_argument("--dino_ckpt", type=str, default=None)
    parser.add_argument("--sam_ckpt", type=str, default=None)
    parser.add_argument("--sam_type", type=str, default="vit_b",
        help="SAM backbone: vit_b (fast) | vit_l | vit_h (slowest, best)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug_dir", type=str, default="rlds_mask_debug", help="Save sample masked images for inspection")
    parser.add_argument("--debug_every", type=int, default=200, help="Save a sample masked image every N steps (default 200)")
    args = parser.parse_args()

    _ensure_tf_cpu()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_dir = out_root / args.data_mix / "1.0.0"
    src_dir = data_root / args.data_mix / "1.0.0"

    tfrecord_prefix = TFRECORD_PREFIX.get(args.data_mix, args.data_mix.replace("_no_noops", ""))
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")

    os.makedirs(out_dir, exist_ok=True)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    # Masker
    _SAM_CKPT = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }
    dino_config = args.dino_config or str(_OPENVLA_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    dino_ckpt = args.dino_ckpt or str(_OPENVLA_ROOT / "groundingdino_swint_ogc.pth")
    sam_ckpt = args.sam_ckpt or str(_OPENVLA_ROOT / _SAM_CKPT.get(args.sam_type, "sam_vit_b_01ec64.pth"))
    cfg = GroundedSAMConfig(
        dino_config_path=dino_config,
        dino_checkpoint_path=dino_ckpt,
        sam_checkpoint_path=sam_ckpt,
        sam_type=args.sam_type,
        device=args.device,
    )
    print("Loading GroundedSAM...", flush=True)
    masker = GroundedSAMMasker(cfg)
    print("GroundedSAM loaded.", flush=True)

    # Load RLDS at episode level
    builder = tfds.builder(args.data_mix, data_dir=str(data_root))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    num_episodes = builder.info.splits["train"].num_examples
    if args.max_episodes is not None:
        num_episodes = min(num_episodes, args.max_episodes)
    print("Total episodes:", num_episodes)

    # Resume
    resume_file = out_root / RESUME_FILE
    resume_from = 0
    if args.resume and resume_file.exists():
        try:
            with open(resume_file) as f:
                state = json.load(f)
            resume_from = int(state.get("last_episode", 0))
            print(f"Resuming from episode {resume_from}")
        except Exception as e:
            print(f"Could not load resume state: {e}")

    ds = ds.skip(resume_from)
    total_to_process = num_episodes - resume_from
    if args.max_episodes is not None:
        total_to_process = min(total_to_process, args.max_episodes)
        ds = ds.take(args.max_episodes)

    # Open shard writers
    n_shards = args.num_shards
    shard_files = [
        out_dir / f"{tfrecord_prefix}-train.tfrecord-{i:05d}-of-{n_shards:05d}"
        for i in range(n_shards)
    ]
    writers = [tf.io.TFRecordWriter(str(p)) for p in shard_files]
    shard_counts = [0] * n_shards

    features = builder.info.features
    from tensorflow_datasets.core import example_serializer
    example_serializer_obj = example_serializer.ExampleSerializer(features.get_serialized_info())
    total_steps = 0
    next_save_at = args.debug_every

    try:
        for ep_idx, episode in enumerate(tqdm(ds, total=total_to_process, desc="RLDS mask")):
            global_ep = resume_from + ep_idx
            modified_steps = _mask_episode_steps(episode, masker, mask_wrist=not args.no_mask_wrist)
            steps_this_ep = len(modified_steps)
            ep_meta_raw = episode.get("episode_metadata")
            ep_metadata = {}
            if ep_meta_raw is not None:
                try:
                    raw = ep_meta_raw.numpy() if hasattr(ep_meta_raw, "numpy") else ep_meta_raw
                    if isinstance(raw, dict):
                        ep_metadata = {k: _to_numpy(v) for k, v in raw.items()}
                    elif isinstance(raw, (bytes, str)):
                        ep_metadata = {"file_path": raw if isinstance(raw, bytes) else raw.encode()}
                except Exception:
                    pass
            modified_episode = {
                "steps": modified_steps,
                "episode_metadata": ep_metadata,
            }
            encoded = features.encode_example(modified_episode)
            if isinstance(encoded, bytes):
                serialized = encoded
            elif isinstance(encoded, dict):
                serialized = example_serializer_obj.serialize_example(encoded)
            else:
                serialized = encoded  # hope it's bytes-like
            shard_id = global_ep % n_shards
            writers[shard_id].write(serialized)
            shard_counts[shard_id] += 1

            # 每隔 debug_every 步保存一张 mask 后的图片供检查
            if args.debug_dir and total_steps < next_save_at <= total_steps + steps_this_ep:
                first_step = modified_steps[0]
                img_arr = first_step["observation"]["image"]
                if img_arr is not None and img_arr.size > 0:
                    img = Image.fromarray(img_arr)
                    img.save(Path(args.debug_dir) / f"step{next_save_at:06d}_ep{global_ep:05d}_masked.png")
                next_save_at += args.debug_every

            total_steps += steps_this_ep

            if (ep_idx + 1) % SAVE_PROGRESS_EVERY == 0:
                with open(resume_file, "w") as f:
                    json.dump({"last_episode": global_ep + 1}, f)
    finally:
        for w in writers:
            w.close()

    # Copy metadata
    for name in ["dataset_info.json", "features.json"]:
        src = src_dir / name
        dst = out_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {name} to {dst}")
        else:
            print(f"Warning: {src} not found, skipping")

    # Update dataset_info.json shard lengths if we changed episode count
    info_path = out_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        splits = info.get("splits", [])
        for s in splits:
            if s.get("name") == "train":
                s["shardLengths"] = [str(c) for c in shard_counts]
                break
        with open(info_path, "w") as f:
            json.dump(info, f, indent=1)

    with open(resume_file, "w") as f:
        json.dump({"last_episode": resume_from + total_to_process}, f)

    print("DONE. Output:", out_dir)
    print("TFRecord files:", [str(p) for p in shard_files])
    print("Directly usable for training with --data_root", out_root)


if __name__ == "__main__":
    main()
