# ------------------------------------------------------------
# RLDS OFFLINE MASK PREPROCESS
# 模仿 RLDSBatchTransform 写法
# ------------------------------------------------------------

from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image

import numpy as np

from prismatic.vla.datasets.rlds.dataset import (
    get_oxe_dataset_kwargs_and_weights,
    make_interleaved_dataset,
)

from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)

from mask_processor import (
    GroundedSAMConfig,
    GroundedSAMMasker,
)

# =============================
# CONFIG
# =============================

DATA_ROOT = "/vol/data/modified_libero_rlds"
OUT_ROOT = "/vol/data/masked_libero_rlds"
DATA_MIX = "libero_goal_no_noops"

RESOLUTION = (224, 224)

# =============================
# INIT MASKER (只初始化一次)
# =============================

cfg = GroundedSAMConfig(
    dino_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    dino_checkpoint_path="groundingdino_swint_ogc.pth",
    sam_checkpoint_path="sam_vit_h_4b8939.pth",
    sam_type="vit_h",
    device="cuda",
)

print("Loading GroundedSAM...")
masker = GroundedSAMMasker(cfg)


# =============================
# BUILD RLDS (模仿 RLDSDataset)
# =============================

mixture_spec = [(DATA_MIX, 1.0)]

per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
    DATA_ROOT,
    mixture_spec,
    load_camera_views=("primary",),
    load_depth=False,
    load_proprio=False,
    load_language=True,
    action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
)

rlds_config = dict(
    traj_transform_kwargs=dict(
        window_size=1,
        future_action_window_size=0,
        skip_unlabeled=True,
        goal_relabeling_strategy="uniform",
    ),
    frame_transform_kwargs=dict(
        resize_size=RESOLUTION,
        num_parallel_calls=16,
    ),
    dataset_kwargs_list=per_dataset_kwargs,
    shuffle_buffer_size=1,
    sample_weights=weights,
    balance_weights=True,
    traj_transform_threads=1,
    traj_read_threads=1,
    train=True,
)

dataset, dataset_length, _ = make_interleaved_dataset(**rlds_config)

print("Dataset size:", dataset_length)

# =============================
# MAIN LOOP
# =============================

os.makedirs(OUT_ROOT, exist_ok=True)

for idx, rlds_batch in enumerate(tqdm(dataset.as_numpy_iterator())):

    # ===== 完全模仿 RLDSBatchTransform =====
    img = Image.fromarray(
        rlds_batch["observation"]["image_primary"][0]
    )

    lang = rlds_batch["task"]["language_instruction"].decode().lower()

    # ===== MASK =====
    masked = masker.mask_image_from_lang(
        img.convert("RGB"),
        lang,
    )

    # ===== SAVE =====
    out_path = Path(OUT_ROOT) / f"{idx:08d}.png"
    masked.save(out_path)

print("DONE.")