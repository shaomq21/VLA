from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image

from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.dataset import make_interleaved_dataset
import sys
from pathlib import Path

OPENVLA_ROOT = Path("/home/ubuntu/16831pro_fine_tune/openvla-oft")
sys.path.insert(0, str(OPENVLA_ROOT))

from mask_processor import GroundedSAMConfig, GroundedSAMMasker
# ====== paths ======
DATA_ROOT = "/home/ubuntu/16831pro_fine_tune/openvla-oft/datasets/modified_libero_rlds"
OUT_ROOT  = "/home/ubuntu/16831pro_fine_tune/openvla-oft/datasets/masked_libero_rlds"
DATA_MIX  = "libero_goal_no_noops"   
RESOLUTION = (224, 224)

# ====== masker cfg ======
cfg = GroundedSAMConfig(
    dino_config_path="/home/ubuntu/16831pro_fine_tune/openvla-oft/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    dino_checkpoint_path="/home/ubuntu/16831pro_fine_tune/openvla-oft/groundingdino_swint_ogc.pth",
    sam_checkpoint_path="/home/ubuntu/16831pro_fine_tune/openvla-oft/sam_vit_h_4b8939.pth",
    sam_type="vit_h",
    device="cuda",
)

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("Loading GroundedSAM...")
    masker = GroundedSAMMasker(cfg)

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
        traj_transform_threads=len(mixture_spec),
        traj_read_threads=len(mixture_spec),
        train=True,
    )

    dataset, dataset_length, _ = make_interleaved_dataset(**rlds_config)
    print("dataset_length:", dataset_length)

    for idx, rlds_batch in enumerate(tqdm(dataset.as_numpy_iterator(), total=dataset_length)):
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0]).convert("RGB")
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        out = masker.mask_image_from_lang(img, lang)

        out_path = Path(OUT_ROOT) / f"{idx:08d}.png"
        out.save(out_path)

    print("DONE:", OUT_ROOT)

if __name__ == "__main__":
    main()