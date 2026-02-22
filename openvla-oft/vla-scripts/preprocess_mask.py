from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights, OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.mask_processor import GroundedSAMMasker, GroundedSAMConfig


import os, json, argparse
from pathlib import Path

import numpy as np
from PIL import Image



def build_rlds_dataset(data_root_dir: Path, data_mix: str, resize_resolution=(224,224), shuffle_buffer_size=1, train=True):
    
    if data_mix in OXE_NAMED_MIXTURES:
        mixture_spec = OXE_NAMED_MIXTURES[data_mix]
    else:
        mixture_spec = [(data_mix, 1.0)]

    if "aloha" in data_mix:
        load_camera_views = ("primary", "left_wrist", "right_wrist")
    else:
        load_camera_views = ("primary", "wrist")

    per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        data_root_dir,
        mixture_spec,
        load_camera_views=load_camera_views,
        load_depth=False,
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
    )

    rlds_config = dict(
        traj_transform_kwargs=dict(
            window_size=1,
            future_action_window_size=NUM_ACTIONS_CHUNK - 1,
            skip_unlabeled=True,
            goal_relabeling_strategy="uniform",
        ),
        frame_transform_kwargs=dict(
            resize_size=resize_resolution,
            num_parallel_calls=16,
        ),
        dataset_kwargs_list=per_dataset_kwargs,
        shuffle_buffer_size=shuffle_buffer_size,   
        sample_weights=weights,
        balance_weights=True,
        traj_transform_threads=len(mixture_spec),
        traj_read_threads=len(mixture_spec),
        train=train,
    )
    dataset, _, _ = make_interleaved_dataset(**rlds_config)
    return dataset

def main():
    print("preprocess_mask.py started")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root_dir", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--index_path", type=str, required=True)
    ap.add_argument("--max_items", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--dino_config", type=str, required=True)
    ap.add_argument("--dino_ckpt", type=str, required=True)
    ap.add_argument("--sam_ckpt", type=str, required=True)
    ap.add_argument("--sam_type", type=str, default="vit_h")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    
    cfg = GroundedSAMConfig(
        dino_config_path=args.dino_config,
        dino_checkpoint_path=args.dino_ckpt,
        sam_checkpoint_path=args.sam_ckpt,
        sam_type=args.sam_type,
        device=args.device,
    )
    masker = GroundedSAMMasker(cfg)

    dataset = build_rlds_dataset(
        data_root_dir=Path(args.data_root_dir),
        data_mix=args.dataset_name,
        resize_resolution=(224, 224),
        shuffle_buffer_size=1,  
        train=True,
    )

    index_f = open(args.index_path, "w", encoding="utf-8")
    print(rlds_batch.keys())
    n = 0
    for rlds_batch in dataset.as_numpy_iterator():
        
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0]).convert("RGB")
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        
        masked = masker.mask_image_from_lang(img, lang)

        
        key = f"{args.dataset_name}:{n}"
        rel_path = f"{args.dataset_name}/{n:08d}.png"
        save_path = out_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        masked.save(save_path)
        


        
        index_f.write(json.dumps({"key": key, "masked_path": str(save_path)}, ensure_ascii=False) + "\n")

        n += 1
        if args.max_items > 0 and n >= args.max_items:
            break

    index_f.close()
    print(f"[OK] wrote {n} masks to {out_dir}")
    print(f"[OK] index at {args.index_path}")

if __name__ == "__main__":
    main()
