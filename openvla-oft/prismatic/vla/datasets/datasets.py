"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image



def language_mask_processor(lang: str) -> str:
    """
    Process natural-language task instruction with masked-object rules.

    Input example:
        "put the bowl on the plate"
        "open the middle drawer of the cabinet"
        "push the plate to the front of the stove"
        "turn on the stove"
    """

    s = lang.strip().lower()
    import re
    # -------- verb-based masked insertion --------
    if s.startswith("put "):
        # put the -> put the red masked
        s = re.sub(r"\bput the\b", "put the red masked", s, count=1)
        # on the -> on the green masked
        s = re.sub(r"\bon the\b", "on the green masked", s, count=1)
        s = re.sub(r"\bon top of the\b", "on on top of the green masked", s, count=1)
        s = re.sub(r"\bin the\b", "in the green masked", s, count=1)

    elif s.startswith("turn on "):
        # turn on -> turn on green masked
        s = re.sub(r"\bturn on the\b", "turn on the green masked", s, count=1)

    elif s.startswith("push "):
        # push the -> push the green masked
        s = re.sub(r"\bpush the\b", "push the red masked", s, count=1)
        s = re.sub(r"\bto the front of the stove\b", "to the green masked place", s, count=1)

    elif s.startswith("open "):

        # open the middle -> open the green masked
        s = re.sub(
            r"\bopen the middle\b",
            "open the green masked",
            s,
            count=1,
        )
        # open the top -> open the green masked top
        s = re.sub(
            r"\bopen the top\b",
            "open the green masked",
            s,
            count=1,
        )

    # -------- object category replacement --------
    # cabinet / rack -> square object
    s = re.sub(r"\bcabinet\b", "square object", s)
    s = re.sub(r"\brack\b", "square object", s)

    # stove / plate -> flat-shaped object
    s = re.sub(r"\bstove\b", "flat-shaped object", s)
    s = re.sub(r"\bplate\b", "flat-shaped object", s)
    s = re.sub(r"\bcream cheese\b", "flat-shaped object", s)

    return s

import subprocess, tempfile, os
from PIL import Image

VLA_PREPROCESS_PY = "/home/ubuntu/miniconda3/envs/vla-preprocess/bin/python"  


from pathlib import Path

cur = Path(__file__).resolve()

while cur.name != "16831pro_fine_tune":
    cur = cur.parent

MASK_ONE_SCRIPT = cur / "tools" / "mask_one.py"



DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CKPT   = "groundingdino_swint_ogc.pth"
SAM_CKPT    = "sam_vit_h_4b8939.pth"
SAM_TYPE    = "vit_h"
DEVICE      = "cuda"


def mask_image_via_other_env(img_pil: Image.Image, lang: str, out_path: str) -> Image.Image:
    print("---activate env vla-preprocess---")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"   
    for k in ["WORLD_SIZE","RANK","LOCAL_RANK","LOCAL_WORLD_SIZE","MASTER_ADDR","MASTER_PORT"]:
        env.pop(k, None)

    tmp_dir = tempfile.mkdtemp(prefix="mask_tmp_")
    in_path = os.path.join(tmp_dir, "in.png")
    img_pil.save(in_path)

    cmd = [
        str(VLA_PREPROCESS_PY),
        "-u",
        str(MASK_ONE_SCRIPT),
        "--image_in", str(in_path),
        "--image_out", str(out_path),
        "--lang", str(lang),
        "--dino_config", str(DINO_CONFIG),
        "--dino_ckpt", str(DINO_CKPT),
        "--sam_ckpt", str(SAM_CKPT),
        "--sam_type", str(SAM_TYPE),
        "--device", "cuda",   
    ]

    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,             
    )

    if r.returncode != 0:
        raise RuntimeError(
            "mask_one.py failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{r.stdout}\n"
            f"stderr:\n{r.stderr}\n"
        )

    return Image.open(out_path).convert("RGB")


import io
from PIL import Image




@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
       
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        lang_object = None

        #process masked image
        lang_object = lang
        lang = language_mask_processor(lang)

        
        safe_name = lang_object.replace(" ", "_").replace("/", "_")
        out_path = f"debug_masked/{safe_name}.png"
        os.makedirs("debug_masked", exist_ok=True)

        img = mask_image_via_other_env(img.convert("RGB"), lang_object, out_path)
       

            


          

        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

# test_tasks = [
#     "open the middle drawer of the cabinet",
#     "open the top drawer and put the bowl inside",
#     "push the plate to the",
#     "put the bowl on the plate",
#     "put the bowl on the stove",
#     "put the bowl on top of the cabinet",
#     "put the cream cheese in the bowl",
#     "put the wine bottle on the rack",
#     "put the wine bottle on top of the cabinet",
#     "turn on the stove",
# ]
# for t in test_tasks:
#     print(t, " -> ", RLDSBatchTransform.language_mask_processor(t))

