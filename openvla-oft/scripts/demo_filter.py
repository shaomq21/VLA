import tensorflow_datasets as tfds
import tensorflow as tf
import os
import re
from tqdm import tqdm

# ===================================
# PATH
# ===================================
PARENT_DIR = "/home/hongyi/16831pro_fine_tune/openvla-oft/modified_libero_rlds"
SRC = "libero_object_no_noops"
OUT = "libero_object_filtered_no_noops"

# 完整路径
SRC_DIR = f"{PARENT_DIR}/{SRC}/1.0.0"
OUT_DIR = f"{PARENT_DIR}/{OUT}"

os.makedirs(OUT_DIR, exist_ok=True)

# ===================================
# Load original dataset
# ===================================
orig = tfds.load(SRC, data_dir=PARENT_DIR)

train = orig["train"]


# ===============================
# Target regex
# ===============================
patterns = [
    re.compile(r"pick.*alphabet.*basket", re.IGNORECASE),
    re.compile(r"pick.*tomato.*basket", re.IGNORECASE),
]


def match_episode(ep):
    steps = list(ep["steps"].as_numpy_iterator())
    lang = steps[0]["language_instruction"]
    if hasattr(lang, "decode"):
        lang = lang.decode()
    lang = lang.lower().strip()

    for p in patterns:
        if p.search(lang):
            return True
    return False


# ===============================
# Filtering
# ===============================
filtered = []
for ep in tqdm(train):
    if match_episode(ep):
        filtered.append(ep)

print("Filtered episodes:", len(filtered))


# ===================================
# TFRecord export
# ===================================
OUTPUT_TFRECORD = os.path.join(OUT_DIR, "filtered.tfrecord")
with tf.io.TFRecordWriter(OUTPUT_TFRECORD) as w:
    for ep in filtered:
        w.write(ep.SerializeToString())

print("Saved to:", OUTPUT_TFRECORD)


# ===================================
# copy metadata
# ===================================
for f in ["dataset_info.json", "features.json"]:
    os.system(f"cp {SRC_DIR}/{f} {OUT_DIR}/{f}")

print("\n=== DONE ===")
