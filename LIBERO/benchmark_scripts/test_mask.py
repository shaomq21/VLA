#!/usr/bin/env python3
import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 静音 TF 噪声
warnings.filterwarnings(
    "ignore",
    message="The key `labels` is will return integer ids",
    category=FutureWarning,
)

import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamProcessor,
    SamModel,
)

# ================== 配置区 ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G_DINO_ID = "IDEA-Research/grounding-dino-base"   # 或 tiny 版
SAM_ID    = "facebook/sam-vit-base"               # 可换 vit-l / vit-h
IMAGE_PATH = "/home/hongyi/16831pro/LIBERO/benchmark_tasks/Libero.png"
QUERIES    = ["can", "basket"]
BOX_THR    = 0.20
TEXT_THR   = 0.15
IOU_THR_NMS= 0.50
# ============================================


def build_query_string(queries):
    return " . ".join(q.strip() for q in queries) + " ."

def make_output_dir(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = f"./outputs_ground_sam_{name}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_grounding_models():
    processor = AutoProcessor.from_pretrained(G_DINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(G_DINO_ID).to(DEVICE)
    return processor, model

def load_sam_models():
    sam_proc = SamProcessor.from_pretrained(SAM_ID)
    sam_model = SamModel.from_pretrained(SAM_ID).to(DEVICE)
    return sam_proc, sam_model

@torch.no_grad()
def detect_boxes(image: Image.Image, queries):
    processor, model = load_grounding_models()
    query_str = build_query_string(queries)

    inputs = processor(images=image, text=query_str, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    results_list = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THR,
        text_threshold=TEXT_THR,
        target_sizes=[image.size[::-1]],
    )
    res = results_list[0]
    boxes = res["boxes"]
    scores = res["scores"]

    if "text_labels" in res and res["text_labels"] is not None:
        labels = res["text_labels"]
    else:
        idx = res["labels"].tolist() if torch.is_tensor(res["labels"]) else res["labels"]
        labels = [queries[i] if 0 <= i < len(queries) else str(i) for i in idx]

    return boxes, scores, labels

def per_class_nms(boxes, scores, labels, iou_thr=0.5):
    if boxes.numel() == 0:
        return boxes, scores, labels

    keep_idx = []
    for cls in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == cls]
        if not idxs:
            continue
        b = boxes[idxs]
        s = scores[idxs]
        kept = torchvision.ops.nms(b, s, iou_threshold=iou_thr)
        keep_idx.extend([idxs[i] for i in kept])

    keep_idx = sorted(set(keep_idx))
    boxes_kept  = boxes[keep_idx]
    scores_kept = scores[keep_idx]
    labels_kept = [labels[i] for i in keep_idx]
    return boxes_kept, scores_kept, labels_kept

@torch.no_grad()
def boxes_to_sam_masks(image: Image.Image, boxes_xyxy: torch.Tensor):
    if boxes_xyxy.numel() == 0:
        H, W = image.size[1], image.size[0]
        return torch.zeros((0, H, W), dtype=torch.bool)

    sam_proc, sam_model = load_sam_models()

    input_boxes = boxes_xyxy.detach().cpu().unsqueeze(0)
    sam_inputs = sam_proc(image, input_boxes=input_boxes, return_tensors="pt").to(DEVICE)
    out = sam_model(**sam_inputs)

    post = sam_proc.post_process_masks(
        out,
        sam_inputs["original_sizes"],
        sam_inputs["reshaped_input_sizes"]
    )[0]

    masks_all = post["masks"].squeeze(1)
    iou_scores = out.iou_scores[0]
    num_boxes = boxes_xyxy.shape[0]
    multi = iou_scores.numel() // max(1, num_boxes)
    if multi < 1:
        multi = 1

    H, W = masks_all.shape[-2:]
    masks_all = masks_all.view(num_boxes, multi, H, W)
    iou_scores = iou_scores.view(num_boxes, multi)
    best_idx = torch.argmax(iou_scores, dim=1)
    masks = masks_all[torch.arange(num_boxes), best_idx]
    return (masks > 0.5).to(torch.bool)

def draw_boxes(image: Image.Image, boxes, labels, scores=None, save_path=None):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=3)
        caption = labels[i]
        if scores is not None:
            caption += f" {float(scores[i]):.2f}"
        tw, th = draw.textlength(caption), 12
        draw.rectangle([x1, max(0, y1-14), x1+tw+6, y1], fill=(0,0,0))
        draw.text((x1+3, max(0, y1-13)), caption, fill=(255,255,0))
    if save_path:
        img.save(save_path)
    return img

def overlay_masks(image: Image.Image, masks: torch.Tensor, labels, alpha=0.45, save_path=None):
    base = np.array(image).astype(np.float32) / 255.0
    H, W = base.shape[:2]
    out = base.copy()
    rng = np.random.default_rng(1234)
    colors = rng.random((len(labels), 3))
    for i, m in enumerate(masks):
        m_np = m.cpu().numpy().astype(bool)
        c = colors[i]
        out[m_np] = (1 - alpha) * out[m_np] + alpha * c
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(out)
    if save_path:
        img.save(save_path)
    return img

def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    out_dir = make_output_dir(IMAGE_PATH)
    print(f"[info] image size: {image.size}, save_dir: {out_dir}")

    boxes, scores, labels = detect_boxes(image, QUERIES)
    print(f"[detect] raw: N={boxes.shape[0]}")

    boxes, scores, labels = per_class_nms(boxes, scores, labels, IOU_THR_NMS)
    print(f"[detect] after per-class NMS: N={boxes.shape[0]}, labels={labels}")

    # 1️⃣ 保存检测框可视化
    box_vis_path = os.path.join(out_dir, "vis_boxes.png")
    draw_boxes(image, boxes, labels, scores, save_path=box_vis_path)
    print(f"[save] boxes -> {box_vis_path}")

    # 2️⃣ 生成mask
    masks = boxes_to_sam_masks(image, boxes)
    print(f"[sam] masks: {tuple(masks.shape)}")

    # 3️⃣ 保存mask叠加图
    mask_vis_path = os.path.join(out_dir, "vis_masks.png")
    overlay_masks(image, masks, labels, alpha=0.45, save_path=mask_vis_path)
    print(f"[save] masks -> {mask_vis_path}")

    # 4️⃣ 保存每个mask单独文件
    for i, m in enumerate(masks):
        inst = Image.fromarray((m.cpu().numpy().astype(np.uint8) * 255))
        inst_path = os.path.join(out_dir, f"mask_{i:02d}_{labels[i].replace(' ','_')}.png")
        inst.save(inst_path)
    print(f"[done] all results saved in {out_dir}")

if __name__ == "__main__":
    main()
