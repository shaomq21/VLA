#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


# --------- 这些是 rollout 里用到的模型依赖 ---------
import torch
from transformers import AutoProcessor as _GDINOProcAlias, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor


# ========== 一些通用小工具 ==========
def _ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _sanitize_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _load_image_as_rgb(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return np.array(img)


def _save_image_rgb(image_rgb: np.ndarray, path: str) -> None:
    Image.fromarray(image_rgb).save(path)


def _save_mask_uint8(mask_bool: np.ndarray, path: str) -> None:
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)


# ========== 和 rollout 完全一致的 phrase 拆分逻辑 ==========
def _extract_phrases_from_task_desc(task_desc: str) -> list:
    """
    只根据输入语句自动拆短语（no candidates!）
    支持以下结构：
    - pick up X ... and place it on Y ...
    - put X on Y
    - move X to Y
    - stack X on Y
    - 更一般地：按 'and', 'on', 'into', 'to', 'in', 'onto' 等介词自动拆
    """
    desc = task_desc.lower().strip()
    desc = desc.replace("_", " ")

    phrases = []

    # ===== 1) 专门处理  pick up X ... and place (it) on Y =====
    if "pick up" in desc and "place" in desc:
        try:
            # after_pick: "the book on the right and place it on the cabinet shelf"
            after_pick = desc.split("pick up", 1)[1]

            # part1: 'the book on the right and '
            part1, rest = after_pick.split("place", 1)
            obj1 = part1.replace("and", " ").strip(" ,.")

            # rest: ' it on the cabinet shelf'
            # after_on: 'the cabinet shelf'
            if " on " in rest:
                after_on = rest.split(" on ", 1)[1]
                obj2 = after_on.strip(" ,.")
            else:
                obj2 = None

            if obj1:
                phrases.append(obj1)
            if obj2:
                phrases.append(obj2)

            if phrases:
                return phrases
        except Exception:
            pass

    # ===== 2) 一般结构：X on Y / X into Y / X to Y =====
    for sep in [" on ", " into ", " to ", " onto ", " in "]:
        if sep in desc:
            parts = [p.strip(" ,.") for p in desc.split(sep) if p.strip(" ,.")]
            if len(parts) >= 2:
                return parts

    # ===== 3) 按 ' and ' 粗暴拆，保证多个短语 =====
    if " and " in desc:
        parts = [p.strip(" ,.") for p in desc.split(" and ") if p.strip(" ,.")]
        if len(parts) >= 2:
            return parts

    # ===== 4) fallback: 只返回原句 =====
    return [desc]

def _draw_box_on_image(
    image_rgb: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 2,
) -> np.ndarray:
    """
    在一张 RGB 图像上画出一个矩形框，用于 debug 可视化。
    color: (R,G,B)
    """
    img = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box_xyxy
    for w in range(width):
        draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=color)
    return np.array(img)


# ========== rollout 中 simple_segment_gdino_sam 的逻辑 ==========
_GDINO_PROCESSOR = None
_GDINO_MODEL = None
_SAM_MODEL = None
_SAM_PREDICTOR = None
_SAM_CHECKPOINT_DEFAULT = "sam_vit_h_4b8939.pth"
_GDINO_MODEL_ID_DEFAULT = "IDEA-Research/grounding-dino-tiny"

_DEBUG_IMAGE_COUNT = 0  # 这里如果你愿意，也可以让它一直存，不做上限


@dataclass
class SegTestResult:
    mask: np.ndarray             # bool HxW，总并集
    soft_rgb: np.ndarray         # HxW x3, 已经做了前景提亮 & 背景减弱
    phrases: List[str]
    total_boxes: int
    max_score: float

    # 新增：每个 box / mask 的详细信息（不影响 rollout 逻辑）
    instance_masks: List[np.ndarray]
    instance_boxes: List[Tuple[float, float, float, float]]
    instance_scores: List[float]
    instance_phrases: List[str]



def simple_segment_gdino_sam(
    image_rgb: np.ndarray,
    prompt: str,
    *,
    box_threshold: float = 0.20,
    soft_bg_alpha: float = 0.3,
    sam_checkpoint: Optional[str] = None,
    gdino_model_id: str = _GDINO_MODEL_ID_DEFAULT,
    debug: bool = True,
    debug_dir: str = "seg_debug_cli",
) -> SegTestResult:
    """
    完全照 rollout 版的逻辑实现：
    - 文本 → 多个短语
    - 每个短语用 GroundingDINO 找 box
    - 每个 box 用 SAM 做 mask
    - 所有 mask 做并集
    - 前景强力提亮 + 上 tint，背景乘 soft_bg_alpha
    """
    global _GDINO_PROCESSOR, _GDINO_MODEL, _SAM_MODEL, _SAM_PREDICTOR, _DEBUG_IMAGE_COUNT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if sam_checkpoint is None:
        sam_checkpoint = _SAM_CHECKPOINT_DEFAULT

    # ----- 0. 初始化 GDINO / SAM（懒加载） -----
    if _GDINO_PROCESSOR is None or _GDINO_MODEL is None:
        _GDINO_PROCESSOR = _GDINOProcAlias.from_pretrained(
            gdino_model_id,
            trust_remote_code=True,
        )
        _GDINO_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model_id
        ).to(device)

    if _SAM_MODEL is None or _SAM_PREDICTOR is None:
        _SAM_MODEL = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        _SAM_MODEL.to(device=device)
        _SAM_PREDICTOR = SamPredictor(_SAM_MODEL)

    # ----- 1. 从 task 描述里抽多个“物体短语” -----
    phrases = _extract_phrases_from_task_desc(prompt)
    pil = Image.fromarray(image_rgb)
    H, W = image_rgb.shape[:2]
    if "robot arm" not in phrases:
        phrases.append("robot arm")
    

    if debug:
        print(f"[SEG] phrases: {phrases}", flush=True)

    # ----- 2. 对每个 phrase 跑一次 GDINO + SAM，所有 mask 并集 -----
    _SAM_PREDICTOR.set_image(image_rgb)  # 同一张图只 set 一次

    final_mask = np.zeros((H, W), dtype=bool)
    total_boxes = 0
    max_score = 0.0
    # 记录每个 box / mask 的信息，方便之后逐个可视化
    all_instance_masks: List[np.ndarray] = []
    all_instance_boxes: List[Tuple[float, float, float, float]] = []
    all_instance_scores: List[float] = []
    all_instance_phrases: List[str] = []

    for phrase in phrases:
        cleaned = phrase.replace("_", " ").strip()
        if not cleaned:
            continue

        inputs = _GDINO_PROCESSOR(
            images=pil,
            text=cleaned,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = _GDINO_MODEL(**inputs)

        target_sizes = torch.tensor([(H, W)]).to(device)

                # 兼容不同 transformers 版本的参数名
        try:
            results = _GDINO_PROCESSOR.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=0.05,
                text_threshold=0.25,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = _GDINO_PROCESSOR.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=0.05,
                text_threshold=0.25,
                target_sizes=target_sizes,
            )[0]

        # 先把 boxes / scores 取出来
        boxes = results["boxes"].detach().cpu().numpy()   # (N, 4)
        scores = results["scores"].detach().cpu().numpy() # (N,)

        total_boxes += boxes.shape[0]
        if scores.size > 0:
            max_score = max(max_score, float(scores.max()))

        if debug:
            print(
                f"[SEG] phrase={cleaned!r}, boxes={boxes.shape[0]}, "
                f"max_score={float(scores.max()) if scores.size>0 else None}",
                flush=True,
            )

        # 没有框就跳过这个 phrase
        if boxes.shape[0] == 0:
            continue


        # ---- 对每个 phrase 先筛一遍 box，只保留“像话”的那几个 ----
        # 计算每个 box 的中心和面积
        x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        areas = (x1 - x0) * (y1 - y0)
        img_area = float(H * W)

        phrase_l = cleaned.lower()

        # score 阈值：对 book 放宽一点，希望能把右边那些低分框保留下来
        if "book" in phrase_l:
            score_thr = box_threshold * 0.5  # 例如 box_threshold=0.2 -> 这里 0.1
        else:
            score_thr = box_threshold

        keep = scores >= score_thr
        max_keep = 1  # 默认每个 phrase 只用 1 个 box


        # ===== 针对不同 phrase 做一点启发式过滤 =====
        if "book" in phrase_l:
            # 书：小物体、在桌面上、靠右
            #   - 相对整图面积不能太大
            #   - y 在图像下半部分附近（桌面区域）
            #   - x 在右半边
            keep &= (areas / img_area < 0.20)   # 太大的（整块桌子/机械臂）直接扔掉
            keep &= (cy > 0.45 * H) & (cy < 0.95 * H)
            keep &= (cx > 0.5 * W)
            max_keep = 3   # 右边如果有多本书，可以最多留 3 个 box
        elif "shelf" in phrase_l:
            # 柜子/架子：大物体、偏上，不要整块桌面
            keep &= (areas / img_area > 0.10)
            keep &= (cy < 0.7 * H)   # 避免把整块桌面 + 书一起当成 shelf
            max_keep = 1

        idxs = np.nonzero(keep)[0]
        if idxs.size == 0:
            if "book" in phrase_l:
                # 对 book：如果没有任何框通过 “右边小物体 + score 阈值” 的筛选，
                # 就直接跳过，不要硬选一个左边大框冒充书。
                if debug:
                    print(f"[SEG] phrase={phrase_l!r} no box passed filter, skip this phrase")
                continue

            # 其他类别兜底：退回只按 score 选一个
            idxs = np.array([int(np.argmax(scores))])


        # 按 score 从高到低排序，截断到 max_keep
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        idxs = idxs[:max_keep]

        # ---- 只对选中的这些 box 跑 SAM，并到 final_mask 里 ----
        for i in idxs:
            box_xyxy = boxes[i].astype(np.float32)

            masks, sam_scores, _ = _SAM_PREDICTOR.predict(
                box=box_xyxy[None, :],
                multimask_output=True,
            )

            # === 和之前一样的“小目标优先选择小 mask”的逻辑 ===
            sam_areas = masks.reshape(masks.shape[0], -1).sum(axis=1)  # [K]
            x0b, y0b, x1b, y1b = box_xyxy
            box_area = max((x1b - x0b), 1) * max((y1b - y0b), 1)

            if any(k in phrase_l for k in ["book", "block", "can", "cup", "mug", "bottle"]):
                small_idx = np.where(sam_areas < 0.7 * box_area)[0]
                if small_idx.size > 0:
                    best = int(small_idx[np.argmax(sam_scores[small_idx])])
                else:
                    best = int(np.argmax(sam_scores))
            else:
                best = int(np.argmax(sam_scores))

            mask_i = masks[best].astype(bool)
            final_mask |= mask_i

            # 如果你在前面加了 all_instance_* 之类的记录，这里也别忘了 append
            all_instance_masks.append(mask_i)
            all_instance_boxes.append(tuple(map(float, box_xyxy.tolist())))
            all_instance_scores.append(float(scores[i]))
            all_instance_phrases.append(cleaned)



    if debug:
        print(f"[SEG] total_boxes={total_boxes}, max_score_all={max_score}", flush=True)

    if not final_mask.any():
        if debug:
            print("[SEG] final_mask is empty → return original image (no masking)", flush=True)
        return SegTestResult(
            mask=np.zeros_like(final_mask),
            soft_rgb=image_rgb.copy(),
            phrases=phrases,
            total_boxes=total_boxes,
            max_score=max_score,
            instance_masks=[],
            instance_boxes=[],
            instance_scores=[],
            instance_phrases=[],
        )

    mask = final_mask

    # ----- 3. 前景显著提亮 + 轻微上色，背景减弱（完全照 rollout） -----
    img_f = image_rgb.astype(np.float32)

    alpha_bg = float(soft_bg_alpha)    # 背景缩暗因子

    fg_gain = 1.6
    fg_bias = 40.0
    fg_tint = np.array([200.0, 200.0, 240.0], dtype=np.float32)  # 浅灰蓝

    img_out = img_f.copy()

    fg_linear = img_f[mask] * fg_gain + fg_bias
    img_out[mask] = 0.5 * fg_linear + 0.5 * fg_tint

    img_out[~mask] = img_f[~mask] * alpha_bg

    soft_rgb = np.clip(img_out, 0, 255).astype(np.uint8)

    # ----- 4. CLI 测试版也顺便保存一份 debug（三张图） -----
    if debug:
        _DEBUG_IMAGE_COUNT += 1
        try:
            os.makedirs(debug_dir, exist_ok=True)
            uid = np.random.randint(0, 10**8)

            Image.fromarray(image_rgb).save(os.path.join(debug_dir, f"orig_{uid}.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(debug_dir, f"mask_{uid}.png"))
            Image.fromarray(soft_rgb).save(os.path.join(debug_dir, f"soft_{uid}.png"))

            print(f"[SEG] Saved debug images to {debug_dir} (uid={uid})", flush=True)
        except Exception as e:
            print(f"[SEG][DEBUG] save images error: {e}", flush=True)

    return SegTestResult(
        mask=mask,
        soft_rgb=soft_rgb,
        phrases=phrases,
        total_boxes=total_boxes,
        max_score=max_score,
        instance_masks=all_instance_masks,
        instance_boxes=all_instance_boxes,
        instance_scores=all_instance_scores,
        instance_phrases=all_instance_phrases,
    )



# ========== CLI 部分：只测图片 ==========
def main():
    parser = argparse.ArgumentParser(
        description="Test GroundingDINO+SAM segmentation logic (same as rollout) on a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Instruction / task description, e.g. 'pick up the book on the right and place it on the cabinet shelf'",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=_SAM_CHECKPOINT_DEFAULT,
        help=f"Path to SAM checkpoint (default: {_SAM_CHECKPOINT_DEFAULT}).",
    )
    parser.add_argument(
        "--gdino_model_id",
        type=str,
        default=_GDINO_MODEL_ID_DEFAULT,
        help=f"HuggingFace id for GroundingDINO (default: {_GDINO_MODEL_ID_DEFAULT}).",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.20,
        help="Box score threshold for GroundingDINO.",
    )
    parser.add_argument(
        "--soft_bg_alpha",
        type=float,
        default=0.3,
        help="Background intensity scale (0~1). Smaller = 更暗背景。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="seg_test_out",
        help="Directory to save orig/mask/soft images.",
    )
    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="Disable extra debug save under seg_debug_cli/.",
    )

    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    img = _load_image_as_rgb(args.image)
    print(f"[INFO] loaded image: {args.image}, shape={img.shape}")

    result = simple_segment_gdino_sam(
        image_rgb=img,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        soft_bg_alpha=args.soft_bg_alpha,
        sam_checkpoint=args.sam_checkpoint,
        gdino_model_id=args.gdino_model_id,
        debug=(not args.no_debug),
    )

    base = os.path.splitext(os.path.basename(args.image))[0]
    prompt_tag = _sanitize_filename(args.prompt)

    # 保存三张图：原图 / mask / soft
    orig_out = os.path.join(args.out_dir, f"{base}__{prompt_tag}__orig.png")
    mask_out = os.path.join(args.out_dir, f"{base}__{prompt_tag}__mask.png")
    soft_out = os.path.join(args.out_dir, f"{base}__{prompt_tag}__soft.png")

    _save_image_rgb(img, orig_out)
    _save_mask_uint8(result.mask, mask_out)
    _save_image_rgb(result.soft_rgb, soft_out)

    print(f"[OK] Saved orig : {orig_out}")
    print(f"[OK] Saved mask : {mask_out}")
    print(f"[OK] Saved soft : {soft_out}")
    print(f"[INFO] phrases    : {result.phrases}")
    print(f"[INFO] total_boxes: {result.total_boxes}, max_score={result.max_score:.3f}")

    # ===== 额外：对每个 box / phrase 单独保存 mask 和 bbox 可视化 =====
    if result.instance_masks:
        for i, (m, box, score, phrase) in enumerate(
            zip(
                result.instance_masks,
                result.instance_boxes,
                result.instance_scores,
                result.instance_phrases,
            )
        ):
            ptag = _sanitize_filename(phrase)

            # 1) 保存这个 box 的单独 mask
            inst_mask_path = os.path.join(
                args.out_dir,
                f"{base}__{ptag}__box{i}_mask.png",
            )
            _save_mask_uint8(m, inst_mask_path)

            # 2) 在原图上只画出这个 box 的矩形框
            box_img = _draw_box_on_image(img, box, color=(255, 0, 0), width=2)
            inst_box_path = os.path.join(
                args.out_dir,
                f"{base}__{ptag}__box{i}_bbox.png",
            )
            _save_image_rgb(box_img, inst_box_path)

            print(
                f"[BOX {i}] phrase='{phrase}', "
                f"score={score:.3f}, "
                f"xyxy=({box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}), "
                f"mask_area={m.sum()} px"
            )
            print(f"        -> mask_png: {inst_mask_path}")
            print(f"        -> bbox_png: {inst_box_path}")
    else:
        print("[INFO] no instance_masks recorded (no boxes passed threshold).")
    


if __name__ == "__main__":
    main()
