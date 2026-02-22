#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any, List

import numpy as np
from PIL import Image


def _ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _sanitize_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _overlay_mask_rgba(
    image_rgb: np.ndarray,
    mask_bool: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    # Convert BGR to RGB for overlay color
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    overlay = image_rgb.copy()
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool] + alpha * np.array(color_rgb, dtype=np.float32)
    )
    return overlay.astype(np.uint8)


def _load_image_as_rgb(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return np.array(img)


def _save_image_rgb(image_rgb: np.ndarray, path: str) -> None:
    Image.fromarray(image_rgb).save(path)


def _save_mask_uint8(mask_bool: np.ndarray, path: str) -> None:
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)


@dataclass
class SegmentationResult:
    mask: np.ndarray  # bool array HxW
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    overlay_rgb: Optional[np.ndarray]
    backend: str
    text_backend: str
    # Optional multi-instance info
    instance_masks: Optional[List[np.ndarray]] = None
    instance_boxes: Optional[List[Tuple[float, float, float, float]]] = None
    instance_scores: Optional[List[float]] = None
    instance_phrases: Optional[List[str]] = None


def _predict_boxes_with_owlvit(
    image_rgb: np.ndarray,
    prompt: str,
    box_threshold: float = 0.20,
    device: Optional[str] = None,
    model_name: str = "google/owlvit-base-patch32",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        boxes_xyxy: Float tensor (N, 4)
        scores: Float tensor (N,)
    """
    import torch
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model first to read text encoder's max positions
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
    processor = OwlViTProcessor.from_pretrained(model_name)

    # OwlViT text encoder has a short context length (often 16).
    # Ensure tokenizer truncates/pads to the exact maximum to avoid dim mismatch.
    try:
        max_text_len = int(model.config.text_config.max_position_embeddings)
    except Exception:
        # Fallback to 16 if not available
        max_text_len = 16

    # Make prompt tokenizer-friendly: replace underscores by spaces
    cleaned_prompt = prompt.replace("_", " ").strip()
    pil = Image.fromarray(image_rgb)
    inputs = processor(
        text=[cleaned_prompt],
        images=pil,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([pil.size[::-1]]).to(device)  # (H, W)
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=box_threshold
    )[0]
    boxes = results["boxes"]  # (N, 4) xyxy
    scores = results["scores"]
    return boxes.detach().cpu(), scores.detach().cpu()


def _predict_boxes_with_groundingdino(
    image_rgb: np.ndarray,
    prompt: str,
    *,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: Optional[str] = None,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        boxes_xyxy: Float tensor (N, 4)
        scores: Float tensor (N,)
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    pil = Image.fromarray(image_rgb)
    # Replace underscores for better tokenization
    cleaned_prompt = prompt.replace("_", " ").strip()
    inputs = processor(images=pil, text=cleaned_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    h, w = image_rgb.shape[:2]
    target_sizes = torch.tensor([(h, w)]).to(device)
    # Post-process per HF API. Be robust to transformers version differences and retry on empty detections.
    def _post_process(_outputs, _inputs_ids, _box_th, _text_th):
        try:
            results = processor.post_process_grounded_object_detection(
                _outputs,
                _inputs_ids,
                box_threshold=_box_th,
                text_threshold=_text_th,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                _outputs,
                _inputs_ids,
                target_sizes=target_sizes,
                threshold=_box_th,
            )[0]
        return results

    results = _post_process(outputs, inputs["input_ids"], box_threshold, text_threshold)

    # Fallback: if no boxes, try relaxed thresholds and ensure trailing period (common GDINO convention)
    if results["boxes"].shape[0] == 0:
        relaxed_box_th = max(0.05, box_threshold * 0.5)
        relaxed_text_th = max(0.05, text_threshold * 0.7)
        retry_text = cleaned_prompt if cleaned_prompt.endswith(".") else (cleaned_prompt + ".")
        inputs2 = processor(images=pil, text=retry_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs2 = model(**inputs2)
        results = _post_process(outputs2, inputs2["input_ids"], relaxed_box_th, relaxed_text_th)

    boxes = results["boxes"]  # (N, 4) xyxy in pixel coords
    scores = results["scores"]  # (N,)
    return boxes.detach().cpu(), scores.detach().cpu()


def _segment_with_sam(
    image_rgb: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    sam_checkpoint: str,
    model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_h",
    device: Optional[str] = None,
) -> np.ndarray:
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]  # HxW bool
    return mask


def _segment_with_grabcut(
    image_rgb: np.ndarray, box_xyxy: Tuple[float, float, float, float], iterations: int = 5
) -> np.ndarray:
    # Use OpenCV GrabCut as a light-weight mask refinement over a bounding box
    import cv2

    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = map(int, box_xyxy)
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))

    mask = np.zeros((h, w), np.uint8)  # 0: bg, 1: fg, 2: prob bg, 3: prob fg
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (x0, y0, x1 - x0, y1 - y0)

    cv2.grabCut(image_rgb, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
    mask_fg = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
    return mask_fg.astype(bool)


def _choose_box(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    strategy: Literal["best", "leftmost", "largest"] = "best",
) -> Tuple[Tuple[float, float, float, float], float]:
    if boxes_xyxy.shape[0] == 0:
        return (0.0, 0.0, 0.0, 0.0), 0.0
    if strategy == "best":
        idx = int(np.argmax(scores))
    elif strategy == "leftmost":
        idx = int(np.argmin(boxes_xyxy[:, 0]))
    elif strategy == "largest":
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        idx = int(np.argmax(areas))
    else:
        idx = int(np.argmax(scores))
    box = tuple(map(float, boxes_xyxy[idx].tolist()))
    score = float(scores[idx])
    return box, score


def _choose_topk_boxes(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    k: int,
    strategy: Literal["best", "leftmost", "rightmost", "largest", "center"] = "best",
    image_shape: Optional[Tuple[int, int]] = None,
) -> List[int]:
    if boxes_xyxy.shape[0] == 0:
        return []
    k = max(1, min(k, boxes_xyxy.shape[0]))
    if strategy == "best":
        order = np.argsort(-scores)  # descending
    elif strategy == "leftmost":
        order = np.argsort(boxes_xyxy[:, 0])
    elif strategy == "rightmost":
        order = np.argsort(-boxes_xyxy[:, 0])
    elif strategy == "largest":
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        order = np.argsort(-areas)
    elif strategy == "center":
        if image_shape is None:
            order = np.argsort(np.abs((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5))
        else:
            h, w = image_shape
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
            order = np.argsort(np.abs(cx - (w * 0.5)))
    else:
        order = np.argsort(-scores)
    return order[:k].tolist()


def _infer_strategy_from_text(text: str, default: str = "best") -> str:
    # t = text.lower()
    # if "right" in t:
    #     return "rightmost"
    # if "left" in t:
    #     return "leftmost"
    # if "middle" in t or "center" in t or "centre" in t:
    #     return "center"
    return default


def segment_image_by_prompt(
    image_path: str,
    prompt: str,
    *,
    text_backend: Literal["owlvit", "groundingdino"] = "owlvit",
    seg_backend: Literal["grabcut", "sam"] = "grabcut",
    select_strategy: Literal["best", "leftmost", "rightmost", "largest", "center"] = "best",
    box_threshold: float = 0.20,
    gdino_model_id: str = "IDEA-Research/grounding-dino-tiny",
    gdino_text_threshold: float = 0.25,
    phrases: Optional[List[str]] = None,
    split_on_and: bool = False,
    top_k_per_phrase: int = 1,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_h",
    overlay_color_bgr: Tuple[int, int, int] = (0, 0, 255),
    overlay_alpha: float = 0.5,
    device: Optional[str] = None,
    return_overlay: bool = True,
) -> SegmentationResult:
    """
    Segments an object referred by a text prompt and optionally overlays the mask.
    """
    image_rgb = _load_image_as_rgb(image_path)

    # Prepare phrases
    if phrases is not None and len(phrases) > 0:
        phrase_list = phrases
    else:
        cleaned = prompt.replace("_", " ").strip()
        if split_on_and:
            # naive split on ' and '
            phrase_list = [p.strip() for p in re.split(r"\band\b", cleaned) if p.strip()]
        else:
            phrase_list = [cleaned]

    H, W = image_rgb.shape[:2]
    all_instance_masks: List[np.ndarray] = []
    all_instance_boxes: List[Tuple[float, float, float, float]] = []
    all_instance_scores: List[float] = []
    all_instance_phrases: List[str] = []

    for phrase in phrase_list:
        # 1) Text-guided detection per phrase
        if text_backend == "owlvit":
            boxes_xyxy, scores = _predict_boxes_with_owlvit(
                image_rgb=image_rgb,
                prompt=phrase,
                box_threshold=box_threshold,
                device=device,
            )
        elif text_backend == "groundingdino":
            boxes_xyxy, scores = _predict_boxes_with_groundingdino(
                image_rgb=image_rgb,
                prompt=phrase,
                box_threshold=box_threshold,
                text_threshold=gdino_text_threshold,
                device=device,
                model_id=gdino_model_id,
            )
        else:
            raise ValueError(f"Unsupported text backend: {text_backend}")

        boxes_np = boxes_xyxy.numpy()
        scores_np = scores.numpy()
        if boxes_np.shape[0] == 0:
            continue

        # Infer strategy hints from phrase if includes left/right/center
        local_strategy = _infer_strategy_from_text(phrase, default=select_strategy)
        sel_indices = _choose_topk_boxes(
            boxes_np, scores_np, top_k_per_phrase, local_strategy, image_shape=(H, W)
        )
        for idx in sel_indices:
            box_xyxy = tuple(map(float, boxes_np[idx].tolist()))
            score = float(scores_np[idx])

            # 2) Segmentation from each selected box
            if seg_backend == "sam":
                if not sam_checkpoint:
                    raise ValueError("sam_checkpoint must be provided for seg_backend='sam'.")
                mask_bool = _segment_with_sam(
                    image_rgb=image_rgb,
                    box_xyxy=box_xyxy,
                    sam_checkpoint=sam_checkpoint,
                    model_type=sam_model_type,
                    device=device,
                )
            elif seg_backend == "grabcut":
                mask_bool = _segment_with_grabcut(image_rgb=image_rgb, box_xyxy=box_xyxy, iterations=5)
            else:
                raise ValueError(f"Unsupported seg backend: {seg_backend}")

            all_instance_masks.append(mask_bool)
            all_instance_boxes.append(box_xyxy)
            all_instance_scores.append(score)
            all_instance_phrases.append(phrase)

    if len(all_instance_masks) == 0:
        raise RuntimeError("No boxes detected for the given prompt(s) and threshold(s).")

    # Merge masks
    merged_mask = np.zeros((H, W), dtype=bool)
    for m in all_instance_masks:
        merged_mask |= m

    overlay = None
    if return_overlay:
        overlay = _overlay_mask_rgba(
            image_rgb=image_rgb,
            mask_bool=merged_mask,
            color_bgr=overlay_color_bgr,
            alpha=overlay_alpha,
        )

    return SegmentationResult(
        mask=merged_mask,
        bbox_xyxy=all_instance_boxes[0],
        score=all_instance_scores[0],
        overlay_rgb=overlay,
        backend=seg_backend,
        text_backend=text_backend,
        instance_masks=all_instance_masks,
        instance_boxes=all_instance_boxes,
        instance_scores=all_instance_scores,
        instance_phrases=all_instance_phrases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text-prompt segmentation with overlay and optional negative mask."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path. Example: /home/hongyi/16831pro/LIBERO/benchmark_tasks/your.png",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt, e.g. 'put_the_can_on_the_left_in_the_basket'",
    )
    parser.add_argument(
        "--text_backend",
        type=str,
        default="owlvit",
        choices=["owlvit", "groundingdino"],
        help="Text-to-box backend. 'owlvit' is default and light-weight; 'groundingdino' supports stronger grounding.",
    )
    parser.add_argument(
        "--seg_backend",
        type=str,
        default="grabcut",
        choices=["grabcut", "sam"],
        help="Segmentation backend. 'grabcut' needs no extra weights; 'sam' requires checkpoint.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (required if --seg_backend sam).",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type.",
    )
    parser.add_argument(
        "--select_strategy",
        type=str,
        default="best",
        choices=["best", "leftmost", "rightmost", "largest", "center"],
        help="How to select a single box if multiple detections are found.",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.20,
        help="Score threshold for text detector (OWL-ViT).",
    )
    parser.add_argument(
        "--gdino_model_id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face model id for Grounding DINO (if --text_backend groundingdino).",
    )
    parser.add_argument(
        "--gdino_text_threshold",
        type=float,
        default=0.25,
        help="Text threshold for Grounding DINO phrase grounding.",
    )
    parser.add_argument(
        "--phrases",
        type=str,
        nargs="+",
        default=None,
        help='Optional list of phrases (e.g., can "on the right" basket). If set, overrides --prompt.',
    )
    parser.add_argument(
        "--split_on_and",
        action="store_true",
        help='If set and --phrases not provided, split prompt by "and" into multiple phrases.',
    )
    parser.add_argument(
        "--top_k_per_phrase",
        type=int,
        default=1,
        help="Select top-k instances per phrase (after applying selection strategy).",
    )
    parser.add_argument(
        "--save_instance_masks",
        action="store_true",
        help="Save individual instance masks as separate files.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.5,
        help="Overlay transparency (0.0 ~ 1.0).",
    )
    parser.add_argument(
        "--overlay_color_bgr",
        type=int,
        nargs=3,
        default=(0, 0, 255),
        help="Overlay color in BGR (three ints). Default red mask: 0 0 255",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/hongyi/16831pro/LIBERO/benchmark_tasks",
        help="Directory to save results (overlay and masks).",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="Save positive mask as *_mask.png (255 where object, 0 elsewhere).",
    )
    parser.add_argument(
        "--save_negative_mask",
        action="store_true",
        help="Save negative mask as *_mask_neg.png (255 where NOT object, 0 on object).",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Disable saving overlay image; only output masks if requested.",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    prompt_tag = _sanitize_filename(args.prompt)

    result = segment_image_by_prompt(
        image_path=args.image,
        prompt=args.prompt,
        text_backend=args.text_backend,
        seg_backend=args.seg_backend,
        select_strategy=args.select_strategy,
        box_threshold=args.box_threshold,
        gdino_model_id=args.gdino_model_id,
        gdino_text_threshold=args.gdino_text_threshold,
        phrases=args.phrases,
        split_on_and=args.split_on_and,
        top_k_per_phrase=args.top_k_per_phrase,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        overlay_color_bgr=tuple(args.overlay_color_bgr),
        overlay_alpha=args.overlay_alpha,
        return_overlay=not args.no_overlay,
    )

    # Save overlay
    if result.overlay_rgb is not None and not args.no_overlay:
        overlay_path = os.path.join(
            args.out_dir, f"{base_name}__{prompt_tag}__overlay.png"
        )
        _save_image_rgb(result.overlay_rgb, overlay_path)
        print(f"[OK] Saved overlay: {overlay_path}")

    # Save masks (positive and/or negative)
    if args.save_mask:
        mask_path = os.path.join(args.out_dir, f"{base_name}__{prompt_tag}__mask.png")
        _save_mask_uint8(result.mask, mask_path)
        print(f"[OK] Saved mask: {mask_path}")

    # Save individual instance masks if requested
    if args.save_instance_masks and result.instance_masks is not None:
        for i, (m, phrase) in enumerate(zip(result.instance_masks, result.instance_phrases or [])):
            ptag = _sanitize_filename(phrase) if phrase else f"phrase{i}"
            ipath = os.path.join(args.out_dir, f"{base_name}__{ptag}__mask_{i}.png")
            _save_mask_uint8(m, ipath)
            print(f"[OK] Saved instance mask [{i}]: {ipath}")

    if args.save_negative_mask:
        neg_mask = ~result.mask
        neg_mask_path = os.path.join(
            args.out_dir, f"{base_name}__{prompt_tag}__mask_neg.png"
        )
        _save_mask_uint8(neg_mask, neg_mask_path)
        print(f"[OK] Saved negative mask: {neg_mask_path}")

    # Print selected instances for logging/integration
    if result.instance_boxes:
        print(f"[INFO] instances={len(result.instance_boxes)}, text_backend={result.text_backend}, seg_backend={result.backend}")
        for i, (b, s, ph) in enumerate(zip(result.instance_boxes, result.instance_scores or [], result.instance_phrases or [])):
            x0, y0, x1, y1 = b
            print(f"  - [{i}] phrase='{ph}' bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) score={s:.3f}")
    else:
        x0, y0, x1, y1 = result.bbox_xyxy
        print(
            f"[INFO] bbox_xyxy=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}), "
            f"score={result.score:.3f}, text_backend={result.text_backend}, seg_backend={result.backend}"
        )


if __name__ == "__main__":
    main()


