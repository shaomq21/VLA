# masked_grounded_sam.py
# ------------------------------------------------------------
# Grounded-SAM masking for OpenVLA RLDS images
# - background black
# - source objects painted RED, target objects painted GREEN
# - gripper centers from Roboflow model as white dots (optional)
#
# Gripper detection requires: pip install inference
# Set ROBOFLOW_API_KEY for hosted model: https://app.roboflow.com/margarets-workspace/gripper_box/models
# ------------------------------------------------------------

from __future__ import annotations

import os

from future.types import disallow_types

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
#from debian.debtags import output

# ========== GroundingDINO ==========
# You need to have GroundingDINO repo installed/importable.
# Common import pattern:
#   git clone https://github.com/IDEA-Research/GroundingDINO.git
#   pip install -e GroundingDINO
from groundingdino.util.inference import Model as GroundingDINOModel
#from orca.orca_state import device
#from reportlab.rl_settings import imageReaderFlags

# ========== SAM ==========
# You need SAM installed:
#   git clone https://github.com/facebookresearch/segment-anything.git
#   pip install -e segment-anything
from segment_anything import sam_model_registry, SamPredictor


# --------------------------
# 1) Language parsing rules
# --------------------------

@dataclass
class MaskSpec:
    red_phrases: List[str]
    green_phrases: List[str]
    red_points_xy: List[Tuple[float, float]] = None   
    green_points_xy: List[Tuple[float, float]] = None




def build_mask_spec_from_lang(lang: str) -> MaskSpec:
    """
    Your rules (as you described):
      - open ... : green mask drawers (middle/top drawer)
      - push ... : red mask plate + table region in front of stove (approx via 'plate' + 'table'/'stove front')
      - put ...  : red mask the grasped object (bowl/cream cheese/wine bottle),
                  green mask destination (stove/cabinet/rack/plate/bowl) depending on phrase
      - turn on  : mask stove (green)
    """


    
    # OPEN
    if lang.startswith("open "):
        
        green_points = []

        
        TOP_HANDLE    = (0.71, 0.58)
       
        MIDDLE_HANDLE = (0.71, 0.63)

        if "top drawer" in lang:
            green_points = [TOP_HANDLE]
        elif "middle drawer" in lang:
            green_points = [MIDDLE_HANDLE]
        elif "drawer" in lang:
            green_points = [MIDDLE_HANDLE]

        return MaskSpec(
            red_phrases=[],
            green_phrases=[],          
            red_points_xy=[],
            green_points_xy=green_points,
        )

    # PUSH
    if lang.startswith("push "):
        # Your request: red mask plate + the table area before "front of the stove"
        # In practice: Grounded-SAM can't segment "table region in front of stove" perfectly via text,
        # so we approximate with "plate" (strong) and "table" or "stove" (weak).
        reds = []
        greens = []

        if "plate" in lang:
            reds.append("plate")
        # Stove: try both "white rectangular" (light) and "stove" (LIBERO stove is dark brown/black)
        greens += ["white rectangular box on the left", "stove"]
        return MaskSpec(red_phrases=reds, green_phrases=greens)

    # PUT
    if lang.startswith("put "):
        # parse: "put the X on/in/inside/on top of the Y"
        # red := X
        # green := Y (destination)
        # handle:
        #   "put the cream cheese in the bowl"
        #   "put the wine bottle on the rack"
        #   "put the bowl on the stove" etc.
        # We'll do a simple regex.
        # Note: dataset language is "put the bowl on the plate" style.

        red_obj = None
        green_obj = None

        m = re.match(r"put the (.+?) (on top of|on|in|inside) the (.+)$", lang)
        if m:
            red_obj = m.group(1).strip()
            green_obj = m.group(3).strip()
        else:
            # fallback: try "put the X" only
            m2 = re.match(r"put the (.+)$", lang)
            if m2:
                red_obj = m2.group(1).strip()
                
        reds = [red_obj] if red_obj else []
        greens = [green_obj] if green_obj else []

        red_points = []
        if red_obj and "cream cheese" in red_obj:
            reds = [] 
            red_points = [(0.33, 0.6)] 


        return MaskSpec(
        red_phrases=reds,
        green_phrases=greens,
        red_points_xy=red_points,
        green_points_xy=[],
    )

    # TURN ON  (stove -> 左边的扁方块)
    if lang.startswith("turn on "):
        return MaskSpec(red_phrases=[], green_phrases=["white rectangular box on the left"])

    # default: no masks
    return MaskSpec(red_phrases=[], green_phrases=[])


# --------------------------
# 2) Grounded-SAM wrapper
# --------------------------

@dataclass
class GroundedSAMConfig:
    # GroundingDINO
    dino_config_path: str
    dino_checkpoint_path: str
    box_threshold: float = 0.30
    text_threshold: float = 0.25

    # SAM
    sam_type: str = "vit_h"  # vit_h / vit_l / vit_b
    sam_checkpoint_path: str = ""

    # Gripper detection via Roboflow (optional)
    # Model: 需在 app.roboflow.com 创建，格式 workspace/project/version
    gripper_model_id: Optional[str] = "gripper_box/1"
    gripper_enabled: bool = True

    device: str = "cuda"


from PIL import ImageDraw

# Lazy-load Roboflow inference for gripper detection
_gripper_model = None

def _get_gripper_model(model_id: str):
    """Lazy load Roboflow gripper model. Requires: pip install inference"""
    global _gripper_model
    if _gripper_model is None:
        try:
            from inference import get_model
            _gripper_model = get_model(model_id=model_id)
        except ImportError:
            raise ImportError(
                "Gripper detection requires 'inference' package. Install with: pip install inference"
            )
    return _gripper_model


def _detect_gripper_centers(image_rgb: np.ndarray, model_id: str) -> List[Tuple[int, int]]:
    """
    Run Roboflow gripper model; returns center points (x,y) of bounding boxes in pixel coords.
    Typically returns 2 boxes (sometimes 1). Uses ObjectDetectionPrediction.x, .y (center coords).
    """
    model = _get_gripper_model(model_id)
    result = model.infer(image_rgb)
    # infer() can return list for batch; take first element
    if isinstance(result, (list, tuple)) and len(result) > 0:
        result = result[0]
    centers = []
    if hasattr(result, "predictions") and result.predictions:
        for p in result.predictions:
            if hasattr(p, "x") and hasattr(p, "y"):
                cx = int(round(p.x))
                cy = int(round(p.y))
                centers.append((cx, cy))
            elif hasattr(p, "bbox"):
                b = p.bbox  # x_min, y_min, x_max, y_max
                cx = int(round((b[0] + b[2]) / 2))
                cy = int(round((b[1] + b[3]) / 2))
                centers.append((cx, cy))
    return centers


def _draw_white_dots(img_arr: np.ndarray, centers_xy: List[Tuple[int, int]], radius: int = 12) -> np.ndarray:
    """Draw filled white circles at center points on image. Returns modified array."""
    if not centers_xy:
        return img_arr
    img_pil = Image.fromarray(img_arr, mode="RGB")
    draw = ImageDraw.Draw(img_pil)
    for (cx, cy) in centers_xy:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255), outline=(255, 255, 255))
    return np.array(img_pil)


def _draw_points_overlay(img_pil: Image.Image, points_xy, *, color=(255, 0, 0), r=8, w=3):
    
    if not points_xy:
        return img_pil

    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    for (px, py) in points_xy:
        x = int(round(px * W))
        y = int(round(py * H))

        
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=w)

        
        draw.line((x - 2*r, y, x + 2*r, y), fill=color, width=w)
        draw.line((x, y - 2*r, x, y + 2*r), fill=color, width=w)

        
        draw.text((x + r + 2, y + r + 2), f"({x},{y})", fill=color)

    return img


class GroundedSAMMasker:
    def __init__(self, cfg: GroundedSAMConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Resolve GroundingDINO config path (use package config if local path does not exist)
        dino_config_path = cfg.dino_config_path
        if not os.path.isfile(dino_config_path):
            import groundingdino
            _pkg_dir = os.path.dirname(os.path.abspath(groundingdino.__file__))
            dino_config_path = os.path.join(_pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")
        if not os.path.isfile(dino_config_path):
            raise FileNotFoundError(
                f"GroundingDINO config not found at {cfg.dino_config_path} nor at {dino_config_path}"
            )

        # GroundingDINO model
        self.dino = GroundingDINOModel(
            model_config_path=dino_config_path,
            model_checkpoint_path=cfg.dino_checkpoint_path,
            device=str(self.device),
        )

        # SAM predictor
        sam = sam_model_registry[cfg.sam_type](checkpoint=cfg.sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    @torch.inference_mode()
    def _segment_phrases(self, image_rgb: np.ndarray, phrases: List[str], lang: str = "") -> np.ndarray:
        """
        Returns a union mask (H,W) bool for all phrases.
        """
        if not phrases:
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        H, W = image_rgb.shape[:2]
        union = np.zeros((H, W), dtype=bool)
        # 任务里没有 cabinet/rack 时，不 box 右边的物体（避免误框柜子）
        skip_right = ("cabinet" not in lang.lower()) and ("rack" not in lang.lower())

        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue

            # GroundingDINO expects "phrase." sometimes; keep it robust
            prompt = phrase if phrase.endswith(".") else (phrase + ".")

            # Predict boxes with GroundingDINO
            detections = self.dino.predict_with_classes(
                image=image_rgb,
                classes=[prompt],
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
            )
            # detections.xyxy is (N,4) in pixel coords
            if detections is None or len(detections) == 0:
                continue

            boxes_xyxy = detections.xyxy
            if boxes_xyxy is None or len(boxes_xyxy) == 0:
                continue

            # plate 检测出两个（盘+碗）时，取下面的 mask（盘子）
            if "plate" in phrase.lower() and len(boxes_xyxy) > 1:
                bottommost_idx = np.argmax(boxes_xyxy[:, 3])  # y2 最大 = 最下面
                boxes_xyxy = np.array([boxes_xyxy[bottommost_idx]], dtype=boxes_xyxy.dtype)

            # stove/左边扁方块：多个框时取最左边的一个
            if ("on the left" in phrase.lower() or "left" in phrase.lower()) and len(boxes_xyxy) > 1:
                leftmost_idx = np.argmin((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2)
                boxes_xyxy = np.array([boxes_xyxy[leftmost_idx]], dtype=boxes_xyxy.dtype)

            # 任务无 cabinet/rack 时，排除靠右的 box（多是柜子误检）
            if skip_right and len(boxes_xyxy) > 0:
                cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
                keep_mask = cx <= 0.6 * W
                boxes_xyxy = boxes_xyxy[keep_mask]
            if len(boxes_xyxy) == 0:
                continue

            # SAM expects boxes as torch tensor on device in XYXY
            boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=self.device)

            # Transform boxes to SAM input space
            transformed = self.sam_predictor.transform.apply_boxes_torch(boxes_t, (H, W))

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
            # masks: (N,1,H,W) bool tensor
            m = masks.squeeze(1).detach().cpu().numpy().astype(bool)  # (N,H,W)
            union |= m.any(axis=0)

        return union

    @torch.inference_mode()
    def _segment_points(self, image_rgb: np.ndarray, points_xy):
        """
        Use point + small box constraint.
        """
        if not points_xy:
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        H, W = image_rgb.shape[:2]
        

        px, py = points_xy[0]   
        cx = int(px * W)
        cy = int(py * H)

       
        box_half = 40   

        x1 = max(0, cx - box_half)
        y1 = max(0, cy - box_half)
        x2 = min(W, cx + box_half)
        y2 = min(H, cy + box_half)

        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        pts = np.array([[cx, cy]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=pts,
            point_labels=labels,
            box=box,
            multimask_output=False,
        )

        return masks[0].astype(bool)



    def mask_image_from_lang(
        self,
        img_pil: Image.Image,
        lang: str,
        *,
        return_masks: bool = False,
        alpha: float = 0.35
    ):
        """
        Output: PIL RGB image with black background and colored masks:
          - red objects -> (255,0,0)
          - green objects -> (0,255,0)

        If return_masks=True, also returns (red_mask, green_mask).
        """
        # stove -> 左边的扁方块（白色矩形盒）
        lang = lang.replace("stove", "white rectangular box on the left")

        lang = lang.replace("rack",
                          "the yellow and white striped rack near the edge of the table")
        lang = lang.replace("wine bottle","right wine bottle")
        spec = build_mask_spec_from_lang(lang)

        image_rgb = np.array(img_pil.convert("RGB"),dtype=np.uint8)
        self.sam_predictor.set_image(image_rgb)

        # === RED ===
        if spec.red_points_xy:                      
            red_mask = self._segment_points(image_rgb, spec.red_points_xy)
        else:
            red_mask = self._segment_phrases(image_rgb, spec.red_phrases, lang=lang)

        # === GREEN ===
        if spec.green_points_xy:                      
            green_mask = self._segment_points(image_rgb, spec.green_points_xy)
        else:
            green_mask = self._segment_phrases(image_rgb, spec.green_phrases, lang=lang)

        if spec.red_points_xy or spec.green_points_xy:
            debug_rgb = Image.fromarray(image_rgb.copy(), mode="RGB")
            if getattr(spec, "green_points_xy", None):
                debug_rgb = _draw_points_overlay(debug_rgb, spec.green_points_xy, color=(0, 255, 0), r=10, w=3)
            if getattr(spec, "red_points_xy", None):
                debug_rgb = _draw_points_overlay(debug_rgb, spec.red_points_xy, color=(255, 0, 0), r=10, w=3)

            os.makedirs("debug_points", exist_ok=True)
            safe = re.sub(r"[^a-zA-Z0-9_]+", "_", lang)[:120]
            debug_rgb.save(f"debug_points/{safe}_points.png")


        # Compose: black background
        out = np.zeros_like(image_rgb, dtype=np.uint8)
        light_red = np.array([255, 120, 120], dtype=np.float32)
        light_green = np.array([120, 255, 120], dtype=np.float32)
        red_mask = red_mask & (~green_mask)
        if red_mask.any():
            tinted = (1.0 - alpha) * image_rgb[red_mask] + alpha * light_red
            out[red_mask] = np.clip(tinted, 0, 255).astype(np.uint8)
        if green_mask.any():
            tinted = (1.0 - alpha) * image_rgb[green_mask] + alpha * light_green
            out[green_mask] = np.clip(tinted, 0, 255).astype(np.uint8)

        # Gripper detection: draw center points as white dots on black mask
        gripper_centers = []
        if getattr(self.cfg, "gripper_enabled", True) and getattr(self.cfg, "gripper_model_id", None):
            try:
                gripper_centers = _detect_gripper_centers(image_rgb, self.cfg.gripper_model_id)
                if gripper_centers:
                    H, W = image_rgb.shape[:2]
                    radius = max(2, min(5, min(H, W) // 70))
                    out = _draw_white_dots(out, gripper_centers, radius=radius)
            except Exception as e:
                import warnings
                warnings.warn(f"Gripper detection failed: {e}")

        # If nothing was drawn (all black): return original image
        if not red_mask.any() and not green_mask.any() and not gripper_centers:
            return img_pil

        out_pil = Image.fromarray(out, mode="RGB")
        
        

        if getattr(spec, "red_points_xy", None):
            out_pil = _draw_points_overlay(out_pil, spec.red_points_xy, color=(255, 0, 0), r=10, w=3)
        if getattr(spec, "green_points_xy", None):
            out_pil = _draw_points_overlay(out_pil, spec.green_points_xy, color=(0, 255, 0), r=10, w=3)



        return out_pil


# --------------------------
# 3) Minimal CLI test
# --------------------------

def main():
    """
    Example:
      python masked_grounded_sam.py --image /path/to/frame.png --lang "put the bowl on the stove"
    """
    import argparse

    ap = argparse.ArgumentParser()






    dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_ckpt = "groundingdino_swint_ogc.pth"
    sam_ckpt = "sam_vit_h_4b8939.pth"
    sam_type = "vit_h"
    #sam_type = "vit_b"
    device = "cuda"
    out_path = "/home/ubuntu/16831pro_fine_tune/zz/masked_cheese.png"

    cfg = GroundedSAMConfig(
        dino_config_path=dino_config,
        dino_checkpoint_path=dino_ckpt,
        sam_checkpoint_path=sam_ckpt,
        sam_type=sam_type,
        device=device,
    )
    lang = "put the cream cheese in the bowl"
    image = "/home/ubuntu/16831pro_fine_tune/zt/plate.jpg"
    masker = GroundedSAMMasker(cfg)
    img = Image.open(image).convert("RGB")
    out = masker.mask_image_from_lang(img, lang)

    out.save(out_path)
    print(f"[OK] Saved masked image to: {out_path}")


if __name__ == "__main__":
    main()


