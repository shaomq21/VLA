#!/usr/bin/env python3
"""
Test Roboflow gripper detection.

Usage:
  cd /home/ubuntu/16831pro_fine_tune/openvla-oft
  PYTHONPATH=. python test_roboflow_gripper.py [--image /path/to/image.png]

Requires: pip install inference, ROBOFLOW_API_KEY env var.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


def test_roboflow_gripper(image_path: str = None, model_id: str = "gripper_box/1") -> bool:
    """Test Roboflow gripper model. Returns True if OK, False otherwise."""
    # 1. Check API key
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("[FAIL] ROBOFLOW_API_KEY not set. Set it for hosted model access.")
        return False
    print("[OK] ROBOFLOW_API_KEY is set")

    # 2. Try import
    try:
        from inference import get_model
    except ImportError as e:
        print(f"[FAIL] inference package not found: {e}")
        print("  Install with: pip install inference")
        return False
    print("[OK] inference package imported")

    # 3. Load model
    try:
        model = get_model(model_id=model_id)
        print(f"[OK] Model {model_id} loaded")
    except Exception as e:
        print(f"[FAIL] Failed to load model {model_id}: {e}")
        return False

    # 4. Run inference on a test image
    if image_path and os.path.isfile(image_path):
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        # Create a minimal 256x256 RGB image (black) as fallback
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        print(f"[INFO] No image provided, using 256x256 black image")

    try:
        result = model.infer(img)
        if isinstance(result, (list, tuple)) and len(result) > 0:
            result = result[0]
        centers = []
        if hasattr(result, "predictions") and result.predictions:
            for p in result.predictions:
                if hasattr(p, "x") and hasattr(p, "y"):
                    centers.append((int(round(p.x)), int(round(p.y))))
                elif hasattr(p, "bbox"):
                    b = p.bbox
                    cx = int(round((b[0] + b[2]) / 2))
                    cy = int(round((b[1] + b[3]) / 2))
                    centers.append((cx, cy))
        print(f"[OK] Inference ran. Found {len(centers)} gripper center(s): {centers}")
        return True
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Test Roboflow gripper detection")
    ap.add_argument("--image", type=str, default=None, help="Path to test image (optional)")
    ap.add_argument("--model_id", type=str, default="gripper_box/1", help="Roboflow model id")
    args = ap.parse_args()

    ok = test_roboflow_gripper(args.image, args.model_id)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
