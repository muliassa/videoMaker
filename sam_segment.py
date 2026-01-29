#!/usr/bin/env python3
"""
SAM segmentation helper script
Usage: python sam_segment.py <input_image> <output_mask> <x> <y> <w> <h>
"""

import sys
import os
import numpy as np
import cv2

# Look for models in these directories
MODEL_DIRS = ["models", ".", ".."]

def find_checkpoint():
    """Find SAM checkpoint in model directories"""
    checkpoints = [
        ("sam_vit_h_4b8939.pth", "vit_h"),
        ("sam_vit_l_0b3195.pth", "vit_l"),
        ("sam_vit_b_01ec64.pth", "vit_b"),
        ("sam3.pt", "vit_h"),
    ]
    
    for dir in MODEL_DIRS:
        for ckpt, model_type in checkpoints:
            path = os.path.join(dir, ckpt)
            if os.path.exists(path):
                return path, model_type
    
    return None, None

def segment_with_sam(input_path, output_path, bbox):
    """Segment using SAM with bbox prompt"""
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        import torch
    except ImportError:
        print("ERROR: segment-anything not installed")
        print("Run: pip install segment-anything torch torchvision")
        sys.exit(1)
    
    checkpoint, model_type = find_checkpoint()
    if not checkpoint:
        print("ERROR: No SAM checkpoint found")
        print("Place sam_vit_h_4b8939.pth in ./models/ directory")
        sys.exit(1)
    
    print(f"Loading SAM: {checkpoint} ({model_type})")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    
    if torch.cuda.is_available():
        sam.to("cuda")
        print("Using CUDA")
    
    predictor = SamPredictor(sam)
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print(f"ERROR: Cannot load {input_path}")
        sys.exit(1)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    # Box prompt [x1, y1, x2, y2]
    x, y, w, h = bbox
    box = np.array([x, y, x + w, y + h])
    
    # Predict
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )
    
    # Best mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, mask_uint8)
    print(f"Saved: {output_path} (score: {scores[best_idx]:.3f})")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python sam_segment.py <input> <output_mask> <x> <y> <w> <h>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    bbox = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    
    segment_with_sam(input_path, output_path, bbox)
