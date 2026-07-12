"""
app.py — HuggingFace Spaces entry point
Downloads model weights from HF Hub to cache (if not present locally),
then loads them into memory and launches the Gradio interface.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent
MIN_WEIGHT_BYTES = 1 * 1024 * 1024  # 1 MB

def _is_valid_weight(path: Path) -> bool:
    """Kiểm tra file tồn tại và không phải LFS pointer (> 1 MB)."""
    return path.exists() and path.stat().st_size > MIN_WEIGHT_BYTES

# ── Resolve Detector weights ───────────────────────────────────────────────────
local_det_path = BASE_DIR / "weights" / "detector" / "dect_v2.pt"
if _is_valid_weight(local_det_path):
    print(f"[*] Using local detector weights: {local_det_path}")
    det_weights_path = str(local_det_path)
else:
    print("[*] Downloading detector weights from Hugging Face Hub...")
    det_weights_path = hf_hub_download(
        repo_id="trieu123x/garbage-detect",
        filename="dect_v2.pt"
    )
    print(f"[*] Detector weights downloaded to cache: {det_weights_path}")

# ── Resolve Classifier weights ─────────────────────────────────────────────────
local_cls_path = BASE_DIR / "weights" / "classifier" / "best_resnet50_garbage.pth"
if _is_valid_weight(local_cls_path):
    print(f"[*] Using local classifier weights: {local_cls_path}")
    cls_weights_path = str(local_cls_path)
else:
    print("[*] Downloading classifier weights from Hugging Face Hub...")
    cls_weights_path = hf_hub_download(
        repo_id="trieu123x/garbage-cls",
        filename="best_resnet50_garbage.pth"
    )
    print(f"[*] Classifier weights downloaded to cache: {cls_weights_path}")

# ── Load models ────────────────────────────────────────────────────────────────
from detector import load_detector
from classifier import load_classifier

print("[*] Loading models into memory...")
load_detector(det_weights_path)
load_classifier(cls_weights_path)
print("[*] Models ready.")

# ── Launch Gradio ──────────────────────────────────────────────────────────────
from gradio_app import create_app

demo = create_app()
demo.launch()
