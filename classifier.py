"""
classifier.py — Quản lý ResNet50 Classification Model
Chịu trách nhiệm: load model, transform ảnh, classify ROI
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình đường dẫn & ngưỡng
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = r"C:\Users\admin\Downloads\code_ai\Phan_loai_rac(12cls)\checkpoints\best_resnet50_garbage.pth"

CLS_CONF_THRESH = 0.10   # Ngưỡng confidence tối thiểu của classifier

# Fallback class names nếu checkpoint không lưu sẵn
FALLBACK_CLASSES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

# ──────────────────────────────────────────────────────────────────────────────
# Singleton model
# ──────────────────────────────────────────────────────────────────────────────
_classifier_model = None
_class_names      = []
_transform        = None
_device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_transform():
    """Pipeline biến đổi ảnh đầu vào cho ResNet50."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _build_model(num_classes: int, device):
    """Khởi tạo kiến trúc ResNet50 với fc layer tuỳ chỉnh."""
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def load_classifier(weights_path=None, device=None):
    """
    Tải ResNet50 classifier (singleton).

    Args:
        weights_path: đường dẫn file .pth, dùng DEFAULT_WEIGHTS nếu None
        device:       torch.device, tự detect GPU/CPU nếu None
    Returns:
        (model, class_names, transform)
    """
    global _classifier_model, _class_names, _transform, _device

    if _classifier_model is None:
        if device:
            _device = device
        path = weights_path or DEFAULT_WEIGHTS
        print(f"[Classifier] Đang tải ResNet50 từ: {path}")

        state = torch.load(str(path), map_location=_device, weights_only=False)

        # Hỗ trợ 2 dạng checkpoint: dict đầy đủ hoặc state_dict thuần
        if "model_state_dict" in state:
            weights     = state["model_state_dict"]
            c_names     = state.get("class_names") or FALLBACK_CLASSES
            num_classes = state.get("num_classes", len(c_names))
        else:
            weights     = state
            c_names     = FALLBACK_CLASSES
            num_classes = len(c_names)

        # Xử lý prefix 'module.' từ DataParallel training
        weights = {k.replace("module.", ""): v for k, v in weights.items()}

        model = _build_model(num_classes, _device)
        model.load_state_dict(weights)
        model.eval()

        _classifier_model = model
        _class_names      = c_names
        _transform        = _build_transform()

        print(f"[Classifier] ✅ Đã tải xong. {num_classes} classes: {c_names}")

    return _classifier_model, _class_names, _transform


def get_classifier():
    """Trả về (model, class_names, transform) — lazy load nếu chưa có."""
    return load_classifier()


def classify(roi_bgr, conf_thresh: float = CLS_CONF_THRESH):
    """
    Phân loại 1 ROI (BGR numpy array).

    Args:
        roi_bgr:     vùng ảnh BGR đã crop + letterbox (từ detector.py)
        conf_thresh: ngưỡng confidence tối thiểu

    Returns:
        dict | None:
            - label:    str — tên lớp
            - cls_conf: float — confidence
            - class_idx: int — index lớp
        Trả về None nếu confidence < conf_thresh
    """
    import cv2
    model, c_names, tfm = get_classifier()

    # BGR → RGB → PIL
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)

    tensor = tfm(pil_img).unsqueeze(0).to(_device)

    with torch.no_grad():
        out  = model(tensor)
        prob = torch.nn.functional.softmax(out, dim=1)
        cls_conf, pred = torch.max(prob, 1)
        cls_conf  = float(cls_conf)
        class_idx = int(pred)

    if cls_conf < conf_thresh:
        return None

    label = c_names[class_idx] if c_names else str(class_idx)
    return {
        "label":     label,
        "cls_conf":  round(cls_conf, 3),
        "class_idx": class_idx,
    }


def get_class_names():
    """Trả về danh sách class names (lazy load)."""
    _, c_names, _ = get_classifier()
    return c_names


def reload_classifier(weights_path, device=None):
    """
    Tải lại classifier với weights mới (dùng khi đổi model).
    Args:
        weights_path: đường dẫn file .pth mới
        device: torch.device
    """
    global _classifier_model, _class_names, _transform
    _classifier_model = None   # reset singleton
    _class_names      = []
    _transform        = None
    return load_classifier(weights_path, device)
