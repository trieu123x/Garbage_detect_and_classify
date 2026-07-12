"""
detector.py — Quản ly YOLO Detection Model
Workflow khop voi realtime_pipeline.py:
  1. YOLO detect (stream=True)
  2. Filter conf < 0.15
  3. Expand bbox 15%
  4. Filter crop size < 10
  5. Filter area < MIN_BOX_AREA
  6. Crop ROI
  7. Filter blur (Laplacian < MIN_BLUR_SCORE)
  8. Letterbox padding
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Hugging Face Hub — nguon weights chinh thuc
HF_REPO_ID       = "trieu123x/garbage-detect"
HF_FILENAME      = "dect_v2.pt"
# Fallback local (neu khong co mang / HF bi loi)
DEFAULT_WEIGHTS  = BASE_DIR / "garbage_detector_v2" / "weights" / "best.pt"

DET_CONF_THRESH = 0.15   # Nguong confidence YOLO (khop realtime_pipeline)
MIN_BOX_AREA    = 1500   # Dien tich bbox goc toi thieu (px^2)
MIN_BLUR_SCORE  = 60.0   # Laplacian variance toi thieu

# Kich thuoc inference YOLO mac dinh cho realtime (nho de nhanh)
# test_image.py se truyen infer_size=1280 hoac cao hon de chinh xac hon
INFER_W, INFER_H = 640, 480
DEFAULT_INFER_SIZE = 640   # imgsz mac dinh khi chay realtime

# ------------------------------------------------------------------------------
# Singleton model
# ------------------------------------------------------------------------------
_yolo_model = None


def load_detector(weights_path=None):
    """
    Tai YOLO model (singleton).
    Thu tu uu tien:
      1. weights_path truyen vao (neu co)
      2. HuggingFace Hub  -> trieu123x/garbage-detect / dect_v2.pt
      3. Local fallback   -> DEFAULT_WEIGHTS
    """
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    if weights_path:
        # Caller chi dinh ro duong dan
        resolved = str(weights_path)
        print(f"[Detector] Dang tai YOLO tu duong dan chi dinh: {resolved}")
    else:
        # Uu tien tai tu HF Hub
        print(f"[Detector] Dang tai weights '{HF_FILENAME}' tu HF Hub ({HF_REPO_ID})...")
        try:
            from huggingface_hub import hf_hub_download
            resolved = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
            )
            print(f"[Detector] Da tai thanh cong: {resolved}")
        except Exception as e:
            print(f"[!] Khong the tai tu HF Hub: {e}")
            resolved = str(DEFAULT_WEIGHTS)
            print(f"[Detector] Dung fallback local: {resolved}")

    _yolo_model = YOLO(resolved)
    print("[Detector] YOLO san sang.")
    return _yolo_model


def get_detector():
    return load_detector()


def is_blurry(roi_gray, threshold: float) -> bool:
    """True neu ROI qua mo (Laplacian variance < threshold)."""
    return cv2.Laplacian(roi_gray, cv2.CV_64F).var() < threshold


def detect(frame_bgr, conf_thresh: float = DET_CONF_THRESH,
           infer_size: int = DEFAULT_INFER_SIZE,
           min_box_area: int = MIN_BOX_AREA,
           min_blur_score: float = MIN_BLUR_SCORE):
    """
    Chay YOLO detect tren 1 frame.

    Args:
        frame_bgr:      anh BGR (numpy array), bat ky kich thuoc
        conf_thresh:    nguong confidence YOLO
        infer_size:     kich thuoc imgsz truyen vao YOLO (640 realtime, 1280+ anh tinh)
        min_box_area:   dien tich bbox toi thieu (px^2), 0 = tat loc
        min_blur_score: Laplacian variance toi thieu, 0 = tat loc blur

    Returns:
        List[dict] moi phan tu:
            - bbox_orig : [x1, y1, x2, y2]  toa do tren FRAME GOC
            - roi       : numpy BGR da letterbox padding (cho ResNet50)
            - det_conf  : float
    """
    model = get_detector()

    # Giu nguyen kich thuoc goc; YOLO tu resize noi bo theo imgsz
    # (tranh mat chi tiet khi ta resize tay xuong 640x480)
    orig_h, orig_w = frame_bgr.shape[:2]

    hits = []
    results = model(frame_bgr, imgsz=infer_size, verbose=False, stream=True)

    for result in results:
        for box in result.boxes:
            # Buoc 2: Filter conf
            det_conf = float(box.conf[0])
            if det_conf < conf_thresh:
                continue

            # Toa do bbox — YOLO tra ve tren khong gian anh GOC (vi ta truyen frame_bgr)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Buoc 3: Mo rong bbox 15% (tren khong gian anh goc)
            margin_x = int((x2 - x1) * 0.15)
            margin_y = int((y2 - y1) * 0.15)
            cx1 = max(0, x1 - margin_x)
            cy1 = max(0, y1 - margin_y)
            cx2 = min(orig_w, x2 + margin_x)
            cy2 = min(orig_h, y2 + margin_y)

            # Buoc 4: Filter crop qua nho (nhieu pixel)
            if (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
                continue

            # Buoc 5: Filter area qua nho (vat the qua xa) — bo qua neu min_box_area=0
            box_area = (x2 - x1) * (y2 - y1)
            if min_box_area > 0 and box_area < min_box_area:
                continue

            # Buoc 6: Crop ROI tu anh GOC (chat luong cao hon)
            roi = frame_bgr[cy1:cy2, cx1:cx2]

            # Buoc 7: Filter blur — bo qua neu min_blur_score=0
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if min_blur_score > 0 and is_blurry(gray, min_blur_score):
                continue

            # Buoc 8: Letterbox padding -> hinh vuong
            rh, rw = roi.shape[:2]
            md  = max(rh, rw)
            pt  = (md - rh) // 2;  pb = md - rh - pt
            pl  = (md - rw) // 2;  pr = md - rw - pl
            roi_padded = cv2.copyMakeBorder(
                roi, pt, pb, pl, pr,
                cv2.BORDER_CONSTANT, value=[114, 114, 114]
            )

            # Toa do bbox da la tren anh goc, khong can quy doi
            hits.append({
                "bbox_orig": [x1, y1, x2, y2],
                "roi":       roi_padded,
                "det_conf":  round(det_conf, 3),
            })

    return hits


def reload_detector(weights_path):
    """Tai lai detector voi weights moi."""
    global _yolo_model
    print(f"[Detector] Reload tu: {weights_path}")
    _yolo_model = YOLO(str(weights_path))
    print("[Detector] Reload xong.")
    return _yolo_model
