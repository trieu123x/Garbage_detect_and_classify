"""
test_image.py — Thử nghiệm nhận diện và phân loại rác trên 1 bức ảnh cục bộ
Cách chạy:
    python test_image.py --image assets/img/image.png --output output_result.png
"""

import os
import argparse
import cv2
from pathlib import Path

# Thêm import các module model
from detector import detect, load_detector, DET_CONF_THRESH
from classifier import classify, load_classifier, CLS_CONF_THRESH

# Màu sắc đại diện cho các lớp waste
CLASS_COLORS = {
    "battery":    (0,   80, 255),  # Cam/đỏ
    "biological": (0,  200, 100),  # Xanh lá
    "cardboard":  (30, 144, 255),  # Xanh dương nhạt
    "clothes":    (147, 20, 255),  # Tím
    "glass":      (0,  255, 255),  # Vàng chanh
    "metal":      (180, 105, 255), # Hồng tím
    "paper":      (0,  165, 255),  # Cam sáng
    "plastic":    (0,  255, 127),  # Xanh ngọc
    "shoes":      (255, 160, 122), # Cam đất
    "trash":      (128, 128, 128), # Xám
    "default":    (0,  255,   0),  # Xanh lá mặc định
}

def main():
    parser = argparse.ArgumentParser(description="Test Garbage Detection & Classification pipeline on a single image.")
    parser.add_argument("--image", type=str, default="assets/img/image.png", help="Đường dẫn đến ảnh đầu vào.")
    parser.add_argument("--output", type=str, default="output_result.png", help="Đường dẫn lưu ảnh kết quả.")
    parser.add_argument("--det-thresh", type=float, default=DET_CONF_THRESH, help="Ngưỡng YOLO detector.")
    parser.add_argument("--cls-thresh", type=float, default=CLS_CONF_THRESH, help="Ngưỡng ResNet50 classifier.")
    parser.add_argument("--infer-size", type=int, default=1920,
                        help="Kích thước inference YOLO (imgsz). Mặc định 1920 để tối đa độ chính xác. "
                             "Dùng 1280 để cân bằng; 640 giống realtime.")
    args = parser.parse_args()

    # 1. Kiểm tra ảnh đầu vào
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[!] Lỗi: Không tìm thấy ảnh tại: {image_path}")
        print("Các ảnh mẫu có sẵn:")
        for img in Path("assets/img").glob("*.png"):
            print(f"  - {img}")
        return

    # 2. Khởi tạo & Load model
    print("[*] Đang tải các mô hình (YOLO + ResNet50)...")
    try:
        load_detector()
        load_classifier()
        print("[*] Tải mô hình thành công.")
    except Exception as e:
        print(f"[!] Lỗi khi tải mô hình: {e}")
        return

    # 3. Đọc ảnh
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[!] Lỗi: Không thể đọc ảnh {image_path}")
        return
    
    annotated = frame.copy()
    print(f"[*] Đọc ảnh: {image_path.name} | Kích thước: {frame.shape[1]}x{frame.shape[0]}")

    # 4. Chạy Detection
    print(f"[*] Đang chạy YOLO Detector (Ngưỡng: {args.det_thresh} | imgsz: {args.infer_size})...")
    hits = detect(frame, conf_thresh=args.det_thresh, infer_size=args.infer_size)
    print(f"[*] Tìm thấy {len(hits)} vùng đối tượng khả nghi.")

    detections = []
    
    # 5. Phân loại từng vùng (Crop & Classify)
    print(f"[*] Đang chạy ResNet50 Classifier (Ngưỡng tối thiểu: {max(args.cls_thresh, 0.10)})...")
    for idx, hit in enumerate(hits):
        x1, y1, x2, y2 = hit["bbox_orig"]
        roi_bgr = hit["roi"]
        det_conf = hit["det_conf"]

        # Phân loại vùng ROI
        effective_cls = max(args.cls_thresh, 0.10)
        result = classify(roi_bgr, conf_thresh=effective_cls)
        
        if result is None:
            # Bị lọc do không đủ độ tin cậy phân loại
            continue

        label = result["label"]
        cls_conf = result["cls_conf"]

        detections.append({
            "class": label,
            "cls_conf": cls_conf,
            "det_conf": det_conf,
            "bbox": [x1, y1, x2, y2],
        })

        # Vẽ Bounding Box và Nhãn lên ảnh kết quả
        color = CLASS_COLORS.get(label, CLASS_COLORS["default"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label.upper()} ({cls_conf:.2f})"
        
        # Tạo khung nền cho chữ dễ đọc
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 6. In bảng kết quả ra Terminal
    print("\n" + "="*60)
    print(f" KẾT QUẢ NHẬN DIỆN CHO {image_path.name.upper()}:")
    print("="*60)
    if not detections:
        print("  Không phát hiện/phân loại được rác thải nào.")
    else:
        print(f"  {'STT':<4} | {'Phân loại':<12} | {'Độ tin cậy':<10} | {'Tọa độ BBox':<18}")
        print("  " + "-"*54)
        for i, det in enumerate(detections, 1):
            bbox_str = f"[{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]}]"
            print(f"  {i:<4} | {det['class'].capitalize():<12} | {det['cls_conf']:.1%}      | {bbox_str:<18}")
    print("="*60 + "\n")

    # 7. Lưu ảnh kết quả
    output_path = Path(args.output)
    cv2.imwrite(str(output_path), annotated)
    print(f"[+] Đã vẽ kết quả và lưu ảnh tại: {output_path.resolve()}")

if __name__ == "__main__":
    main()
