import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import resnet50
from PIL import Image
from torchvision import transforms

try:
    from ultralytics import YOLO
except ImportError:
    print("Vui lòng cài đặt ultralytics và torchvision (nếu chưa có): pip install ultralytics torchvision")
    exit()

def get_classifier_transforms():
    """Các phép biến đổi ảnh để đưa vào model phân loại"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_classifier(model_path, device):
    """Tải model phân loại ResNet50"""
    print(f"Đang tải trọng số phân loại từ {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Checkpoint chứa dictionary với metadata
    if 'model_state_dict' in state_dict:
        model_weights = state_dict['model_state_dict']  
        class_names = state_dict.get('class_names', None)
        num_classes = state_dict.get('num_classes', 10)
    else:
        # Nếu checkpoint chỉ lưu state_dict (dự phòng)
        model_weights = state_dict
        class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
        num_classes = len(class_names)
        
    # Xử lý trường hợp mô hình được huấn luyện bằng nn.DataParallel (bị thêm tiền tố 'module.')
    model_weights = {k.replace('module.', ''): v for k, v in model_weights.items()}
        
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(model_weights)
    model = model.to(device)
    model.eval()
    
    return model, class_names

# ── Ngưỡng lọc vật thể ─────────────────────────────────────────────────────
# Diện tích bounding box tối thiểu (px²). Vật thể nhỏ hơn → quá xa, bỏ qua.
MIN_BOX_AREA    = 400    # ~20×20 px — giảm để không bỏ sót vật thể nhỏ/xa
# Laplacian variance tối thiểu. Nhỏ hơn → ROI quá mờ, bỏ qua.
# Video thực tế (ban đêm, ánh sáng yếu) thường có variance rất thấp!
# Đặt = 0 để tắt hoàn toàn bộ lọc blur; tăng dần nếu muốn lọc nhiễu hơn.
MIN_BLUR_SCORE  = 20.0    # 0 = tắt lọc blur; dùng 10~20 nếu muốn lọc nhẹ
# ────────────────────────────────────────────────────────────────────────────

# Bật/tắt debug log (hiển thị số box bị lọc mỗi giây)
DEBUG_FILTER = True

def is_blurry(roi_gray: "np.ndarray", threshold: float) -> bool:
    """Trả về True nếu ROI quá mờ (Laplacian variance < threshold)."""
    return cv2.Laplacian(roi_gray, cv2.CV_64F).var() < threshold

def main():
    # 1. Cấu hình thiết bị (Sử dụng GPU nếu có để mượt hơn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Pipeline đang chạy trên thiết bị: {device}")
    
    BASE_DIR = Path(__file__).resolve().parent
    yolo_weights = r"C:\Users\admin\Downloads\code_ai\Phan_loai_rac(12cls)\weights\detector\dect_v2.pt"
    classifier_weights = r"C:\Users\admin\Downloads\code_ai\Phan_loai_rac(12cls)\weights\classifier\best_resnet50_garbage.pth"
    
    # 2. Khởi tạo mô hình YOLO
    print("[*] Đang khởi tạo mô hình YOLO detect...")
    try:
        yolo_model = YOLO(yolo_weights)
        # Nếu dùng GPU, model tự động sử dụng thiết bị tương ứng
    except Exception as e:
        print(f"Lỗi khi tải YOLO: {e}")
        return

    # 3. Khởi tạo mô hình phân loại (10 lớp)
    try:
        classifier_model, class_names = load_classifier(classifier_weights, device)
        transform = get_classifier_transforms()
        print(f"[*] Đã tải thành công classifier với {len(class_names)} lớp: {class_names}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình phân loại: {e}")
        return
       
    # 4. Mở Video
    # Đổi đường dẫn nếu muốn chạy với video khác trong assets/videos/
    video_path = r"C:\Users\admin\Downloads\code_ai\Phan_loai_rac(12cls)\assets\istockphoto-1297634786-640_adpp_is.mp4"
    print(f"[*] Đang mở video test: {video_path}")
    print("[*] (Nhấn 'q' trên cửa sổ video để thoát)")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Lỗi: Không thể đọc file video {video_path}!")
        return

    # Lấy thông số FPS của video để tính thời gian chờ (delay) tua chậm 0.5 lần
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if not original_fps or original_fps == 0:
        original_fps = 30
    delay_ms = int(1000 / (original_fps * 0.5)) # Chạy ở tốc độ 0.5x

    # Biến debug để đếm box bị lọc
    import time
    debug_timer   = time.time()
    cnt_raw       = 0  # tổng box từ YOLO
    cnt_area_drop = 0  # bị lọc do diện tích
    cnt_blur_drop = 0  # bị lọc do mờ
    cnt_conf_drop = 0  # bị lọc do confidence classify
    cnt_shown     = 0  # box hiển thị được

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] Hết video hoặc không đọc được frame. Đang kết thúc...")
            break
            
        # Để tăng tốc, resize frame trước khi đưa vào pipeline
        frame = cv2.resize(frame, (640, 480))
        
        # Detect bằng YOLO để tìm vùng chứa đối tượng
        results = yolo_model(frame, stream=True, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cnt_raw += 1
                # Độ tin cậy của detect
                conf_detect = box.conf[0].item()
                
                if conf_detect < 0.1:
                    continue
                    
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w, _ = frame.shape
                
                # Mở rộng bounding box thêm 15%
                margin_x = int((x2 - x1) * 0.15)
                margin_y = int((y2 - y1) * 0.15)
                
                x1_crop = max(0, x1 - margin_x)
                y1_crop = max(0, y1 - margin_y)
                x2_crop = min(w, x2 + margin_x)
                y2_crop = min(h, y2 + margin_y)
                
                # Bỏ qua nếu box quá nhỏ (nhiễu pixel)
                if x2_crop - x1_crop < 10 or y2_crop - y1_crop < 10:
                    continue

                # ── Lọc 1: Vật thể quá xa (diện tích bbox gốc quá nhỏ) ──────
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < MIN_BOX_AREA:
                    cnt_area_drop += 1
                    continue  # bỏ qua: vật thể quá nhỏ / quá xa

                # Cắt (crop) vùng chứa đối tượng đã được mở rộng
                roi = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                # ── Lọc 2: ROI quá mờ (MIN_BLUR_SCORE=0 = tắt lọc) ──────────
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur_val = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                if MIN_BLUR_SCORE > 0 and blur_val < MIN_BLUR_SCORE:
                    cnt_blur_drop += 1
                    continue  # bỏ qua: ảnh mờ
                
                # Padding ảnh ROI thành hình vuông (Letterbox) để hàm CenterCrop(224) không bị cắt mất 2 đầu của các vật thể dài
                roi_h, roi_w = roi.shape[:2]
                max_dim = max(roi_h, roi_w)
                pad_top = (max_dim - roi_h) // 2
                pad_bottom = max_dim - roi_h - pad_top
                pad_left = (max_dim - roi_w) // 2
                pad_right = max_dim - roi_w - pad_left
                roi_padded = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
                
                # Đưa ảnh crop vào model phân loại: BGR -> RGB -> PIL
                roi_rgb = cv2.cvtColor(roi_padded, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_rgb)
                
                # Tiền xử lý (Transform)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                # Thực hiện phân loại với ResNet50
                with torch.no_grad():
                    outputs = classifier_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    conf_cls, preds = torch.max(probabilities, 1)
                    
                    class_idx = preds.item()
                    confidence = conf_cls.item() # Độ tin cậy của lớp phân loại
                    label_name = class_names[class_idx] if class_names else str(class_idx)
                
                # Bỏ qua không vẽ box nếu mô hình phân loại không thực sự chắc chắn
                if confidence < 0.10:
                    cnt_conf_drop += 1
                    continue
                cnt_shown += 1
                
                # Hiển thị lên màn hình
                color = (0, 255, 0) # Xanh lá cây
                
                # 1. Vẽ Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 2. Tên lớp + 3. Độ tin cậy
                text = f"{label_name.upper()} ({confidence:.2f})"
                
                # Vẽ nền cho chữ để dễ nhìn
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # ── Debug: in thống kê mỗi 2 giây ──────────────────────────────────
        if DEBUG_FILTER and (time.time() - debug_timer) >= 2.0:
            print(f"[DEBUG] raw={cnt_raw} | area_drop={cnt_area_drop} "
                  f"| blur_drop={cnt_blur_drop} | conf_drop={cnt_conf_drop} "
                  f"| shown={cnt_shown}")
            # Reset counters
            cnt_raw = cnt_area_drop = cnt_blur_drop = cnt_conf_drop = cnt_shown = 0
            debug_timer = time.time()

        # Show video
        cv2.imshow('Realtime Garbage Classification Pipeline', frame)
        
        # Thoát bằng phím 'q'
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
