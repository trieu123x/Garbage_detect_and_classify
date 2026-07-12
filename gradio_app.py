"""
gradio_app.py — Giao diện Gradio
Pipeline: detector.py (YOLO) -> ResNet50 (Classify)
Ho tro: Tai anh | Tai video | Webcam
"""

import os
import cv2
import torch
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image

# Import 2 module model
from detector   import detect,   load_detector,   DET_CONF_THRESH
from classifier import classify, load_classifier, CLS_CONF_THRESH, get_class_names

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
CLASS_COLORS = {
    "battery":    (0,   80, 255),
    "biological": (0,  200, 100),
    "cardboard":  (30, 144, 255),
    "clothes":    (147, 20, 255),
    "glass":      (0,  255, 255),
    "metal":      (180, 105, 255),
    "paper":      (0,  165, 255),
    "plastic":    (0,  255, 127),
    "shoes":      (255, 160, 122),
    "trash":      (128, 128, 128),
    "default":    (0,  255,   0),
}

BASE_DIR   = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "assets" / "videos"
IMGS_DIR   = BASE_DIR / "assets" / "img"

SAMPLE_VIDEOS = [
    str(VIDEOS_DIR / "istockphoto-1297634786-640_adpp_is.mp4"),
    str(VIDEOS_DIR / "istockphoto-1346313542-640_adpp_is.mp4"),
    str(VIDEOS_DIR / "istockphoto-1205810942-640_adpp_is.mp4"),
    str(VIDEOS_DIR / "istockphoto-638542230-640_adpp_is.mp4"),
]

SAMPLE_IMAGES = [
    str(IMGS_DIR / "image.png"),
    str(IMGS_DIR / "image copy.png"),
    str(IMGS_DIR / "image copy 2.png"),
    str(IMGS_DIR / "image copy 3.png"),
]


# ------------------------------------------------------------------------------
# Pipeline: detector -> classifier -> annotate
# ------------------------------------------------------------------------------
def process_frame(frame_bgr, det_thresh=DET_CONF_THRESH, cls_thresh=CLS_CONF_THRESH,
                  infer_size=640, min_blur_score=0.0, min_box_area=400):
    """
    Chay full pipeline tren 1 frame BGR.
    infer_size:     imgsz truyen vao YOLO
    min_blur_score: Laplacian threshold (0 = tat loc blur)
    min_box_area:   dien tich bbox toi thieu px^2 (0 = tat loc)
    Returns: (annotated_bgr, list[dict])
    """
    annotated  = frame_bgr.copy()
    detections = []

    hits = detect(frame_bgr, conf_thresh=det_thresh, infer_size=infer_size,
                  min_box_area=int(min_box_area), min_blur_score=float(min_blur_score))

    for hit in hits:
        x1, y1, x2, y2 = hit["bbox_orig"]
        roi_bgr  = hit["roi"]
        det_conf = hit["det_conf"]

        effective_cls = max(cls_thresh, 0.10)
        result = classify(roi_bgr, conf_thresh=effective_cls)
        if result is None:
            continue

        label    = result["label"]
        cls_conf = result["cls_conf"]

        detections.append({
            "class":    label,
            "cls_conf": cls_conf,
            "det_conf": det_conf,
            "bbox":     [x1, y1, x2, y2],
        })

        color = CLASS_COLORS.get(label, CLASS_COLORS["default"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label.upper()} ({cls_conf:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated, detections


def build_stats_md(detections):
    """Tao Markdown bang thong ke."""
    if not detections:
        return "### Khong phat hien vat the nao\n_Hay thu anh/video khac hoac dieu chinh nguong._"

    stats: dict[str, list] = {}
    for d in detections:
        stats.setdefault(d["class"], []).append(d["cls_conf"])

    lines = [f"### Phat hien **{len(detections)}** vat the\n"]
    lines.append("| Loai rac | So luong | Conf TB |")
    lines.append("|---|:---:|:---:|")
    for cls, confs in sorted(stats.items(), key=lambda x: -len(x[1])):
        avg = sum(confs) / len(confs)
        lines.append(f"| **{cls.capitalize()}** | {len(confs)} | {avg:.0%} |")

    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Gradio Handlers
# ------------------------------------------------------------------------------
def predict_image(pil_image, det_thresh, cls_thresh, infer_size=1280,
                  min_blur_score=0.0, min_box_area=400):
    if pil_image is None:
        return None, "Vui long tai anh len."
    frame_bgr     = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    ann_bgr, dets = process_frame(frame_bgr, det_thresh, cls_thresh,
                                  infer_size=int(infer_size),
                                  min_blur_score=float(min_blur_score),
                                  min_box_area=int(min_box_area))
    ann_rgb       = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(ann_rgb), build_stats_md(dets)


def predict_image_example(img_path):
    """Wrapper cho gr.Examples — dung nguong mac dinh."""
    if not img_path or not os.path.exists(str(img_path)):
        return None, "Anh mau khong tim thay."
    pil_image = Image.open(str(img_path)).convert("RGB")
    return predict_image(pil_image, DET_CONF_THRESH, CLS_CONF_THRESH, infer_size=1280)


def predict_video(video_path, det_thresh, cls_thresh, infer_size=640,
                  min_blur_score=0.0, min_box_area=400):
    if video_path is None:
        return None, "Vui long tai video len."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Khong the mo video."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SKIP = 2

    out_path = str(BASE_DIR / "assets" / "output_annotated.mp4")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_dets = []
    n_frames = 0
    last_ann = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if n_frames % SKIP == 0:
            ann, dets = process_frame(frame, det_thresh, cls_thresh,
                                      infer_size=int(infer_size),
                                      min_blur_score=float(min_blur_score),
                                      min_box_area=int(min_box_area))
            all_dets.extend(dets)
            last_ann = ann

        writer.write(last_ann if last_ann is not None else frame)
        n_frames += 1

    cap.release()
    writer.release()

    summary  = build_stats_md(all_dets)
    summary += f"\n\n> Da xu ly **{n_frames}** frames"
    return out_path, summary


def predict_webcam(frame_rgb, det_thresh, cls_thresh, infer_size=640,
                   min_blur_score=0.0, min_box_area=400):
    if frame_rgb is None:
        return None, "Dang cho webcam..."
    frame_bgr     = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
    ann_bgr, dets = process_frame(frame_bgr, det_thresh, cls_thresh,
                                  infer_size=int(infer_size),
                                  min_blur_score=float(min_blur_score),
                                  min_box_area=int(min_box_area))
    ann_rgb       = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(ann_rgb), build_stats_md(dets)


# ------------------------------------------------------------------------------
# CSS & HTML
# ------------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Dark-mode base ── */
body, .gradio-container {
    background: #0d1117 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #c9d1d9 !important;
}

/* Header */
#app-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 28px 40px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
#app-header h1 {
    margin: 0 0 6px;
    font-size: 1.65rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.3px;
}
#app-header p {
    margin: 0;
    font-size: 0.88rem;
    color: #8b949e;
    font-weight: 400;
}

/* Status bar */
#status-bar {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 0.875rem;
    color: #c9d1d9 !important;
}

/* Tabs */
.tab-nav button {
    background: transparent !important;
    color: #8b949e !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 10px 18px !important;
    transition: all 0.15s !important;
}
.tab-nav button.selected {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    font-weight: 600 !important;
}
.tab-nav button:hover { color: #e6edf3 !important; }

/* Panels / blocks */
.gradio-block, .block, .panel {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

/* Labels & text inside blocks */
.block label, .block .label-wrap span,
.svelte-1gfkn6j, .output-class {
    color: #8b949e !important;
}

/* Input fields */
textarea, input[type=text], input[type=number] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
}

/* Buttons */
button.primary {
    background: #1f6feb !important;
    border: none !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 8px 20px !important;
    box-shadow: 0 0 0 1px rgba(31,111,235,0.4) !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}
button.primary:hover {
    background: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(31,111,235,0.3) !important;
}

button.secondary {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}
button.secondary:hover {
    background: #30363d !important;
    border-color: #8b949e !important;
}

/* Sliders */
input[type=range] { accent-color: #58a6ff; }

/* Stats / markdown panel */
#stats-panel {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 16px !important;
    font-size: 0.875rem !important;
    color: #c9d1d9 !important;
}
#stats-panel table { width: 100%; border-collapse: collapse; }
#stats-panel th {
    color: #8b949e;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 6px 12px;
    border-bottom: 1px solid #30363d;
    text-align: left;
}
#stats-panel td {
    padding: 7px 12px;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
}

/* Markdown headings */
.prose h1, .prose h2, .prose h3, .prose h4 {
    color: #e6edf3 !important;
}
.prose p, .prose li { color: #c9d1d9 !important; }

/* Examples gallery */
.examples {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
.examples .label { color: #8b949e !important; }

/* Footer */
#footer-note {
    color: #8b949e;
    text-align: center;
    font-size: 0.78rem;
    margin-top: 20px;
    padding: 12px;
    border-top: 1px solid #30363d;
}
"""

HEADER_HTML = """
<div id="app-header">
  <h1>Garbage Detection &amp; Classification</h1>
  <p>Pipeline: YOLOv8 Detector &rarr; ResNet-50 Classifier &nbsp;&middot;&nbsp; 10 waste categories</p>
</div>
"""

FOOTER_HTML = """
<div id="footer-note">
  YOLOv8 + ResNet-50 &nbsp;&middot;&nbsp; Gradio &nbsp;&middot;&nbsp;
  Classes: battery &middot; biological &middot; cardboard &middot; clothes &middot; glass
  &middot; metal &middot; paper &middot; plastic &middot; shoes &middot; trash
</div>
"""


def make_sliders(default_infer_size=640):
    """4 slider dieu chinh chinh sach: det, cls, infer_size, blur_score + box_area."""
    det = gr.Slider(0.05, 0.95, value=DET_CONF_THRESH, step=0.05,
                    label="Detection Confidence (YOLO)",
                    info="Minimum YOLO confidence to keep a bounding box")
    cls = gr.Slider(0.05, 0.95, value=CLS_CONF_THRESH, step=0.05,
                    label="Classification Confidence (ResNet-50)",
                    info="Minimum ResNet-50 confidence to confirm a class")
    infer = gr.Slider(320, 1920, value=default_infer_size, step=32,
                      label="Inference Size (imgsz)",
                      info="Higher = more accurate but slower. Image: 1280, Video/Webcam: 640")
    blur = gr.Slider(0, 100, value=0, step=1,
                     label="Blur Filter Score (Laplacian)",
                     info="Minimum sharpness score. 0 = disabled. Increase to skip blurry ROIs.")
    box = gr.Slider(0, 5000, value=400, step=50,
                    label="Min Box Area (px²)",
                    info="Skip detections smaller than this area. 0 = disabled.")
    return det, cls, infer, blur, box


# ------------------------------------------------------------------------------
# Build Gradio App
# ------------------------------------------------------------------------------
def create_app():
    dark_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0d1117",
        body_background_fill_dark="#0d1117",
        body_text_color="#c9d1d9",
        body_text_color_dark="#c9d1d9",
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        block_border_color="#30363d",
        block_border_color_dark="#30363d",
        block_label_text_color="#8b949e",
        block_label_text_color_dark="#8b949e",
        input_background_fill="#0d1117",
        input_background_fill_dark="#0d1117",
        input_border_color="#30363d",
        input_border_color_dark="#30363d",
        button_primary_background_fill="#1f6feb",
        button_primary_background_fill_dark="#1f6feb",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#21262d",
        button_secondary_background_fill_dark="#21262d",
        button_secondary_text_color="#c9d1d9",
        button_secondary_border_color="#30363d",
    )
    with gr.Blocks(title="Garbage Detection & Classification", theme=dark_theme, css=CSS) as demo:

        gr.HTML(HEADER_HTML)

        # Status bar
        device_str  = "GPU" if torch.cuda.is_available() else "CPU"
        n_cls       = len(get_class_names())
        status_init = (
            f"Model ready &nbsp;&middot;&nbsp; Device: **{device_str}** "
            f"&nbsp;&middot;&nbsp; Classes: **{n_cls}** "
            f"&nbsp;&middot;&nbsp; YOLO loaded &nbsp;&middot;&nbsp; ResNet-50 loaded"
        )
        with gr.Row():
            model_status = gr.Markdown(status_init, elem_id="status-bar")

        def reload_models_ui():
            try:
                load_detector()
                load_classifier()
                return status_init
            except Exception as e:
                return f"Error: {e}"

        gr.Button("Reload Models", variant="secondary", size="sm") \
          .click(fn=reload_models_ui, outputs=model_status)

        # Tabs
        with gr.Tabs():

            # Tab 1: Image
            with gr.TabItem("Image"):
                gr.Markdown("### Image Inference\nUpload an image to detect and classify waste objects.")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(type="pil", label="Input Image",
                                             sources=["upload", "clipboard"], height=420)
                        with gr.Row():
                            img_det, img_cls, img_infer, img_blur, img_box = make_sliders(default_infer_size=1280)
                        img_btn = gr.Button("Run Detection", variant="primary")

                    with gr.Column(scale=1):
                        img_out   = gr.Image(type="pil", label="Annotated Output",
                                             height=420, interactive=False)
                        img_stats = gr.Markdown(value="Results will appear here after running detection.",
                                                elem_id="stats-panel")

                img_btn.click(fn=predict_image,
                              inputs=[img_input, img_det, img_cls, img_infer, img_blur, img_box],
                              outputs=[img_out, img_stats])
                img_input.upload(fn=predict_image,
                                 inputs=[img_input, img_det, img_cls, img_infer, img_blur, img_box],
                                 outputs=[img_out, img_stats])

                gr.Markdown("#### Sample Images\nClick a sample to load and run inference automatically.")
                gr.Examples(
                    examples=[[path, DET_CONF_THRESH, CLS_CONF_THRESH] for path in SAMPLE_IMAGES],
                    inputs=[img_input, img_det, img_cls],
                    outputs=[img_out, img_stats],
                    fn=predict_image_example,
                    label="Available samples",
                    examples_per_page=4,
                    cache_examples=False,
                )

            # Tab 2: Video
            with gr.TabItem("Video"):
                gr.Markdown("### Video Inference\nUpload a video to process all frames through the pipeline.")
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.Video(label="Input Video",
                                             sources=["upload"], height=380)
                        with gr.Row():
                            vid_det, vid_cls, vid_infer, vid_blur, vid_box = make_sliders(default_infer_size=640)
                        vid_btn = gr.Button("Process Video", variant="primary")

                    with gr.Column(scale=1):
                        vid_out   = gr.Video(label="Annotated Output",
                                             height=380, interactive=False)
                        vid_stats = gr.Markdown(value="Results will appear here after processing.",
                                                elem_id="stats-panel")

                vid_btn.click(fn=predict_video,
                              inputs=[vid_input, vid_det, vid_cls, vid_infer, vid_blur, vid_box],
                              outputs=[vid_out, vid_stats])

                gr.Markdown("#### Sample Videos\nClick a sample to load it into the input player.")
                gr.Examples(examples=[[p] for p in SAMPLE_VIDEOS],
                             inputs=[vid_input], label="Available samples",
                             examples_per_page=4)

            # Tab 3: Webcam
            with gr.TabItem("Webcam"):
                gr.Markdown(
                    "### Real-time Webcam Inference\n"
                    "Click **Start** to begin streaming, **Stop** to end the session."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        cam_input = gr.Image(label="Webcam Feed", sources=["webcam"],
                                             height=420)
                        with gr.Row():
                            cam_det, cam_cls, cam_infer, cam_blur, cam_box = make_sliders(default_infer_size=640)

                    with gr.Column(scale=1):
                        cam_out   = gr.Image(label="Annotated Output",
                                             height=420, interactive=False)
                        cam_stats = gr.Markdown(value="Waiting for webcam stream...",
                                                elem_id="stats-panel")

                cam_input.stream(fn=predict_webcam,
                                 inputs=[cam_input, cam_det, cam_cls, cam_infer, cam_blur, cam_box],
                                 outputs=[cam_out, cam_stats])

        gr.HTML(FOOTER_HTML)

    return demo


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[*] Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("[*] Loading models...")
    load_detector()
    load_classifier()
    print("[*] Models ready. Starting Gradio...")

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860,
               share=False, inbrowser=True)
