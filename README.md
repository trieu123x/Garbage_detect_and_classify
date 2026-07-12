---
title: Garbage Detection and Classification
emoji: 🗑️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.0
python_version: "3.10"
app_file: gradio_app.py
pinned: false
---

# 🗑️ Real-Time Garbage Classification System (12 Classes)


> **A production-ready, end-to-end AI pipeline for automated waste detection and classification.**  
> Combining YOLOv8 object detection with fine-tuned ResNet50 classification, served via a Flask web application.

---

## 🎯 Project Overview

This project implements a **two-stage computer vision pipeline** capable of:
1. **Detecting** garbage objects in real-time from images, URLs, or video streams using **YOLOv8**
2. **Classifying** each detected object into one of **12 waste categories** using a fine-tuned **ResNet50**

The system is deployed as a **Flask REST API** with an interactive web UI, supporting both image upload and URL-based inference.

### Classified Waste Categories
`Battery` · `Biological` · `Cardboard` · `Clothes` · `Glass` · `Metal` · `Paper` · `Plastic` · `Shoes` · `Trash` · `E-Waste` · `Hazardous`

---

## 🏗️ System Architecture

```
Input (Image / Video / URL)
         │
         ▼
  ┌─────────────┐
  │  YOLOv8     │  ← Object Detection (custom-trained)
  │  Detector   │
  └──────┬──────┘
         │ Bounding Boxes + Letterbox Crop
         ▼
  ┌─────────────┐
  │  ResNet50   │  ← Fine-tuned Classifier (Transfer Learning)
  │  Classifier │     ImageNet pretrained → 12-class head
  └──────┬──────┘
         │ Class Label + Confidence Score
         ▼
  ┌─────────────┐
  │  Flask API  │  ← REST API + Annotated Image (base64)
  │  Web Server │
  └─────────────┘
```

---

## 🛠️ Tech Stack & Skills Demonstrated

### Machine Learning & Deep Learning
| Technology | Application |
|---|---|
| **PyTorch** | Model training, inference, custom loss functions |
| **Torchvision / ResNet50** | Transfer learning, feature extraction, fine-tuning |
| **Ultralytics YOLOv8** | Real-time object detection, custom dataset training |
| **CosineAnnealingLR** | Learning rate scheduling for optimal convergence |
| **nn.DataParallel** | Multi-GPU training support |
| **Softmax + Confidence Thresholding** | Probabilistic output filtering |

### Computer Vision
| Technology | Application |
|---|---|
| **OpenCV (cv2)** | Real-time video capture, frame processing, annotation |
| **PIL / Pillow** | Image preprocessing pipeline |
| **Letterbox Padding** | Aspect-ratio-preserving preprocessing |
| **Bounding Box Expansion** | Context-aware ROI cropping (+15% margin) |

### MLOps & Deployment
| Technology | Application |
|---|---|
| **Flask** | REST API server, model serving |
| **Kaggle Notebooks** | Cloud-based model training |
| **Checkpoint Management** | Best-model saving, state dict serialization |
| **CUDA / CPU fallback** | GPU-accelerated inference with graceful degradation |

### Data Engineering
| Skill | Application |
|---|---|
| **Custom Dataset Preparation** | Curated 12-class waste dataset (TACO-based) |
| **Data Augmentation** | Resize, CenterCrop, Normalize with ImageNet stats |
| **Class Imbalance Handling** | Balanced sampling across 12 categories |
| **Train/Val Split** | Proper evaluation methodology |

---

## 🚀 Key Features

- **⚡ Real-time Pipeline** – Processes live video at 0.5x speed for detailed analysis
- **🌐 Web API** – Upload images or provide URLs for instant inference
- **🖼️ Annotated Output** – Returns base64-encoded image with drawn bounding boxes
- **📊 Detection Statistics** – Per-class object count and confidence scores
- **🔧 Robust Model Loading** – Handles `nn.DataParallel` `module.` prefix stripping automatically
- **🛡️ Error Handling** – Graceful fallback for invalid URLs, corrupt images, and system errors

---

## 📁 Project Structure

```
Phan_loai_rac(12cls)/
├── app.py                    # Flask REST API server
├── realtime_pipeline.py      # Real-time video inference pipeline
├── train.py                  # Training loop (CosineAnnealing, best model tracking)
├── evaluate.py               # Model evaluation & metrics
├── data_loader.py            # Dataset loading & augmentation
├── prepare_data.py           # Dataset preparation utilities
├── main.py                   # Training entry point
├── fine-tune-on-taco.ipynb   # Kaggle training notebook
├── args.yaml                 # Hyperparameter configuration
├── weights/                  # YOLOv8 custom trained weights
├── checkpoints/              # ResNet50 checkpoints (best model)
└── templates/
    └── index.html            # Web UI
```

---

## ⚙️ Installation & Usage

### Prerequisites
```bash
pip install torch torchvision ultralytics flask opencv-python pillow numpy
```

### Run Web Application
```bash
python app.py
# → Navigate to http://localhost:5000
```

### Run Real-Time Video Pipeline
```bash
python realtime_pipeline.py
# → Opens video window, press 'q' to quit
```

### Train from Scratch
```bash
python main.py --config args.yaml
```

---

## 🧠 Model Details

### YOLOv8 Detector
- **Architecture**: YOLOv8 (custom-trained on waste dataset)
- **Task**: Object detection — localizes garbage in images
- **Confidence Threshold**: `0.15` (tuned to capture transparent/glass objects)
- **Input**: Any resolution → auto-resized to 640×480

### ResNet50 Classifier
- **Backbone**: ResNet50 pretrained on ImageNet
- **Head**: Custom `nn.Linear(2048 → 12)` classification layer
- **Training**: Fine-tuned with `CosineAnnealingLR`, `CrossEntropyLoss`
- **Optimizer**: Adam / SGD with momentum
- **Preprocessing**: Resize(256) → CenterCrop(224) → Normalize(ImageNet stats)
- **Inference**: `torch.no_grad()` + Softmax probability output

---

## 📈 Results

### 🎯 Model Performance

| Model | Metric | Value |
|---|---|---|
| **ResNet50 Classifier** | Validation Accuracy | **84%** |
| **YOLOv8 Detector** | mAP@50 | **80%** |
| **YOLOv8 Detector** | Confidence Threshold | 0.25 (API) · 0.15 (real-time) |
| **ResNet50 Classifier** | Confidence Filter | 0.60 (API) · 0.10 (real-time) |

> 🏋️ Trained on a curated 12-class waste dataset (TACO-based) via Kaggle GPU (P100).

### ⚙️ System Specs

| Metric | Value |
|---|---|
| **Classes** | 12 waste categories |
| **GPU Support** | ✅ CUDA (auto-fallback to CPU) |
| **Inference Mode** | Real-time video + REST API |

---

## 🔬 Technical Challenges Solved

1. **Multi-GPU Training Artifact** – Automatically strips `module.` prefix from `nn.DataParallel` state dicts for single-GPU/CPU inference compatibility
2. **Aspect Ratio Preservation** – Implemented letterbox padding to prevent shape distortion before `CenterCrop(224)`
3. **URL Image Fetching** – Added `User-Agent` spoofing to bypass `403 Forbidden` blocks from image CDNs
4. **Confidence Calibration** – Tuned separate thresholds for detection and classification stages independently
5. **Model Warm-Up** – Lazy model loading on first request via `@app.before_request` to avoid blocking server startup

---

## 👤 About Me

I am an **AI Engineer** with hands-on experience building end-to-end deep learning systems — from dataset preparation and model training to REST API deployment and real-time inference optimization.

### Core Competencies
- 🧠 **Deep Learning**: PyTorch, Transfer Learning, CNN Architectures (ResNet50, EfficientNet, YOLOv8), HuggingFace Transformers, Sentence Transformers
- 👁️ **Computer Vision**: Object Detection (YOLOv8 · mAP50 **0.80**), Image Classification (ResNet50 · Acc **0.84**), Real-time Video Processing (OpenCV)
- 📐 **Model Evaluation**: mAP, F1-Score, Precision/Recall, Confusion Matrix, Train/Val Accuracy
- 🔍 **NLP & Embeddings**: RAG pipelines, Prompt Engineering, `text-embedding-004` (Gemini), Vector Search — Cosine Similarity (pgvector / Supabase)
- 🤖 **LLM Integration**: Gemini API, GPT, AI chatbot with real-time Streaming (SSE)
- 🚀 **Model Deployment**: Flask, FastAPI, REST APIs, Docker (containerization)
- 🌐 **Full-Stack**: Python (Advanced), Node.js, ReactJS, RESTful API
- ☁️ **Cloud & MLOps**: Kaggle GPU, Google Colab, Linux, checkpoint management
- 🗄️ **Data & Vector DB**: PostgreSQL + pgvector, Supabase, Pandas, NumPy
- 📊 **Data Engineering**: Dataset curation, augmentation strategies, class imbalance handling

### Other Projects
- **AI Medical Chatbot** – RAG-powered doctor recommendation system using Gemini API + FastAPI
- **Hospital Appointment System** – Full-stack web app with JWT authentication and PostgreSQL backend
- **Distributed Database Design** – Horizontal fragmentation schema for multi-branch e-commerce

---

## 📫 Contact

> Feel free to reach out for collaboration or opportunities in **Computer Vision**, **MLOps**, or **AI Engineering**.

---

*Built with ❤️ using PyTorch, YOLOv8, and Flask*
