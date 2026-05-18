<<<<<<< HEAD
---
title: Deepfake Image Detection
emoji: 🚀
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
# 🎭 DeepTrace — Deepfake Image Detector

A deep learning-based deepfake image detection system using **MobileNetV2 Transfer Learning** and **Grad-CAM explainability**, deployed as a **Flask web application**.

> Developed by **Vedanti Marne** | TY BTech. AI & DS | AISSMS IOIT, Pune | AY 2024-2025

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training the Model](#training-the-model)
- [Running the Web App](#running-the-web-app)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)

---

## 📖 Overview

This project detects whether a facial image is **real** or **AI-generated (deepfake)** using:

- **MobileNetV2** — a lightweight CNN pre-trained on ImageNet, fine-tuned for binary deepfake classification
- **Grad-CAM** — generates a heatmap highlighting suspicious facial regions (eyes, mouth, jaw boundaries) that influenced the model's prediction
- **OpenCV** — automatically detects and crops the face region before inference
- **Flask** — serves the model as a web application for easy image upload and result display

---

## 🏗️ Architecture

```
Image Upload
     ↓
Face Detection (OpenCV Haar Cascade)
     ↓
Preprocessing (Resize 224×224, Normalize)
     ↓
CNN Model (MobileNetV2 — PyTorch)
     ↓
Prediction (REAL / FAKE) + Confidence Score
     ↓
Grad-CAM Heatmap (highlights suspicious regions)
     ↓
Result Display (Flask Web UI)
```

---

## 🖥️ Demo

After running the app, open **http://127.0.0.1:5000** in your browser:

1. Drag and drop or browse a face image (JPG / PNG / WEBP)
2. Click **Analyze Image**
3. View the prediction verdict, confidence score, and Grad-CAM heatmap

---

## 📁 Project Structure

```
deepfake-detector/
├── app.py                        # Flask backend — inference + API
├── model/
│   ├── train.py                  # Model training script (PyTorch)
│   └── deepfake_detector.pth     # Saved model weights (after training)
├── utils/
│   ├── gradcam.py                # Grad-CAM heatmap generation
│   └── face_detect.py            # OpenCV face detection & cropping
├── static/
│   └── uploads/                  # Temporary uploaded image storage
├── templates/
│   └── index.html                # Frontend web UI
├── requirements.txt              # Python dependencies
└── README.md
```

> **Note:** `deepfake_detector.pth` and the dataset are not included in this repository.
> Run `model/train.py` after downloading the dataset to generate the `.pth` file.

---

## ⚙️ Requirements

### Hardware
| Component | Minimum | Used in This Project |
|-----------|---------|---------------------|
| GPU | Optional (CPU works) | NVIDIA RTX 4050 (6 GB VRAM) |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | ~7.5 GB total (dataset + env) |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 |

### Software
| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| Torchvision | 0.21.0+cu124 |
| OpenCV | 4.10.0 |
| Flask | 3.0.3 |
| NumPy | 1.26.4 |
| Pillow | 10.3.0 |

---

## 🚀 Installation

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detector.git
cd deepfake-detector
```

### Step 2 — Create a Virtual Environment
```bash
python -m venv venv

# Activate
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install PyTorch (GPU version)
```bash
# CUDA 12.4 wheels — compatible with CUDA 13.x drivers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> For CPU only:
> ```bash
> pip install torch torchvision torchaudio
> ```

### Step 4 — Verify GPU Detection
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 4050 Laptop GPU
```

### Step 5 — Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

Download the dataset from Kaggle:

🔗 **[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)**

After downloading, organize it inside the project folder:

```
deepfake-detector/
└── real_vs_fake/
    └── real-vs-fake/
        ├── train/
        │   ├── real/     ← ~50,000 real face images
        │   └── fake/     ← ~50,000 fake face images
        ├── valid/
        │   ├── real/
        │   └── fake/
        └── test/
            ├── real/
            └── fake/
```

Then update `DATASET_DIR` in `model/train.py` to match your path:
```python
DATASET_DIR = "real_vs_fake/real-vs-fake"
```

---

## 🏋️ Training the Model

```bash
python model/train.py
```

### Training Configuration (in `train.py`)
```python
BATCH_SIZE  = 64     # increase to 128 for faster training if VRAM allows
EPOCHS_HEAD = 10     # Phase 1: head-only training
EPOCHS_FINE = 10     # Phase 2: fine-tuning last 3 blocks
NUM_WORKERS = 2      # set to 0 if Windows multiprocessing errors occur
```

### Training Phases
| Phase | What Trains | Learning Rate | Epochs |
|-------|------------|---------------|--------|
| Phase 1 | Classification head only (base frozen) | 1e-3 | 10 |
| Phase 2 | Head + last 3 MobileNetV2 blocks | 1e-5 | 10 |

### Expected Training Time
| Hardware | Time per Epoch | Total (~20 epochs) |
|----------|---------------|-------------------|
| RTX 4050 (BATCH=64) | ~5 min | ~1.5–2 hours |
| CPU only | ~50–80 min | ~20+ hours |

### Output
```
Model saved -> model/deepfake_detector.pth
```

> **Windows users:** If DataLoader crashes, set `NUM_WORKERS = 0` in `train.py`

---

## 🌐 Running the Web App

Make sure `deepfake_detector.pth` exists in the `model/` folder, then:

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 🔍 How It Works

### 1. Face Detection
OpenCV's Haar Cascade detects the largest face in the uploaded image and crops it with 20% padding. If no face is detected, the full image is analyzed.

### 2. Inference
The cropped face is resized to `224×224`, normalized with ImageNet statistics, and passed through the trained MobileNetV2 model. The output logit is converted to a probability via sigmoid:
- **Score ≥ 0.5 → FAKE**
- **Score < 0.5 → REAL**

### 3. Grad-CAM
PyTorch backward hooks capture gradients from the last Conv2D layer. These gradients are averaged (Global Average Pooling) to produce channel importance weights. The weighted feature maps form a heatmap — overlaid in JET colormap — showing which facial regions influenced the prediction most.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~91–93% |
| Test Accuracy | ~90–92% |
| Inference Time (GPU) | ~50–100ms |
| Inference Time (CPU) | ~500–800ms |
| Model Size (.pth) | ~12 MB |
| Training Dataset | 100,000 images |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.6, MobileNetV2 |
| Explainability | Grad-CAM (backward hooks) |
| Face Detection | OpenCV Haar Cascade |
| Web Backend | Flask 3.0, Werkzeug |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| GPU Compute | NVIDIA CUDA 12.4 |
| Image Processing | OpenCV, Pillow, NumPy |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `RuntimeError: No CUDA GPUs available` | Reinstall PyTorch with `--index-url https://download.pytorch.org/whl/cu124` |
| `DataLoader` crashes on Windows | Set `NUM_WORKERS = 0` in `train.py` |
| `TemplateNotFound: index.html` | Run `app.py` from the project root directory, not a subfolder |
| `FileNotFoundError: model not found` | Run `model/train.py` first to generate `deepfake_detector.pth` |
| GPU Util stuck at 0% during training | Set `NUM_WORKERS = 2` and ensure `if __name__ == '__main__':` guard is present |
| Laptop overheating | Normal — place on hard flat surface, keep vents clear |

---

## 📄 License

This project is for academic purposes — TY BTech Mini Project, AISSMS IOIT, Pune.

---

## 🙏 Acknowledgements

- **Dr. D. S. Zingade** — Project Guide, AI & DS Department, AISSMS IOIT
- **Dr. R. A. Jamadar** — Head of Department, AI & DS, AISSMS IOIT
- **Kaggle** — 140k Real and Fake Faces dataset by xhlulu
- **PyTorch Team** — MobileNetV2 pretrained weights
>>>>>>> fb0ddf6 (readme added)
