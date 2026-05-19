"""
app.py — Deepfake Detection Flask API (PyTorch backend)
"""

import os
import uuid
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam
from utils.face_detect import detect_and_crop_face

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "model", "deepfake_detector.pth")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE      = 224
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Flask running on device: {device}")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ── Model loader (lazy) ───────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run model/train.py first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    net = models.mobilenet_v2(weights=None)
    in_f = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device).eval()

    _model = net
    print("Model loaded.")
    return _model


# ── Image preprocessing ───────────────────────────────────────────────────────
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = _preprocess(img_rgb)          # (3, H, W)
    return tensor.unsqueeze(0)              # (1, 3, H, W)

def ndarray_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    if not f.filename or "." not in f.filename:
        return jsonify({"error": "Invalid file."}), 400

    ext = f.filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Use JPG / PNG / WEBP."}), 400

    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    f.save(path)

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return jsonify({"error": "Could not read image."}), 400

    original_bgr = img_bgr.copy()

    # Face detection
    face_crop, bbox = detect_and_crop_face(img_bgr, target_size=IMG_SIZE)
    face_detected   = face_crop is not None
    analysis_img    = face_crop if face_detected else cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # Inference
    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    tensor = preprocess(analysis_img)          # (1,3,H,W)

    with torch.no_grad():
        logit = model(tensor.to(device)).squeeze().item()

    score      = float(torch.sigmoid(torch.tensor(logit)))   # 0=real, 1=fake
    label      = "FAKE" if score >= 0.5 else "REAL"
    confidence = score if label == "FAKE" else 1.0 - score

    # Grad-CAM  (needs grad, so re-run with grad enabled)
    heatmap        = make_gradcam_heatmap(tensor, model, device)
    gradcam_overlay = overlay_gradcam(analysis_img.copy(), heatmap)

    # Annotate original
    annotated = original_bgr.copy()
    if face_detected and bbox:
        x, y, w, h = bbox
        color = (0, 0, 220) if label == "FAKE" else (0, 200, 50)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.putText(annotated, f"{label} {confidence:.1%}",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return jsonify({
        "label":         label,
        "confidence":    round(confidence * 100, 2),
        "face_detected": face_detected,
        "original_b64":  ndarray_to_b64(annotated),
        "gradcam_b64":   ndarray_to_b64(gradcam_overlay),
    })


@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)