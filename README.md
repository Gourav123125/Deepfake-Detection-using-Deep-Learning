# DeepFake Detection System
### SVSU B.Tech CSE (AI/ML) Capstone Project
**Team:** Gourav (23UGBC35121) | Divyansh Kumar (23UGBC35116) | Arpit Verma (23UGBC35113)
**Supervisor:** Ms. Manisha Mudgal

---

## Architecture

```
Input Image/Video
       │
       ▼
  Face Detection (MTCNN / OpenCV)
       │
  ┌────┴──────────────────────┐
  │  Stream 1+2               │  Stream 3
  │  EfficientNet-B4 Backbone │  Frequency Analysis Head
  │  + CBAM Attention         │  (Laplacian high-freq)
  │  → (B, 1792)              │  → (B, 128)
  └────┬──────────────────────┘
       │      Concat → (B, 1920)
       │
  [Video only: LSTM (1920→512)]
       │
  Classifier: 1920/512 → 512 → 128 → 2
       │
  Softmax → [P(Real), P(Fake)]
       │
  Grad-CAM Heatmap
```

---

## Installation

```bash
git clone https://github.com/your-repo/deepfake-detector
cd deepfake-detector
pip install -r requirements.txt
```

---

## Dataset Structure

```
dataset/
  real/
    face_001.jpg
    face_002.jpg
    ...
  fake/
    face_001.jpg
    face_002.jpg
    ...
```

Compatible with:
- **FaceForensics++** (extract frames first with `ffmpeg`)
- **DFDC** (DeepFake Detection Challenge)
- **Celeb-DF v2**

---

## Training

```bash
# Train on image dataset (recommended to start)
python train.py \
  --data-root /path/to/dataset \
  --epochs 30 \
  --batch 16 \
  --lr 3e-4 \
  --save-dir checkpoints

# Train with video sequences (LSTM temporal modeling)
python train.py \
  --data-root /path/to/video_frames \
  --mode video \
  --epochs 30 \
  --batch 8 \
  --num-frames 16

# Resume from checkpoint
python train.py \
  --data-root /path/to/dataset \
  --resume checkpoints/deepfake_epoch010.pth \
  --epochs 50

# Quick experiment (limit dataset size)
python train.py \
  --data-root /path/to/dataset \
  --max-per-class 500 \
  --epochs 5
```

---

## Inference

```bash
# ── Image ──────────────────────────────────────────────────────

# Basic prediction
python inference.py face.jpg --ckpt checkpoints/deepfake_best.pth

# With Grad-CAM heatmap saved
python inference.py face.jpg \
  --ckpt checkpoints/deepfake_best.pth \
  --heatmap output/heatmap.png

# Save results as JSON
python inference.py face.jpg \
  --ckpt checkpoints/deepfake_best.pth \
  --json results/result.json

# ── Video ──────────────────────────────────────────────────────

# Basic video prediction
python inference.py video.mp4 --ckpt checkpoints/deepfake_best.pth

# Full video with annotated output
python inference.py video.mp4 \
  --ckpt checkpoints/deepfake_best.pth \
  --output output/annotated.mp4 \
  --sample-rate 5

# Analyze every 3rd frame (faster)
python inference.py video.mp4 \
  --ckpt checkpoints/deepfake_best.pth \
  --sample-rate 3 \
  --output annotated.mp4 \
  --json result.json

# ── Use CPU explicitly ─────────────────────────────────────────
python inference.py input.jpg --ckpt best.pth --device cpu

# ── Use MTCNN for better face detection ───────────────────────
python inference.py input.jpg --ckpt best.pth --face mtcnn
```

---

## Web Application

```bash
# Start Flask server (GPU)
python app.py --ckpt checkpoints/deepfake_best.pth --port 5000

# CPU only
python app.py --ckpt checkpoints/deepfake_best.pth --device cpu

# Development mode
python app.py --ckpt checkpoints/deepfake_best.pth --debug

# Open browser at: http://localhost:5000
```

---

## Project Structure

```
deepfake_system/
├── model/
│   ├── __init__.py
│   └── architecture.py      # CNN-LSTM + CBAM + FreqHead + Grad-CAM
├── utils/
│   ├── __init__.py
│   ├── dataset.py           # Image and video dataset classes
│   ├── transforms.py        # Train/val preprocessing pipelines
│   ├── face_detector.py     # MTCNN and OpenCV face detection
│   └── metrics.py           # MetricTracker + Grad-CAM visualization
├── templates/
│   └── index.html           # Flask UI template
├── train.py                 # Training pipeline
├── inference.py             # Inference engine (image + video)
├── app.py                   # Flask web application
├── requirements.txt
└── README.md
```

---

## Target Performance

| Dataset           | Accuracy | AUC-ROC |
|-------------------|----------|---------|
| FaceForensics++ (c23) | >90%  | >0.95   |
| Celeb-DF v2       | >85%     | >0.90   |

---

## Sample Output

```
══════════════════════════════════════════════════════
  VERDICT:        FAKE
  Fake Prob:      94.3%
  Real Prob:       5.7%
  Confidence:     94.3%
  Inference:       812ms
══════════════════════════════════════════════════════
  [Grad-CAM] Saved → output/heatmap.png
```
