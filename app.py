"""
app.py
======
Flask web application for DeepFake Detection.

Features:
- Drag-and-drop upload for image and video files
- Real/Fake verdict with confidence percentage
- Grad-CAM heatmap visualization
- Processing time display
- Auto-delete uploaded files after processing
- Dark-themed responsive UI

Run:
    python app.py
    python app.py --ckpt checkpoints/deepfake_best.pth --port 5000
"""

import os
import uuid
import time
import argparse
import threading
from pathlib import Path

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, url_for,
)
from werkzeug.utils import secure_filename

from inference import DeepFakeInference


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = "uploads"
RESULTS_FOLDER  = "results"
MAX_MB          = 100
ALLOWED_EXTS    = {".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".webp"}
CLEANUP_DELAY   = 300    # Delete files after 5 minutes

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# Global inference engine (loaded once at startup)
_engine: DeepFakeInference = None


def get_engine() -> DeepFakeInference:
    global _engine
    if _engine is None:
        raise RuntimeError("Model not loaded. Start app with --ckpt flag.")
    return _engine


def _delayed_delete(*paths, delay: int = CLEANUP_DELAY):
    """Delete files after a delay (runs in background thread)."""
    def _delete():
        time.sleep(delay)
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    threading.Thread(target=_delete, daemon=True).start()


# ─────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Accepts: multipart/form-data with 'file' field
    Returns: JSON with prediction result
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Validate extension
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. "
                     f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
        }), 400

    # Save upload with unique name
    uid  = str(uuid.uuid4())[:8]
    safe = secure_filename(f.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{safe}")
    f.save(upload_path)

    heatmap_path     = None
    heatmap_url      = None
    annotated_path   = None
    annotated_url    = None
    is_video = ext in {".mp4", ".avi", ".mov"}

    try:
        engine = get_engine()

        if is_video:
            annotated_path = os.path.join(RESULTS_FOLDER, f"{uid}_annotated.mp4")
            result = engine.predict_video(
                upload_path,
                sample_rate=5,
                output_path=annotated_path,
                temporal_window=5,
            )
            if os.path.exists(annotated_path):
                annotated_url = url_for(
                    "serve_result", filename=f"{uid}_annotated.mp4"
                )
        else:
            heatmap_path = os.path.join(RESULTS_FOLDER, f"{uid}_heatmap.png")
            result = engine.predict_image(
                upload_path,
                save_heatmap=heatmap_path,
            )
            if os.path.exists(heatmap_path):
                heatmap_url = url_for(
                    "serve_result", filename=f"{uid}_heatmap.png"
                )

        # Schedule cleanup
        _delayed_delete(upload_path, heatmap_path, annotated_path)

        # Build safe response (exclude internal paths)
        response = {
            "success":        True,
            "label":          result.get("label") or result.get("verdict"),
            "fake_prob":      result["fake_prob"],
            "real_prob":      result["real_prob"],
            "confidence":     result["confidence"],
            "is_video":       is_video,
            "heatmap_url":    heatmap_url,
            "annotated_url":  annotated_url,
        }

        if is_video:
            response.update({
                "fake_frame_ratio":   result.get("fake_frame_ratio"),
                "frames_analyzed":    result.get("frames_analyzed"),
                "duration_sec":       result.get("duration_sec"),
                "inference_time_sec": result.get("inference_time_sec"),
                "timeline":           result.get("timeline", []),
            })
        else:
            response["inference_time_ms"] = result.get("inference_time_ms")

        return jsonify(response)

    except Exception as e:
        # Clean up on error
        _delayed_delete(upload_path, heatmap_path, annotated_path, delay=0)
        return jsonify({"error": str(e)}), 500


@app.route("/results/<filename>")
def serve_result(filename):
    """Serve heatmap images and annotated videos."""
    return send_from_directory(RESULTS_FOLDER, filename)


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File exceeds {MAX_MB}MB limit"}), 413


# ─────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────
def main():
    global _engine

    parser = argparse.ArgumentParser(description="DeepFake Detector — Flask App")
    parser.add_argument("--ckpt",   type=str, default=None,
                        help="Checkpoint .pth path")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto","cuda","cpu","mps"])
    parser.add_argument("--face",   type=str, default="opencv",
                        choices=["mtcnn","opencv"])
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    print("[App] Loading model...")
    _engine = DeepFakeInference(
        checkpoint_path=args.ckpt,
        device=args.device,
        face_backend=args.face,
    )
    print(f"[App] Ready at http://{args.host}:{args.port}")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
