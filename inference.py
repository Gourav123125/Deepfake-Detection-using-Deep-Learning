"""
inference.py
============
Inference engine for DeepFake Detection.

Supports:
  - Single image prediction + Grad-CAM heatmap
  - Video prediction (frame-by-frame + temporal smoothing)
  - Batch image prediction
  - Annotated video output with per-frame verdicts

CLI Usage:
    # Image inference
    python inference.py image.jpg --ckpt checkpoints/deepfake_best.pth

    # Image with Grad-CAM heatmap
    python inference.py image.jpg --ckpt checkpoints/deepfake_best.pth \
        --heatmap output/heatmap.png

    # Video inference
    python inference.py video.mp4 --ckpt checkpoints/deepfake_best.pth \
        --output output/annotated.mp4 --sample-rate 5

    # JSON output
    python inference.py media.jpg --ckpt best.pth --json result.json
"""

import os
import cv2
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from collections import deque

from model.architecture import build_model
from utils.transforms import get_transforms
from utils.face_detector import FaceDetector
from utils.metrics import save_gradcam_overlay


# ─────────────────────────────────────────────────────────────────
#  Inference Engine
# ─────────────────────────────────────────────────────────────────
class DeepFakeInference:
    """
    Production-ready inference wrapper.

    Args:
        checkpoint_path: Path to .pth checkpoint file (None = ImageNet weights only)
        device:          'auto', 'cuda', 'cpu', or 'mps'
        face_backend:    'mtcnn' or 'opencv'
    """

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        face_backend: str = "opencv",
    ):
        # ── Device ───────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device(
                "cuda"  if torch.cuda.is_available() else
                "mps"   if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"[Inference] Device: {self.device}")

        # ── Model ────────────────────────────────────────────────
        self.model = build_model(pretrained=(checkpoint_path is None))
        self.model.to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            epoch   = ckpt.get("epoch", "?")
            metrics = ckpt.get("metrics", {})
            print(f"[Inference] Loaded '{checkpoint_path}' (epoch {epoch})")
            if metrics:
                print(
                    f"  Checkpoint metrics — "
                    f"AUC: {metrics.get('auc', 0):.4f} | "
                    f"Acc: {metrics.get('acc', 0):.4f} | "
                    f"F1: {metrics.get('f1', 0):.4f}"
                )
        elif checkpoint_path:
            print(f"[Warning] Checkpoint not found: {checkpoint_path}")
            print("[Warning] Running with pretrained ImageNet weights only.")
        else:
            print("[Inference] No checkpoint — using pretrained ImageNet weights.")

        self.model.eval()

        # ── Face Detector ─────────────────────────────────────────
        self.face_detector = FaceDetector(
            backend=face_backend,
            device=str(self.device),
        )

        # ── Transforms ───────────────────────────────────────────
        self.transform = get_transforms("val")

    # ── Preprocessing ─────────────────────────────────────────────
    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Detect face, crop, transform, batch.
        Returns: (1, 3, H, W) tensor
        """
        face = self.face_detector.detect_and_crop(img)
        tensor = self.transform(face).unsqueeze(0).to(self.device)
        return tensor

    # ─────────────────────────────────────────────────────────────
    #  Image Inference
    # ─────────────────────────────────────────────────────────────
    def predict_image(
        self,
        image_path: str,
        save_heatmap: Optional[str] = None,
    ) -> Dict:
        """
        Predict whether an image is real or fake.

        Args:
            image_path:   Path to input image
            save_heatmap: Optional path to save Grad-CAM overlay PNG

        Returns:
            {label, fake_prob, real_prob, confidence, inference_time_ms}
        """
        t0 = time.time()
        img = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(img)

        label, fake_prob, real_prob = self.model.predict(tensor, mode="image")
        confidence = max(fake_prob, real_prob)
        elapsed_ms = (time.time() - t0) * 1000

        result = {
            "path":              image_path,
            "label":             label,
            "fake_prob":         round(fake_prob, 4),
            "real_prob":         round(real_prob, 4),
            "confidence":        round(confidence, 4),
            "inference_time_ms": round(elapsed_ms, 1),
        }

        if save_heatmap:
            self._save_image_heatmap(img, tensor, result, save_heatmap)

        return result

    def _save_image_heatmap(
        self,
        original_img: Image.Image,
        tensor: torch.Tensor,
        result: Dict,
        save_path: str,
    ):
        """Generate and save Grad-CAM heatmap overlay."""
        os.makedirs(Path(save_path).parent, exist_ok=True)
        try:
            heatmap = self.model.grad_cam(tensor.clone(), target_class=1)
            face_crop = self.face_detector.detect_and_crop(original_img)
            save_gradcam_overlay(face_crop, heatmap, save_path, result)
        except Exception as e:
            print(f"  [Warning] Grad-CAM failed: {e}")

    # ─────────────────────────────────────────────────────────────
    #  Video Inference
    # ─────────────────────────────────────────────────────────────
    def predict_video(
        self,
        video_path: str,
        sample_rate: int = 5,
        output_path: Optional[str] = None,
        temporal_window: int = 5,
        save_heatmap_dir: Optional[str] = None,
    ) -> Dict:
        """
        Analyze a video for deepfake manipulation.

        Args:
            video_path:       Path to input video file
            sample_rate:      Analyze every Nth frame (5 = every 5th frame)
            output_path:      Save annotated video here
            temporal_window:  Smoothing window for per-frame verdict
            save_heatmap_dir: Save per-frame heatmaps to this directory

        Returns:
            Full results dict with per-frame analysis and video verdict
        """
        t0 = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration  = total / fps

        print(
            f"\n[Video] {Path(video_path).name}  |  "
            f"{total} frames @ {fps:.1f} fps  |  {duration:.1f}s"
        )

        # ── Setup writer ─────────────────────────────────────────
        writer = None
        if output_path:
            os.makedirs(Path(output_path).parent, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if save_heatmap_dir:
            os.makedirs(save_heatmap_dir, exist_ok=True)

        # ── Process frames ────────────────────────────────────────
        per_frame_results = []
        fake_probs        = []
        smooth_window     = deque(maxlen=temporal_window)
        frame_idx         = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                # Convert BGR → RGB → PIL
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)

                # Predict
                tensor = self._preprocess(pil)
                label, fake_prob, real_prob = self.model.predict(tensor, mode="image")
                fake_probs.append(fake_prob)
                smooth_window.append(fake_prob)

                # Temporal smoothing
                smoothed       = float(np.mean(smooth_window))
                smooth_label   = "FAKE" if smoothed >= 0.5 else "REAL"

                per_frame_results.append({
                    "frame":      frame_idx,
                    "time_sec":   round(frame_idx / fps, 2),
                    "label":      label,
                    "fake_prob":  round(fake_prob, 4),
                    "smooth_prob": round(smoothed, 4),
                })

                # Optional heatmap for this frame
                if save_heatmap_dir:
                    heatmap_path = os.path.join(
                        save_heatmap_dir, f"frame_{frame_idx:06d}.png"
                    )
                    face = self.face_detector.detect_and_crop(pil)
                    try:
                        hmap = self.model.grad_cam(tensor.clone(), target_class=1)
                        save_gradcam_overlay(
                            face, hmap, heatmap_path,
                            {"label": label, "fake_prob": fake_prob,
                             "real_prob": real_prob, "confidence": max(fake_prob, real_prob)}
                        )
                    except Exception:
                        pass

                # Annotate frame if writing output video
                if writer:
                    frame = self._annotate_frame(
                        frame, smooth_label, smoothed, frame_idx, fps
                    )

            if writer:
                writer.write(frame)

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  Processed {frame_idx}/{total} frames "
                      f"({frame_idx/total*100:.0f}%)...")

        cap.release()
        if writer:
            writer.release()
            print(f"  [Output] Annotated video saved → {output_path}")

        # ── Aggregate verdict ─────────────────────────────────────
        overall_fake  = float(np.mean(fake_probs)) if fake_probs else 0.5
        fake_ratio    = sum(1 for p in fake_probs if p >= 0.5) / max(len(fake_probs), 1)
        verdict       = "FAKE" if overall_fake >= 0.5 else "REAL"
        elapsed       = time.time() - t0

        # Timeline: 10-second segments
        timeline = self._build_timeline(per_frame_results, fps)

        print(f"\n  ═══ VIDEO VERDICT: {verdict} ═══")
        print(f"  Fake Probability:  {overall_fake:.1%}")
        print(f"  Fake Frame Ratio:  {fake_ratio:.1%}")
        print(f"  Frames Analyzed:   {len(fake_probs)}")
        print(f"  Processing Time:   {elapsed:.1f}s")

        return {
            "path":              video_path,
            "verdict":           verdict,
            "fake_prob":         round(overall_fake, 4),
            "real_prob":         round(1 - overall_fake, 4),
            "confidence":        round(max(overall_fake, 1 - overall_fake), 4),
            "fake_frame_ratio":  round(fake_ratio, 4),
            "frames_analyzed":   len(fake_probs),
            "duration_sec":      round(duration, 2),
            "inference_time_sec": round(elapsed, 2),
            "per_frame":         per_frame_results,
            "timeline":          timeline,
        }

    @staticmethod
    def _annotate_frame(
        frame: np.ndarray,
        label: str,
        prob: float,
        frame_idx: int,
        fps: float,
    ) -> np.ndarray:
        """Draw verdict overlay on a video frame."""
        h, w = frame.shape[:2]
        is_fake = label == "FAKE"

        # Background bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        # Verdict text
        color = (40, 40, 220) if is_fake else (40, 180, 40)   # BGR
        cv2.putText(
            frame,
            f"{label}  {prob:.1%}",
            (12, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            color,
            2,
            cv2.LINE_AA,
        )

        # Timestamp
        ts = f"{frame_idx/fps:.1f}s"
        cv2.putText(frame, ts, (w - 90, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

        # Side indicator bar
        bar_color = color
        cv2.rectangle(frame, (0, 0), (6, h), bar_color, -1)

        return frame

    @staticmethod
    def _build_timeline(per_frame: List[Dict], fps: float) -> List[Dict]:
        """Aggregate per-frame results into 10-second segments."""
        if not per_frame:
            return []
        max_time = per_frame[-1]["time_sec"]
        timeline = []
        for seg_start in range(0, int(max_time) + 1, 10):
            seg_end = seg_start + 10
            frames = [f for f in per_frame if seg_start <= f["time_sec"] < seg_end]
            if frames:
                prob = float(np.mean([f["fake_prob"] for f in frames]))
                timeline.append({
                    "time_start":  seg_start,
                    "time_end":    seg_end,
                    "fake_prob":   round(prob, 4),
                    "label":       "FAKE" if prob >= 0.5 else "REAL",
                    "frame_count": len(frames),
                })
        return timeline

    # ─────────────────────────────────────────────────────────────
    #  Batch Inference
    # ─────────────────────────────────────────────────────────────
    def batch_predict(
        self,
        paths: List[str],
        save_heatmaps_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        Run inference on a list of image paths.

        Args:
            paths:             List of image file paths
            save_heatmaps_dir: Optional directory to save Grad-CAM PNGs

        Returns:
            List of result dicts
        """
        results = []
        for i, path in enumerate(paths, 1):
            heatmap_path = None
            if save_heatmaps_dir:
                stem = Path(path).stem
                heatmap_path = os.path.join(save_heatmaps_dir, f"{stem}_gradcam.png")

            print(f"[{i:3d}/{len(paths)}] {Path(path).name}", end="  ")
            r = self.predict_image(path, save_heatmap=heatmap_path)
            print(f"→ {r['label']}  fake: {r['fake_prob']:.1%}  "
                  f"({r['inference_time_ms']:.0f}ms)")
            results.append(r)

        return results

    # ─────────────────────────────────────────────────────────────
    #  Auto-detect and predict
    # ─────────────────────────────────────────────────────────────
    def predict(
        self,
        path: str,
        **kwargs,
    ) -> Dict:
        """
        Automatically detect media type and run appropriate prediction.
        Convenience method for CLI and Flask app.
        """
        ext = Path(path).suffix.lower()
        if ext in self.VIDEO_EXTS:
            return self.predict_video(path, **kwargs)
        elif ext in self.IMAGE_EXTS:
            return self.predict_image(path,
                                      save_heatmap=kwargs.get("save_heatmap"))
        else:
            raise ValueError(f"Unsupported file type: {ext}")


# ─────────────────────────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DeepFake Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",        type=str, help="Image or video file path")
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Checkpoint .pth file path")
    parser.add_argument("--heatmap",    type=str, default=None,
                        help="Save Grad-CAM heatmap to this PNG path")
    parser.add_argument("--output",     type=str, default=None,
                        help="Save annotated video to this path")
    parser.add_argument("--sample-rate",type=int, default=5,
                        help="Analyze every Nth video frame")
    parser.add_argument("--json",       type=str, default=None,
                        help="Save results as JSON to this path")
    parser.add_argument("--device",     type=str, default="auto",
                        choices=["auto","cuda","cpu","mps"])
    parser.add_argument("--face",       type=str, default="opencv",
                        choices=["mtcnn","opencv"],
                        help="Face detection backend")
    args = parser.parse_args()

    engine = DeepFakeInference(
        checkpoint_path=args.ckpt,
        device=args.device,
        face_backend=args.face,
    )

    ext = Path(args.input).suffix.lower()

    if ext in DeepFakeInference.VIDEO_EXTS:
        result = engine.predict_video(
            args.input,
            sample_rate=args.sample_rate,
            output_path=args.output,
        )
        print(f"\n{'═'*55}")
        print(f"  VERDICT:        {result['verdict']}")
        print(f"  Fake Prob:      {result['fake_prob']:.1%}")
        print(f"  Real Prob:      {result['real_prob']:.1%}")
        print(f"  Fake Frames:    {result['fake_frame_ratio']:.1%} of "
              f"{result['frames_analyzed']} analyzed")
        print(f"  Duration:       {result['duration_sec']:.1f}s")
        print(f"  Inference:      {result['inference_time_sec']:.1f}s")

    else:
        result = engine.predict_image(args.input, save_heatmap=args.heatmap)
        print(f"\n{'═'*55}")
        print(f"  VERDICT:        {result['label']}")
        print(f"  Fake Prob:      {result['fake_prob']:.1%}")
        print(f"  Real Prob:      {result['real_prob']:.1%}")
        print(f"  Confidence:     {result['confidence']:.1%}")
        print(f"  Inference:      {result['inference_time_ms']:.0f}ms")

    if args.json:
        safe = {k: v for k, v in result.items() if k != "heatmap"}
        with open(args.json, "w") as f:
            json.dump(safe, f, indent=2)
        print(f"\n  Results saved → {args.json}")


if __name__ == "__main__":
    main()
