"""
extract_frames.py
=================
Extracts face frames from UADFV videos and saves them as JPG images.
Run this ONCE before training.

Usage:
    python extract_frames.py --input UADFV/ --output dataset/ --every 5
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, every_nth=5):
    """Extract every Nth frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Skip] Cannot open: {video_path}")
        return 0

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved   = 0
    frame_i = 0

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % every_nth == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_i:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_i += 1

    cap.release()
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="UADFV/",
                        help="Path to UADFV folder (contains real/ and fake/)")
    parser.add_argument("--output", type=str, default="dataset/",
                        help="Output folder for extracted frames")
    parser.add_argument("--every",  type=int, default=5,
                        help="Save every Nth frame (5 = 1 frame per 5 frames)")
    args = parser.parse_args()

    for label in ["real", "fake"]:
        video_dir  = Path(args.input)  / label
        output_dir = Path(args.output) / label

        if not video_dir.exists():
            print(f"[Error] Folder not found: {video_dir}")
            continue

        videos = list(video_dir.glob("*.mp4"))
        print(f"\n[{label.upper()}] Found {len(videos)} videos → extracting frames...")

        total_frames = 0
        for video in tqdm(videos, desc=label):
            # Each video gets its own subfolder
            video_out = output_dir / video.stem
            n = extract_frames(str(video), str(video_out), args.every)
            total_frames += n

        print(f"  Saved {total_frames} frames from {len(videos)} {label} videos")

    print(f"\n✓ Done! Dataset saved to: {args.output}")
    print(f"  Structure:")
    print(f"  {args.output}real/  → real face frames")
    print(f"  {args.output}fake/  → fake face frames")

if __name__ == "__main__":
    main()