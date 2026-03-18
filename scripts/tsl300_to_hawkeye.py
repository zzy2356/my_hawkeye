#!/usr/bin/env python3
"""
Convert TSL-300 dataset to Hawkeye project format.

TSL-300 structure:
  dataset/TSL-300/vid/train/<ID>_<name>.mp4       <- full long videos
  dataset/TSL-300/label/full/train/<ID>.txt        <- timestamps: start,end,class
  dataset/TSL-300/features/train/rgb/<ID>_<name>.npy  <- (N_frames, 1024) rgb features
  dataset/videosenti_gt.json                       <- full annotation DB
  dataset/fps_dict.json                            <- fps per video

Target Hawkeye structure:
  dataset/vid_split/<folder>/1.mp4, 2.mp4, ...    <- 1-second clips
  dataset/vid_noaudio_split/train_new/<folder>/    <- noaudio version (same)
  dataset/pose_feat/train/<folder>/frame_1.npy     <- (5,17,5) pose features
  dataset/rel_feat/<folder>/frame_1.npy            <- (5,353) scene features
  dataset/new_train.json                           <- training annotations

Usage:
    # Step 1: Split videos into 1-second clips
    python scripts/tsl300_to_hawkeye.py --step split \
        --tsl-root dataset/TSL-300 --output-dir dataset

    # Step 2: Generate zero-fill features (if no HigherHRNet/RelTR)
    python scripts/tsl300_to_hawkeye.py --step features \
        --tsl-root dataset/TSL-300 --output-dir dataset

    # Step 3: Generate new_train.json
    python scripts/tsl300_to_hawkeye.py --step json \
        --tsl-root dataset/TSL-300 --output-dir dataset

    # Or run all steps at once:
    python scripts/tsl300_to_hawkeye.py --step all \
        --tsl-root dataset/TSL-300 --output-dir dataset
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fps_dict(dataset_root: str) -> Dict[str, float]:
    path = os.path.join(dataset_root, "fps_dict.json")
    if not os.path.exists(path):
        print(f"[WARN] fps_dict.json not found at {path}, defaulting to 30.0 fps")
        return {}
    with open(path) as f:
        return json.load(f)


def load_gt(dataset_root: str) -> Dict:
    path = os.path.join(dataset_root, "videosenti_gt.json")
    if not os.path.exists(path):
        print(f"[WARN] videosenti_gt.json not found at {path}")
        return {}
    with open(path) as f:
        raw = json.load(f)
    # The GT file has a single key 'database' containing all videos
    if "database" in raw:
        return raw["database"]
    return raw


def load_label_file(label_path: str) -> List[Tuple[float, float, str]]:
    """Parse label txt: start,end,class per line."""
    segments = []
    if not os.path.exists(label_path):
        return segments
    with open(label_path) as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            start, end, cls = float(parts[0]), float(parts[1]), parts[2].strip()
            segments.append((start, end, cls))
        except ValueError:
            continue
    return segments


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                dur = stream.get("duration")
                if dur:
                    return float(dur)
        # Fallback to format duration
        cmd2 = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        info2 = json.loads(result2.stdout)
        return float(info2.get("format", {}).get("duration", 0))
    except Exception as e:
        print(f"[WARN] ffprobe failed for {video_path}: {e}")
        return 0.0


def split_video_into_clips(video_path: str, output_dir: str, clip_duration: float = 1.0) -> List[str]:
    """
    Split a video into fixed-length clips using ffmpeg.
    Returns list of output clip paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_duration(video_path)
    if duration <= 0:
        print(f"[WARN] Cannot get duration for {video_path}, skipping")
        return []

    clips = []
    t = 0.0
    clip_idx = 1
    while t + clip_duration <= duration + 0.01:  # small tolerance
        start = round(t, 3)
        end = round(t + clip_duration, 3)
        out_name = f"{start}_{end}.mp4"
        out_path = os.path.join(output_dir, out_name)

        if not os.path.exists(out_path):
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
                "-t", str(clip_duration),
                "-c:v", "libx264", "-an",  # -an = no audio
                "-loglevel", "error",
                out_path
            ]
            subprocess.run(cmd, timeout=60)

        clips.append(out_path)
        t += clip_duration
        clip_idx += 1

    return clips


def map_rgb_to_frame_features(rgb_feat: np.ndarray, num_clips: int,
                               target_frames: int = 5) -> np.ndarray:
    """
    Map the (N_frames, 1024) rgb feature matrix to per-clip features.

    For each clip i (out of num_clips), we sample `target_frames` evenly-spaced
    rows from the portion of rgb_feat that corresponds to that clip.

    Returns: (num_clips, target_frames, 1024)
    """
    total_feat_frames = rgb_feat.shape[0]
    result = np.zeros((num_clips, target_frames, rgb_feat.shape[1]), dtype=np.float32)

    for clip_i in range(num_clips):
        # Which portion of the feature matrix belongs to this clip
        feat_start = int(round(clip_i / num_clips * total_feat_frames))
        feat_end = int(round((clip_i + 1) / num_clips * total_feat_frames))
        feat_end = max(feat_end, feat_start + 1)
        feat_end = min(feat_end, total_feat_frames)

        chunk = rgb_feat[feat_start:feat_end]  # (K, 1024)
        # Sample target_frames rows from chunk
        indices = np.linspace(0, len(chunk) - 1, target_frames, dtype=int)
        result[clip_i] = chunk[indices]

    return result


def determine_clip_label(clip_start: float, clip_end: float,
                          segments: List[Tuple[float, float, str]]) -> str:
    """
    Determine binary label for a clip based on ground truth segments.
    Label = '1' (negative) if clip overlaps any 'negative'/'n' segment.
    Label = '0' otherwise.
    """
    for seg_start, seg_end, cls in segments:
        # Check overlap
        overlap = min(clip_end, seg_end) - max(clip_start, seg_start)
        if overlap > 0 and cls.lower() in ('negative', 'n'):
            return '1'
    return '0'


# ---------------------------------------------------------------------------
# Step 1: Split videos
# ---------------------------------------------------------------------------

def step_split_videos(tsl_root: str, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Splitting videos into 1-second clips")
    print("=" * 60)

    for split in ("train", "test"):
        vid_dir = os.path.join(tsl_root, "vid", split)
        if not os.path.isdir(vid_dir):
            print(f"[SKIP] Video dir not found: {vid_dir}")
            continue

        # Output directories
        vid_split_dir = os.path.join(output_dir, "vid_split")
        vid_noaudio_dir = os.path.join(output_dir, "vid_noaudio_split",
                                       "train_new" if split == "train" else "test_new")

        video_files = [f for f in os.listdir(vid_dir) if f.endswith(".mp4")]
        print(f"Found {len(video_files)} {split} videos")

        for video_file in sorted(video_files):
            folder_name = os.path.splitext(video_file)[0]  # e.g. '1_Ekman6_disgust_3'
            video_path = os.path.join(vid_dir, video_file)

            out_dir = os.path.join(vid_split_dir, folder_name)
            out_noaudio_dir = os.path.join(vid_noaudio_dir, folder_name)

            if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
                print(f"  [SKIP] Already split: {folder_name} ({len(os.listdir(out_dir))} clips)")
                continue

            print(f"  Splitting: {folder_name}")
            clips = split_video_into_clips(video_path, out_dir, clip_duration=1.0)

            # Create symlinks or copies for noaudio dir (already no audio from ffmpeg -an)
            os.makedirs(out_noaudio_dir, exist_ok=True)
            for clip_path in clips:
                clip_name = os.path.basename(clip_path)
                noaudio_path = os.path.join(out_noaudio_dir, clip_name)
                if not os.path.exists(noaudio_path):
                    # Hard link to save space
                    try:
                        os.link(clip_path, noaudio_path)
                    except OSError:
                        import shutil
                        shutil.copy2(clip_path, noaudio_path)

            print(f"    -> {len(clips)} clips created")

    print("\nStep 1 complete.")


# ---------------------------------------------------------------------------
# Step 2: Generate features
# ---------------------------------------------------------------------------

def step_generate_features(tsl_root: str, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Generating features (pose + scene)")
    print("=" * 60)
    print("NOTE: Using zero-fill for pose (17,5) and scene (353) dimensions.")
    print("      TSL-300 provides RGB features (1024-dim) which are mapped to")
    print("      scene features via linear projection placeholder.")
    print("      Replace with real HigherHRNet/RelTR features when available.")

    vid_split_dir = os.path.join(output_dir, "vid_split")
    if not os.path.isdir(vid_split_dir):
        print(f"[ERROR] vid_split not found: {vid_split_dir}")
        print("        Run --step split first.")
        return

    TARGET_FRAMES = 5
    POSE_DIM = (17, 5)
    SCENE_DIM = 353
    RGB_DIM = 1024

    for split in ("train", "test"):
        rgb_feat_dir = os.path.join(tsl_root, "features", split, "rgb")
        has_rgb = os.path.isdir(rgb_feat_dir)

        for folder_name in sorted(os.listdir(vid_split_dir)):
            folder_path = os.path.join(vid_split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            clips = sorted([f for f in os.listdir(folder_path) if f.endswith(".mp4")])
            if not clips:
                continue
            num_clips = len(clips)

            # Output dirs
            pose_dir = os.path.join(output_dir, "pose_feat", "train", folder_name)
            scene_dir = os.path.join(output_dir, "rel_feat", folder_name)
            os.makedirs(pose_dir, exist_ok=True)
            os.makedirs(scene_dir, exist_ok=True)

            # Try to load RGB features for scene
            rgb_feat_path = os.path.join(rgb_feat_dir, f"{folder_name}.npy") if has_rgb else None
            rgb_feat = None
            if rgb_feat_path and os.path.exists(rgb_feat_path):
                try:
                    rgb_feat = np.load(rgb_feat_path).astype(np.float32)  # (N, 1024)
                except Exception as e:
                    print(f"    [WARN] Failed to load rgb feat {rgb_feat_path}: {e}")

            # Per-clip features
            for clip_idx, clip_file in enumerate(clips):
                # Parse clip start time from filename e.g. '0.0_1.0.mp4'
                clip_stem = os.path.splitext(clip_file)[0]  # '0.0_1.0'
                # train.py uses: int(video_file.split('.')[0])
                # '0.0_1.0.mp4' -> split('.')[0] -> '0' -> int -> 0
                # so frame files must be frame_0.npy, frame_1.npy, ...
                try:
                    frame_id = int(clip_stem.split('.')[0])  # '0.0_1.0' -> '0' -> 0
                except ValueError:
                    frame_id = clip_idx  # fallback

                pose_out = os.path.join(pose_dir, f"frame_{frame_id}.npy")
                scene_out = os.path.join(scene_dir, f"frame_{frame_id}.npy")

                # --- Pose feature: zero-fill (5, 17, 5) ---
                if not os.path.exists(pose_out):
                    pose_feat = np.zeros((TARGET_FRAMES, *POSE_DIM), dtype=np.float32)
                    np.save(pose_out, pose_feat)

                # --- Scene feature: from rgb or zero-fill (5, 353) ---
                if not os.path.exists(scene_out):
                    if rgb_feat is not None:
                        # Map rgb (N,1024) -> (5,1024) for this clip, then project to (5,353)
                        # Simple: take 5 evenly-spaced frames from clip's portion of rgb
                        total_feat = rgb_feat.shape[0]
                        feat_start = int(round(clip_idx / num_clips * total_feat))
                        feat_end = int(round((clip_idx + 1) / num_clips * total_feat))
                        feat_end = max(feat_end, feat_start + 1)
                        feat_end = min(feat_end, total_feat)
                        chunk = rgb_feat[feat_start:feat_end]  # (K, 1024)
                        indices = np.linspace(0, len(chunk) - 1, TARGET_FRAMES, dtype=int)
                        sampled = chunk[indices]  # (5, 1024)
                        # Truncate 1024 -> 353 (take first 353 dims as placeholder)
                        scene_feat = sampled[:, :SCENE_DIM]  # (5, 353)
                    else:
                        scene_feat = np.zeros((TARGET_FRAMES, SCENE_DIM), dtype=np.float32)
                    np.save(scene_out, scene_feat)

            if (list(sorted(os.listdir(vid_split_dir))).index(folder_name) + 1) % 20 == 0:
                print(f"  Processed {list(sorted(os.listdir(vid_split_dir))).index(folder_name) + 1} folders...")

    print("\nStep 2 complete.")


# ---------------------------------------------------------------------------
# Step 3: Generate new_train.json
# ---------------------------------------------------------------------------

def step_generate_json(tsl_root: str, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Generating new_train.json")
    print("=" * 60)

    vid_split_dir = os.path.join(output_dir, "vid_split")
    if not os.path.isdir(vid_split_dir):
        print(f"[ERROR] vid_split not found: {vid_split_dir}")
        print("        Run --step split first.")
        return

    # Load split lists
    split_train_file = os.path.join(output_dir, "split_train.txt")
    split_test_file = os.path.join(output_dir, "split_test.txt")

    with open(split_train_file) as f:
        train_folders = set(line.strip() for line in f if line.strip())

    # Load ground truth labels
    gt_db = load_gt(output_dir)

    PROMPT = (
        "Please determine whether the emotional attributes of the video are negative or not. "
        "If negative, answer 1, else answer 0. "
        "The answer should just contain 0 or 1 without other contents.\n<video>"
    )

    annotations = []
    skipped = 0

    for folder_name in sorted(train_folders):
        folder_path = os.path.join(vid_split_dir, folder_name)
        if not os.path.isdir(folder_path):
            skipped += 1
            continue

        clips = sorted([f for f in os.listdir(folder_path) if f.endswith(".mp4")])
        if not clips:
            skipped += 1
            continue

        # Load ground truth segments for this video
        gt_entry = gt_db.get(folder_name, {})
        gt_segments: List[Tuple[float, float, str]] = []
        for ann in gt_entry.get("annotations", []):
            try:
                seg_start = float(ann["segment"][0])
                seg_end = float(ann["segment"][1])
                label_str = ann["label"]  # 'n' or 'p'
                gt_segments.append((seg_start, seg_end, label_str))
            except (KeyError, ValueError, IndexError):
                continue

        # Fallback: use label txt file if gt_db doesn't have this folder
        if not gt_segments:
            # Try numeric ID from folder name (e.g. '1_Ekman6_disgust_3' -> '1')
            numeric_id = folder_name.split("_")[0]
            label_txt = os.path.join(tsl_root, "label", "full", "train", f"{numeric_id}.txt")
            gt_segments = load_label_file(label_txt)

        # Generate annotation for each clip
        for clip_file in clips:
            clip_stem = os.path.splitext(clip_file)[0]  # '0.0_1.0'
            try:
                parts = clip_stem.split("_")
                clip_start = float(parts[0])
                clip_end = float(parts[1])
            except (IndexError, ValueError):
                clip_start, clip_end = 0.0, 1.0

            label = determine_clip_label(clip_start, clip_end, gt_segments)

            annotations.append({
                "path": f"{folder_name}/{clip_file}",
                "label": label,
                "mode": "train",
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt", "value": label},
                ]
            })

    out_json = os.path.join(output_dir, "new_train.json")
    with open(out_json, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    neg = sum(1 for a in annotations if a["label"] == "1")
    pos = sum(1 for a in annotations if a["label"] == "0")
    print(f"Generated {len(annotations)} annotations ({neg} negative, {pos} non-negative)")
    print(f"Skipped {skipped} folders (not in vid_split)")
    print(f"Saved to: {out_json}")
    print("\nStep 3 complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TSL-300 dataset to Hawkeye project format"
    )
    parser.add_argument(
        "--step",
        choices=["split", "features", "json", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--tsl-root",
        default="dataset/TSL-300",
        help="Path to TSL-300 root directory (default: dataset/TSL-300)",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Output dataset root directory (default: dataset)",
    )
    args = parser.parse_args()

    tsl_root = args.tsl_root
    output_dir = args.output_dir

    if not os.path.isdir(tsl_root):
        print(f"[ERROR] TSL-300 root not found: {tsl_root}")
        sys.exit(1)

    if args.step in ("split", "all"):
        step_split_videos(tsl_root, output_dir)

    if args.step in ("features", "all"):
        step_generate_features(tsl_root, output_dir)

    if args.step in ("json", "all"):
        step_generate_json(tsl_root, output_dir)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"  vid_split/           -> {os.path.join(output_dir, 'vid_split')}")
    print(f"  vid_noaudio_split/   -> {os.path.join(output_dir, 'vid_noaudio_split')}")
    print(f"  pose_feat/train/     -> {os.path.join(output_dir, 'pose_feat', 'train')}")
    print(f"  rel_feat/            -> {os.path.join(output_dir, 'rel_feat')}")
    print(f"  new_train.json       -> {os.path.join(output_dir, 'new_train.json')}")
    print("\nNext steps:")
    print("  1. Verify: python scripts/verify_dataset.py")
    print("  2. Smoke:  python scripts/qwen3vl/smoke_infer.py --model-path models/Qwen3-VL-8B-Instruct ...")
    print("  3. Train:  bash scripts/qwen3vl/train_debug.sh")


if __name__ == "__main__":
    main()
