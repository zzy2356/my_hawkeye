#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from collections import defaultdict

import cv2


def parse_clip_range(file_name: str):
    stem = os.path.splitext(file_name)[0]
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid clip file name: {file_name}")
    return float(parts[0]), float(parts[1])


def load_unique_clips(json_path: str):
    with open(json_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    by_folder = defaultdict(set)
    for item in records:
        rel_path = item.get("path", "")
        if "/" not in rel_path:
            continue
        folder, file_name = rel_path.split("/", 1)
        by_folder[folder].add(file_name)
    return by_folder


def find_source_video(tsl_root: str, ucf_root: str, folder: str):
    candidates = [
        os.path.join(tsl_root, "vid", "train", f"{folder}.mp4"),
        os.path.join(tsl_root, "vid", "train", f"{folder}.avi"),
        os.path.join(tsl_root, "vid", "train", f"{folder}.mov"),
        os.path.join(tsl_root, "vid", "train", f"{folder}.mkv"),
    ]

    matched = re.match(r"^([A-Za-z]+)[0-9]+_x264$", folder)
    if matched and ucf_root:
        ucf_class = matched.group(1)
        candidates.extend(
            [
                os.path.join(ucf_root, ucf_class, f"{folder}.mp4"),
                os.path.join(ucf_root, ucf_class, f"{folder}.avi"),
                os.path.join(ucf_root, ucf_class, f"{folder}.mov"),
                os.path.join(ucf_root, ucf_class, f"{folder}.mkv"),
            ]
        )

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _cut_clip_with_opencv(src_video: str, dst_video: str, start: float, end: float):
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {src_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open destination video writer: {dst_video}")

    start_frame = max(0, int(round(start * fps)))
    end_frame = max(start_frame + 1, int(round(end * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()


def cut_clip(src_video: str, dst_video: str, start: float, end: float):
    os.makedirs(os.path.dirname(dst_video), exist_ok=True)
    duration = max(0.001, end - start)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        src_video,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-an",
        "-loglevel",
        "error",
        dst_video,
    ]
    try:
        subprocess.run(cmd, check=True)
        return
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass

    _cut_clip_with_opencv(src_video, dst_video, start, end)


def main():
    parser = argparse.ArgumentParser(description="Materialize clips from original new_train.json")
    parser.add_argument("--json-path", type=str, default="dataset/new_train.json")
    parser.add_argument("--tsl-root", type=str, default="dataset/TSL-300")
    parser.add_argument("--ucf-root", type=str, default="dataset/UCF-Crime")
    parser.add_argument("--output-root", type=str, default="dataset/vid_noaudio_split/train_new")
    parser.add_argument("--max-folders", type=int, default=0, help="Debug only: process first N folders, 0=all")
    parser.add_argument("--folders", type=str, default="", help="Comma-separated folder names to process")
    args = parser.parse_args()

    clips_by_folder = load_unique_clips(args.json_path)
    folders = sorted(clips_by_folder.keys())
    if args.folders.strip():
        requested = [name.strip() for name in args.folders.split(",") if name.strip()]
        folders = [name for name in requested if name in clips_by_folder]
    if args.max_folders > 0:
        folders = folders[: args.max_folders]

    total = 0
    existed = 0
    created = 0
    missing_src = 0
    failed = 0

    for folder in folders:
        src_video = find_source_video(args.tsl_root, args.ucf_root, folder)
        if src_video is None:
            print(f"[WARN] source video missing: {folder}")
            missing_src += 1
            continue

        for clip_name in sorted(clips_by_folder[folder]):
            total += 1
            dst = os.path.join(args.output_root, folder, clip_name)
            if os.path.exists(dst):
                existed += 1
                continue

            try:
                start, end = parse_clip_range(clip_name)
                cut_clip(src_video, dst, start, end)
                created += 1
            except Exception as exc:
                failed += 1
                print(f"[WARN] failed clip {folder}/{clip_name}: {exc}")

    print("\n=== Materialize Summary ===")
    print(f"folders={len(folders)}")
    print(f"clips_total={total}")
    print(f"clips_existed={existed}")
    print(f"clips_created={created}")
    print(f"missing_source_videos={missing_src}")
    print(f"clip_failures={failed}")


if __name__ == "__main__":
    main()
