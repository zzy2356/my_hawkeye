#!/usr/bin/env python3
"""
Hawkeye dataset preparation helper.

This script aligns the dataset layout with the current Qwen-Hawkeye pipeline:

- train videos: dataset/vid_noaudio_split/train_new/<folder>/<index>.mp4
- test videos: dataset/vid_noaudio_split/test_new/<folder>/<index>.mp4
- train pose: dataset/pose_feat/train/<folder>/frame_<index>.npy
- test pose: dataset/pose_feat/test/<folder>/frame_<index>.npy
- train scene: dataset/rel_feat/train/<folder>/frame_<index>.npy
- test scene: dataset/rel_feat/test/<folder>/frame_<index>.npy
- train json: dataset/new_train.json

The script does not run HigherHRNet or RelTR by itself. It creates the expected
directory layout and prints the exact feature format required by training and
evaluation.
"""

import argparse
import json
import logging
import os
import shutil
from typing import List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_split_file(split_file: str) -> List[str]:
    if not os.path.exists(split_file):
        logger.warning("Split file not found: %s", split_file)
        return []

    with open(split_file, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def organize_videos(source_dir: str, target_dir: str, split_folders: List[str]) -> None:
    os.makedirs(target_dir, exist_ok=True)

    for folder in split_folders:
        source_folder = os.path.join(source_dir, folder)
        if not os.path.isdir(source_folder):
            logger.warning("Folder not found: %s", source_folder)
            continue

        target_folder = os.path.join(target_dir, folder)
        os.makedirs(target_folder, exist_ok=True)

        video_files = sorted(
            file_name for file_name in os.listdir(source_folder) if file_name.endswith((".mp4", ".avi", ".mov"))
        )
        for idx, video_file in enumerate(video_files, start=1):
            source_path = os.path.join(source_folder, video_file)
            target_path = os.path.join(target_folder, f"{idx}.mp4")
            if os.path.exists(target_path):
                logger.info("Already exists: %s", target_path)
                continue
            logger.info("Copying %s -> %s", source_path, target_path)
            shutil.copy2(source_path, target_path)


def _find_tsl_video_file(tsl_root: str, split_name: str, folder: str) -> Optional[str]:
    candidates = []
    vid_root = os.path.join(tsl_root, "vid")
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        candidates.append(os.path.join(vid_root, split_name, f"{folder}{ext}"))
        candidates.append(os.path.join(vid_root, f"{folder}{ext}"))
        candidates.append(os.path.join(tsl_root, split_name, f"{folder}{ext}"))
        candidates.append(os.path.join(tsl_root, f"{folder}{ext}"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def organize_tsl300_flat_videos(tsl_root: str, target_dir: str, split_folders: List[str], split_name: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    copied = 0
    missing = 0

    for folder in split_folders:
        source_path = _find_tsl_video_file(tsl_root, split_name=split_name, folder=folder)
        if source_path is None:
            missing += 1
            logger.warning("TSL-300 video not found for split=%s, folder=%s", split_name, folder)
            continue

        target_folder = os.path.join(target_dir, folder)
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, "1.mp4")
        if os.path.exists(target_path):
            logger.info("Already exists: %s", target_path)
            continue
        logger.info("Copying %s -> %s", source_path, target_path)
        shutil.copy2(source_path, target_path)
        copied += 1

    logger.info("TSL-300 organize summary (%s): copied=%s missing=%s", split_name, copied, missing)


def prepare_feature_layout(
    output_dir: str,
    split_folders: List[str],
    split_name: str,
    feature_name: str,
    expected_shape: str,
    extractor_name: str,
    extractor_url: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for folder in split_folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    logger.info("%s extraction requires %s installation", feature_name.capitalize(), extractor_name)
    logger.info("Please refer to: %s", extractor_url)
    logger.info("Expected %s output layout for split '%s': %s/<folder>/frame_<index>.npy", feature_name, split_name, output_dir)
    logger.info("Expected %s feature shape: %s", feature_name, expected_shape)


def generate_train_json(video_dir: str, split_folders: List[str], output_file: str, task: str = "iasdig") -> None:
    annotations = []

    if task == "iasdig":
        prompt = (
            "Please determine whether the emotional attributes of the video are negative or not. "
            "If negative, answer 1, else answer 0. The answer should just contain 0 or 1 without other contents.\n<video>"
        )
    elif task == "ucf":
        prompt = (
            "Please determine whether the video is an anomalistic video that contains one of Abuse, Arrest, Arson, "
            "Assault, Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and "
            "Vandalism. Answer 1 if yes, 0 if no.\n<video>"
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    for folder in split_folders:
        folder_path = os.path.join(video_dir, folder)
        if not os.path.isdir(folder_path):
            logger.warning("Folder not found: %s", folder_path)
            continue

        video_files = sorted(
            file_name for file_name in os.listdir(folder_path) if file_name.endswith((".mp4", ".avi", ".mov"))
        )
        for idx, _ in enumerate(video_files, start=1):
            if task == "iasdig":
                negative_emotions = ["disgust", "anger", "fear", "sadness"]
                label = "1" if any(emotion in folder.lower() for emotion in negative_emotions) else "0"
            else:
                label = "0" if "Normal" in folder else "1"

            annotations.append(
                {
                    "path": f"{folder}/{idx}.mp4",
                    "label": label,
                    "mode": "train",
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": label},
                    ],
                }
            )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(annotations, file, indent=2, ensure_ascii=False)

    logger.info("Generated %s annotations in %s", len(annotations), output_file)


def _resolve_scene_feature_file(scene_root: str, folder: str, frame_file: str) -> str:
    candidates = [
        os.path.join(scene_root, "graph_feat", "train", folder, frame_file),
        os.path.join(scene_root, "graph_feat", folder, frame_file),
        os.path.join(scene_root, "rel_feat", "train", folder, frame_file),
        os.path.join(scene_root, "rel_feat", folder, frame_file),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def validate_training_layout(
    json_file: str,
    video_root: str,
    pose_root: str,
    scene_root: str,
    max_samples: int = 0,
) -> Tuple[int, int, int, int]:
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as file:
        records = json.load(file)

    total = 0
    missing_video = 0
    missing_pose = 0
    missing_scene = 0

    for item in records:
        if max_samples > 0 and total >= max_samples:
            break
        path = item.get("path", "")
        if "/" not in path:
            continue

        folder, file_name = path.split("/", 1)
        frame_idx_str = file_name.split(".")[0]
        try:
            frame_idx = int(frame_idx_str)
        except ValueError:
            logger.warning("Skip unsupported filename: %s", file_name)
            continue

        frame_file = f"frame_{frame_idx}.npy"
        video_path = os.path.join(video_root, path)
        pose_path = os.path.join(pose_root, folder, frame_file)
        scene_path = _resolve_scene_feature_file(scene_root, folder, frame_file)

        if not os.path.exists(video_path):
            missing_video += 1
        if not os.path.exists(pose_path):
            missing_pose += 1
        if not os.path.exists(scene_path):
            missing_scene += 1
        total += 1

    logger.info(
        "Validation summary: checked=%s missing_video=%s missing_pose=%s missing_scene=%s",
        total,
        missing_video,
        missing_pose,
        missing_scene,
    )
    return total, missing_video, missing_pose, missing_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hawkeye dataset layout")
    parser.add_argument("--tsl-root", type=str, help="Path to TSL-300 dataset root")
    parser.add_argument("--ucf-root", type=str, help="Path to UCF-Crime dataset root")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory")
    parser.add_argument(
        "--tsl-layout",
        type=str,
        default="auto",
        choices=["auto", "legacy-folder", "flat-train-test"],
        help="TSL-300 source layout. 'flat-train-test' means TSL-300/vid/{train,test}/*.mp4",
    )
    parser.add_argument("--extract-frames", action="store_true", help="Copy videos into Hawkeye layout")
    parser.add_argument("--extract-poses", action="store_true", help="Create pose feature layout and print requirements")
    parser.add_argument("--extract-scenes", action="store_true", help="Create scene feature layout and print requirements")
    parser.add_argument("--generate-json", action="store_true", help="Generate training JSON")
    parser.add_argument("--validate-json", action="store_true", help="Validate JSON/video/pose/scene alignment")
    parser.add_argument("--json-path", type=str, default=None, help="Path to training json to generate/validate")
    parser.add_argument("--validate-max-samples", type=int, default=0, help="Validate first N samples only (0=all)")
    args = parser.parse_args()

    split_train_file = os.path.join(args.output_dir, "split_train.txt")
    split_test_file = os.path.join(args.output_dir, "split_test.txt")
    train_folders = load_split_file(split_train_file)
    test_folders = load_split_file(split_test_file)

    logger.info("Loaded %s train folders and %s test folders", len(train_folders), len(test_folders))

    if args.extract_frames:
        if args.tsl_root:
            logger.info("Organizing TSL-300 videos...")
            inferred_layout = args.tsl_layout
            if inferred_layout == "auto":
                if os.path.isdir(os.path.join(args.tsl_root, "vid", "train")):
                    inferred_layout = "flat-train-test"
                else:
                    inferred_layout = "legacy-folder"

            if inferred_layout == "flat-train-test":
                organize_tsl300_flat_videos(
                    args.tsl_root,
                    os.path.join(args.output_dir, "vid_noaudio_split", "train_new"),
                    train_folders,
                    split_name="train",
                )
                organize_tsl300_flat_videos(
                    args.tsl_root,
                    os.path.join(args.output_dir, "vid_noaudio_split", "test_new"),
                    test_folders,
                    split_name="test",
                )
            else:
                organize_videos(args.tsl_root, os.path.join(args.output_dir, "vid_noaudio_split", "train_new"), train_folders)
                organize_videos(args.tsl_root, os.path.join(args.output_dir, "vid_noaudio_split", "test_new"), test_folders)

        if args.ucf_root:
            logger.info("Organizing UCF-Crime videos...")
            organize_videos(args.ucf_root, os.path.join(args.output_dir, "Ucf", "Ucfcrime_split"), train_folders + test_folders)

    if args.extract_poses:
        prepare_feature_layout(
            output_dir=os.path.join(args.output_dir, "pose_feat", "train"),
            split_folders=train_folders,
            split_name="train",
            feature_name="pose",
            expected_shape="(5, 17, 5)",
            extractor_name="HigherHRNet",
            extractor_url="https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation",
        )
        prepare_feature_layout(
            output_dir=os.path.join(args.output_dir, "pose_feat", "test"),
            split_folders=test_folders,
            split_name="test",
            feature_name="pose",
            expected_shape="(5, 17, 5)",
            extractor_name="HigherHRNet",
            extractor_url="https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation",
        )

    if args.extract_scenes:
        prepare_feature_layout(
            output_dir=os.path.join(args.output_dir, "rel_feat", "train"),
            split_folders=train_folders,
            split_name="train",
            feature_name="scene",
            expected_shape="(5, 353)",
            extractor_name="RelTR",
            extractor_url="https://github.com/yrcong/RelTR",
        )
        prepare_feature_layout(
            output_dir=os.path.join(args.output_dir, "rel_feat", "test"),
            split_folders=test_folders,
            split_name="test",
            feature_name="scene",
            expected_shape="(5, 353)",
            extractor_name="RelTR",
            extractor_url="https://github.com/yrcong/RelTR",
        )

    if args.generate_json:
        logger.info("Generating training JSON...")
        json_path = args.json_path or os.path.join(args.output_dir, "new_train.json")
        generate_train_json(
            video_dir=os.path.join(args.output_dir, "vid_noaudio_split", "train_new"),
            split_folders=train_folders,
            output_file=json_path,
            task="iasdig",
        )

    if args.validate_json:
        json_path = args.json_path or os.path.join(args.output_dir, "new_train.json")
        validate_training_layout(
            json_file=json_path,
            video_root=os.path.join(args.output_dir, "vid_noaudio_split", "train_new"),
            pose_root=os.path.join(args.output_dir, "pose_feat", "train"),
            scene_root=args.output_dir,
            max_samples=args.validate_max_samples,
        )


if __name__ == "__main__":
    main()
