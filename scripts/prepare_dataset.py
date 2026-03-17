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
from typing import List


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hawkeye dataset layout")
    parser.add_argument("--tsl-root", type=str, help="Path to TSL-300 dataset root")
    parser.add_argument("--ucf-root", type=str, help="Path to UCF-Crime dataset root")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--extract-frames", action="store_true", help="Copy videos into Hawkeye layout")
    parser.add_argument("--extract-poses", action="store_true", help="Create pose feature layout and print requirements")
    parser.add_argument("--extract-scenes", action="store_true", help="Create scene feature layout and print requirements")
    parser.add_argument("--generate-json", action="store_true", help="Generate training JSON")
    args = parser.parse_args()

    split_train_file = os.path.join(args.output_dir, "split_train.txt")
    split_test_file = os.path.join(args.output_dir, "split_test.txt")
    train_folders = load_split_file(split_train_file)
    test_folders = load_split_file(split_test_file)

    logger.info("Loaded %s train folders and %s test folders", len(train_folders), len(test_folders))

    if args.extract_frames:
        if args.tsl_root:
            logger.info("Organizing TSL-300 videos...")
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
        generate_train_json(
            video_dir=os.path.join(args.output_dir, "vid_noaudio_split", "train_new"),
            split_folders=train_folders,
            output_file=os.path.join(args.output_dir, "new_train.json"),
            task="iasdig",
        )


if __name__ == "__main__":
    main()
