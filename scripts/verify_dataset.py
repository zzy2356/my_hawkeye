#!/usr/bin/env python3
"""
Verify Hawkeye dataset completeness and layout.

Usage:
    python scripts/verify_dataset.py [--dataset-root dataset]
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def pick_existing_dir(*candidates: str) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def load_split_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def check_videos(video_root: str, split_folders: List[str]) -> Tuple[int, List[str]]:
    issues: List[str] = []
    total_videos = 0

    for folder in split_folders:
        folder_path = os.path.join(video_root, folder)
        if not os.path.isdir(folder_path):
            issues.append(f"Missing video folder: {folder_path}")
            continue

        videos = [name for name in os.listdir(folder_path) if name.endswith((".mp4", ".avi", ".mov"))]
        if not videos:
            issues.append(f"No videos in: {folder_path}")
            continue
        total_videos += len(videos)

    return total_videos, issues


def check_features(
    feat_root: str,
    split_folders: List[str],
    expected_shape: Tuple[int, ...],
    feat_name: str,
) -> Tuple[int, List[str]]:
    issues: List[str] = []
    total_features = 0

    for folder in split_folders:
        folder_path = os.path.join(feat_root, folder)
        if not os.path.isdir(folder_path):
            issues.append(f"Missing {feat_name} folder: {folder_path}")
            continue

        features = [name for name in os.listdir(folder_path) if name.endswith(".npy")]
        if not features:
            issues.append(f"No {feat_name} files in: {folder_path}")
            continue

        total_features += len(features)
        sample_path = os.path.join(folder_path, features[0])
        try:
            sample = np.load(sample_path)
        except Exception as exc:
            issues.append(f"Error loading {sample_path}: {exc}")
            continue

        if tuple(sample.shape) != expected_shape:
            issues.append(
                f"Wrong {feat_name} shape in {folder}/{features[0]}: {tuple(sample.shape)}, expected {expected_shape}"
            )

    return total_features, issues


def check_json(json_path: str, video_root: str) -> Tuple[int, List[str]]:
    issues: List[str] = []

    if not os.path.exists(json_path):
        return 0, [f"Missing JSON file: {json_path}"]

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        return 0, [f"Error loading JSON: {exc}"]

    if not isinstance(data, list):
        return 0, [f"JSON root should be list, got {type(data)}"]

    required_keys = ["path", "label", "mode", "conversations"]
    for idx, ann in enumerate(data):
        if not isinstance(ann, dict):
            issues.append(f"Annotation {idx} is not dict")
            continue

        for key in required_keys:
            if key not in ann:
                issues.append(f"Annotation {idx} missing key: {key}")

        rel_path = ann.get("path")
        if rel_path:
            video_path = os.path.join(video_root, rel_path)
            if not os.path.exists(video_path):
                issues.append(f"Video not found for annotation {idx}: {video_path}")

    return len(data), issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Hawkeye dataset")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Dataset root directory")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    train_folders = load_split_file(os.path.join(dataset_root, "split_train.txt"))
    test_folders = load_split_file(os.path.join(dataset_root, "split_test.txt"))
    logger.info("Loaded %s train folders and %s test folders", len(train_folders), len(test_folders))

    video_train_root = pick_existing_dir(
        os.path.join(dataset_root, "vid_noaudio_split", "train_new"),
        os.path.join(dataset_root, "vid_split", "train_new"),
        os.path.join(dataset_root, "vid_split"),
    )
    if video_train_root is None:
        logger.error("Unable to find IASDig train video root under %s", dataset_root)
        sys.exit(1)

    pose_train_root = os.path.join(dataset_root, "pose_feat", "train")
    scene_train_root = pick_existing_dir(
        os.path.join(dataset_root, "graph_feat", "train"),
        os.path.join(dataset_root, "rel_feat", "train"),
    )
    if scene_train_root is None:
        logger.error("Unable to find train scene root under %s", dataset_root)
        sys.exit(1)

    all_issues: List[str] = []

    logger.info("Checking train videos...")
    video_count, issues = check_videos(video_train_root, train_folders)
    all_issues.extend(issues)

    logger.info("Checking train pose features...")
    pose_count, issues = check_features(pose_train_root, train_folders, (5, 17, 5), "pose")
    all_issues.extend(issues)

    logger.info("Checking train scene features...")
    scene_count, issues = check_features(scene_train_root, train_folders, (5, 353), "scene")
    all_issues.extend(issues)

    logger.info("Checking training JSON...")
    json_count, issues = check_json(os.path.join(dataset_root, "new_train.json"), video_train_root)
    all_issues.extend(issues)

    if test_folders:
        test_video_root = pick_existing_dir(
            os.path.join(dataset_root, "vid_noaudio_split", "test_new"),
            os.path.join(dataset_root, "vid_split", "test_new"),
        )
        test_pose_root = os.path.join(dataset_root, "pose_feat", "test")
        test_scene_root = pick_existing_dir(
            os.path.join(dataset_root, "graph_feat", "test"),
            os.path.join(dataset_root, "rel_feat", "test"),
        )
        if test_video_root is not None:
            logger.info("Checking test videos...")
            _, issues = check_videos(test_video_root, test_folders)
            all_issues.extend(issues)
        if os.path.isdir(test_pose_root):
            logger.info("Checking test pose features...")
            _, issues = check_features(test_pose_root, test_folders, (5, 17, 5), "pose(test)")
            all_issues.extend(issues)
        if test_scene_root is not None:
            logger.info("Checking test scene features...")
            _, issues = check_features(test_scene_root, test_folders, (5, 353), "scene(test)")
            all_issues.extend(issues)

    logger.info("=" * 60)
    if all_issues:
        logger.error("Found %s dataset issues:", len(all_issues))
        for issue in all_issues:
            logger.error("  - %s", issue)
        sys.exit(1)

    logger.info("Dataset verification passed")
    logger.info("Train videos: %s", video_count)
    logger.info("Train pose features: %s", pose_count)
    logger.info("Train scene features: %s", scene_count)
    logger.info("Training annotations: %s", json_count)
    sys.exit(0)


if __name__ == "__main__":
    main()
