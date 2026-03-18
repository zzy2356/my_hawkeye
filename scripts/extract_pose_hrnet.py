#!/usr/bin/env python3
import argparse
import json
import os
import sys
import types
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


def add_hrnet_path(hrnet_root: str):
    lib_path = os.path.join(hrnet_root, "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)


def install_hrnet_dataset_shim():
    """
    HigherHRNet core.inference imports dataset.transforms.FLIP_CONFIG.
    The original dataset package imports crowdposetools eagerly, which may be
    unavailable in inference-only usage. Provide a minimal shim for inference.
    """
    flip_config = {
        "COCO": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        "COCO_WITH_CENTER": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17],
        "CROWDPOSE": [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13],
        "CROWDPOSE_WITH_CENTER": [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14],
    }

    dataset_module = types.ModuleType("dataset")
    transforms_module = types.ModuleType("dataset.transforms")
    transforms_module.FLIP_CONFIG = flip_config
    dataset_module.transforms = transforms_module
    sys.modules["dataset"] = dataset_module
    sys.modules["dataset.transforms"] = transforms_module


def load_samples(json_path: str):
    with open(json_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    mapping = defaultdict(set)
    for item in records:
        rel = item.get("path", "")
        if "/" not in rel:
            continue
        folder, clip_name = rel.split("/", 1)
        try:
            frame_id = int(clip_name.split(".")[0])
        except ValueError:
            continue
        mapping[folder].add((clip_name, frame_id))
    return mapping


def build_hrnet(cfg_path: str, ckpt_path: str, hrnet_root: str, device: str):
    add_hrnet_path(hrnet_root)
    install_hrnet_dataset_shim()
    from config import cfg, update_config, check_config
    from core.group import HeatmapParser
    from core.inference import get_multi_stage_outputs, aggregate_results
    from utils.transforms import get_final_preds, get_multi_scale_size, resize_align_multi_scale
    import models

    class _Args:
        def __init__(self):
            self.cfg = cfg_path
            self.opts = ["TEST.MODEL_FILE", ckpt_path]

    args = _Args()
    update_config(cfg, args)
    check_config(cfg)

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    parser = HeatmapParser(cfg)
    return cfg, model, parser, image_transform, get_final_preds, get_multi_scale_size, resize_align_multi_scale, get_multi_stage_outputs, aggregate_results


def infer_pose_17x5(
    frame_bgr,
    cfg,
    model,
    parser,
    image_transform,
    get_final_preds,
    get_multi_scale_size,
    resize_align_multi_scale,
    get_multi_stage_outputs,
    aggregate_results,
    device,
):
    def _as_kpt_array(person):
        arr = np.asarray(person, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                return None
            arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return None
        return arr

    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base_size, center, scale = get_multi_scale_size(
        image,
        cfg.DATASET.INPUT_SIZE,
        1.0,
        min(cfg.TEST.SCALE_FACTOR),
    )

    with torch.no_grad():
        final_heatmaps = None
        tags_list = []
        for s in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale = resize_align_multi_scale(
                image,
                cfg.DATASET.INPUT_SIZE,
                s,
                min(cfg.TEST.SCALE_FACTOR),
            )
            image_tensor = image_transform(image_resized).unsqueeze(0).to(device)

            outputs, heatmaps, tags = get_multi_stage_outputs(
                cfg,
                model,
                image_tensor,
                cfg.TEST.FLIP_TEST,
                cfg.TEST.PROJECT2IMAGE,
                base_size,
            )
            final_heatmaps, tags_list = aggregate_results(cfg, s, final_heatmaps, tags_list, heatmaps, tags)

        final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
        tags = torch.cat(tags_list, dim=4)
        grouped, scores = parser.parse(final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE)
        final_results = get_final_preds(
            grouped,
            center,
            scale,
            [final_heatmaps.size(3), final_heatmaps.size(2)],
        )

    height, width = frame_bgr.shape[:2]
    pose_17x5 = np.zeros((17, 5), dtype=np.float32)

    if len(final_results) == 0 or len(final_results[0]) == 0:
        return pose_17x5

    # Pick one person with highest average confidence.
    persons = final_results[0]
    best_idx = -1
    best_score = -1.0
    for idx, person in enumerate(persons):
        arr = _as_kpt_array(person)
        if arr is None:
            continue
        score = float(np.mean(arr[:, 2]))
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx < 0:
        return pose_17x5
    best = _as_kpt_array(persons[best_idx])
    if best is None:
        return pose_17x5
    if best.shape[0] < 17:
        padded = np.zeros((17, 3), dtype=np.float32)
        padded[: best.shape[0], :3] = best[:, :3]
        best = padded
    elif best.shape[0] > 17:
        best = best[:17, :3]

    pose_17x5[:, 0] = best[:, 0]
    pose_17x5[:, 1] = best[:, 1]
    pose_17x5[:, 2] = best[:, 2]
    pose_17x5[:, 3] = np.clip(best[:, 0] / max(width, 1), 0.0, 1.0)
    pose_17x5[:, 4] = np.clip(best[:, 1] / max(height, 1), 0.0, 1.0)
    return pose_17x5


def sample_frames(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Extract real pose features with HigherHRNet")
    parser.add_argument("--json-path", type=str, default="dataset/new_train.json")
    parser.add_argument("--video-root", type=str, default="dataset/vid_noaudio_split/train_new")
    parser.add_argument("--output-root", type=str, default="dataset/pose_feat/train")
    parser.add_argument("--hrnet-root", type=str, default="HigherHRNet-Human-Pose-Estimation")
    parser.add_argument("--hrnet-cfg", type=str, default="HigherHRNet-Human-Pose-Estimation/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml")
    parser.add_argument("--hrnet-ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--max-folders", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0, help="Process first N clips per folder, 0=all")
    parser.add_argument("--folders", type=str, default="", help="Comma-separated folder names to process")
    args = parser.parse_args()

    mapping = load_samples(args.json_path)
    folders = sorted(mapping.keys())
    if args.folders.strip():
        requested = [name.strip() for name in args.folders.split(",") if name.strip()]
        folders = [name for name in requested if name in mapping]
    if args.max_folders > 0:
        folders = folders[: args.max_folders]

    built = build_hrnet(args.hrnet_cfg, args.hrnet_ckpt, args.hrnet_root, args.device)
    (
        cfg,
        model,
        parser_inst,
        image_transform,
        get_final_preds,
        get_multi_scale_size,
        resize_align_multi_scale,
        get_multi_stage_outputs,
        aggregate_results,
    ) = built

    done = 0
    miss_video = 0
    failed = 0

    for folder in folders:
        clip_items = sorted(mapping[folder], key=lambda x: x[1])
        if args.max_clips > 0:
            clip_items = clip_items[: args.max_clips]
        for clip_name, frame_id in clip_items:
            out_dir = os.path.join(args.output_root, folder)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"frame_{frame_id}.npy")
            if os.path.exists(out_path):
                continue

            video_path = os.path.join(args.video_root, folder, clip_name)
            if not os.path.exists(video_path):
                miss_video += 1
                continue

            try:
                frames = sample_frames(video_path, args.num_frames)
                feat = np.zeros((args.num_frames, 17, 5), dtype=np.float32)
                for i, frame in enumerate(frames[: args.num_frames]):
                    feat[i] = infer_pose_17x5(
                        frame,
                        cfg,
                        model,
                        parser_inst,
                        image_transform,
                        get_final_preds,
                        get_multi_scale_size,
                        resize_align_multi_scale,
                        get_multi_stage_outputs,
                        aggregate_results,
                        args.device,
                    )
                np.save(out_path, feat)
                done += 1
            except Exception as exc:
                failed += 1
                print(f"[WARN] pose failed: {folder}/{clip_name} -> {exc}")

    print("\n=== Pose Extraction Summary ===")
    print(f"folders={len(folders)}")
    print(f"saved={done}")
    print(f"missing_videos={miss_video}")
    print(f"failed={failed}")


if __name__ == "__main__":
    main()
