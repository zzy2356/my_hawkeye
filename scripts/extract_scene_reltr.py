#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def add_reltr_path(reltr_root: str):
    if reltr_root not in sys.path:
        sys.path.insert(0, reltr_root)


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


def build_reltr(reltr_root: str, ckpt_path: str, device: str):
    add_reltr_path(reltr_root)
    from models import build_model
    from inference import get_args_parser

    parser = argparse.ArgumentParser(parents=[get_args_parser()], add_help=False)
    args = parser.parse_args([])
    args.resume = ckpt_path
    args.device = device

    model, _, _ = build_model(args)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def reltr_feature_353(frame_bgr, model, transform, device, threshold: float):
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    # rel_logits: [1, num_triplets, 52] where last is no-relation
    # sub/obj logits: [1, num_triplets, 152] where last is no-object
    rel_probs = outputs["rel_logits"].softmax(-1)[0, :, :-1]
    sub_probs = outputs["sub_logits"].softmax(-1)[0, :, :-1]
    obj_probs = outputs["obj_logits"].softmax(-1)[0, :, :-1]

    keep = torch.logical_and(
        rel_probs.max(-1).values > threshold,
        torch.logical_and(
            sub_probs.max(-1).values > threshold,
            obj_probs.max(-1).values > threshold,
        ),
    )

    if keep.sum().item() == 0:
        return np.zeros((353,), dtype=np.float32)

    keep_idx = torch.nonzero(keep, as_tuple=True)[0]
    score = (
        rel_probs[keep_idx].max(-1).values
        * sub_probs[keep_idx].max(-1).values
        * obj_probs[keep_idx].max(-1).values
    )
    best_query = keep_idx[torch.argmax(score)].item()

    # 151 + 151 + 51 = 353
    feat = torch.cat(
        [
            sub_probs[best_query],
            obj_probs[best_query],
            rel_probs[best_query],
        ],
        dim=0,
    )
    return feat.detach().cpu().numpy().astype(np.float32)


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
    parser = argparse.ArgumentParser(description="Extract real scene features with RelTR")
    parser.add_argument("--json-path", type=str, default="dataset/new_train.json")
    parser.add_argument("--video-root", type=str, default="dataset/vid_noaudio_split/train_new")
    parser.add_argument("--output-root", type=str, default="dataset/rel_feat/train")
    parser.add_argument("--reltr-root", type=str, default="RelTR")
    parser.add_argument("--reltr-ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
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

    model, transform = build_reltr(args.reltr_root, args.reltr_ckpt, args.device)

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
                feat = np.zeros((args.num_frames, 353), dtype=np.float32)
                for i, frame in enumerate(frames[: args.num_frames]):
                    feat[i] = reltr_feature_353(frame, model, transform, args.device, args.threshold)
                np.save(out_path, feat)
                done += 1
            except Exception as exc:
                failed += 1
                print(f"[WARN] scene failed: {folder}/{clip_name} -> {exc}")

    print("\n=== Scene Extraction Summary ===")
    print(f"folders={len(folders)}")
    print(f"saved={done}")
    print(f"missing_videos={miss_video}")
    print(f"failed={failed}")


if __name__ == "__main__":
    main()
