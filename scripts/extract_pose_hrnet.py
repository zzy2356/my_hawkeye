#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import types
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


def is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg


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

    def _collect_person_arrays(final_results_obj):
        persons_list = []

        def _append_candidate(candidate):
            arr = np.asarray(candidate, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                persons_list.append(arr)
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                for i in range(arr.shape[0]):
                    persons_list.append(arr[i])

        if isinstance(final_results_obj, np.ndarray):
            _append_candidate(final_results_obj)
            return persons_list

        if isinstance(final_results_obj, (list, tuple)):
            for item in final_results_obj:
                _append_candidate(item)
            return persons_list

        _append_candidate(final_results_obj)
        return persons_list

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

    persons = _collect_person_arrays(final_results)
    if len(persons) == 0:
        return pose_17x5

    # Pick one person with highest average confidence.
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


def infer_pose_batch_17x5(
    frames_bgr,
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
    if not frames_bgr:
        return []

    rgb_images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]

    base_sizes = []
    centers = []
    scales = []
    for image in rgb_images:
        base_size, center, scale = get_multi_scale_size(
            image,
            cfg.DATASET.INPUT_SIZE,
            1.0,
            min(cfg.TEST.SCALE_FACTOR),
        )
        base_sizes.append(base_size)
        centers.append(center)
        scales.append(scale)

    height_width = [frame.shape[:2] for frame in frames_bgr]

    with torch.no_grad():
        final_heatmaps = None
        tags_list = []
        for s in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            resized_tensors = []
            for image in rgb_images:
                image_resized, _, _ = resize_align_multi_scale(
                    image,
                    cfg.DATASET.INPUT_SIZE,
                    s,
                    min(cfg.TEST.SCALE_FACTOR),
                )
                resized_tensors.append(image_transform(image_resized))

            image_tensor = torch.stack(resized_tensors, dim=0).to(device)
            outputs, heatmaps, tags = get_multi_stage_outputs(
                cfg,
                model,
                image_tensor,
                cfg.TEST.FLIP_TEST,
                cfg.TEST.PROJECT2IMAGE,
                base_sizes[0],
            )
            final_heatmaps, tags_list = aggregate_results(cfg, s, final_heatmaps, tags_list, heatmaps, tags)

        final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
        tags = torch.cat(tags_list, dim=4)

    feats = []
    for idx in range(len(frames_bgr)):
        fh_i = final_heatmaps[idx : idx + 1]
        tag_i = tags[idx : idx + 1]
        grouped, scores = parser.parse(fh_i, tag_i, cfg.TEST.ADJUST, cfg.TEST.REFINE)
        final_results = get_final_preds(
            grouped,
            centers[idx],
            scales[idx],
            [fh_i.size(3), fh_i.size(2)],
        )

        height, width = height_width[idx]
        pose_17x5 = np.zeros((17, 5), dtype=np.float32)

        persons = []
        if isinstance(final_results, np.ndarray):
            if final_results.ndim == 2 and final_results.shape[1] >= 3:
                persons = [final_results]
            elif final_results.ndim == 3 and final_results.shape[2] >= 3:
                persons = [final_results[i] for i in range(final_results.shape[0])]
        elif isinstance(final_results, (list, tuple)):
            for item in final_results:
                arr = np.asarray(item, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    persons.append(arr)
                elif arr.ndim == 3 and arr.shape[2] >= 3:
                    persons.extend([arr[i] for i in range(arr.shape[0])])

        if persons:
            best = max(persons, key=lambda p: float(np.mean(p[:, 2])) if p.shape[1] >= 3 else -1.0)
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

        feats.append(pose_17x5)

    return feats


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
    parser.add_argument("--batch-size", type=int, default=24, help="Clip batch size for inference (process up to N clips together)")
    parser.add_argument("--max-folders", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0, help="Process first N clips per folder, 0=all")
    parser.add_argument("--folders", type=str, default="", help="Comma-separated folder names to process")
    parser.add_argument("--resume-state", type=str, default="", help="Path to resume state JSON (default: <output-root>/_resume_pose.json)")
    parser.add_argument("--reset-resume", action="store_true", help="Ignore existing resume state and start from beginning")
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

    resume_state_path = args.resume_state.strip() or os.path.join(args.output_root, "_resume_pose.json")
    start_task_index = 0
    if not args.reset_resume and os.path.exists(resume_state_path):
        try:
            with open(resume_state_path, "r", encoding="utf-8") as file:
                state = json.load(file)
            start_task_index = int(state.get("next_task_index", 0))
            print(f"[INFO] Resume enabled: next_task_index={start_task_index} from {resume_state_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load resume state {resume_state_path}: {exc}")

    all_tasks = []
    for folder in folders:
        clip_items = sorted(mapping[folder], key=lambda x: x[1])
        if args.max_clips > 0:
            clip_items = clip_items[: args.max_clips]
        for clip_name, frame_id in clip_items:
            all_tasks.append((folder, clip_name, frame_id))

    def save_resume_state(next_task_index, folder, clip_name, frame_id):
        os.makedirs(os.path.dirname(resume_state_path), exist_ok=True)
        payload = {
            "next_task_index": next_task_index,
            "folder": folder,
            "clip_name": clip_name,
            "frame_id": frame_id,
        }
        last_error = None
        for attempt in range(5):
            tmp_path = resume_state_path + f".tmp.{os.getpid()}.{attempt}"
            try:
                with open(tmp_path, "w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False)
                try:
                    os.replace(tmp_path, resume_state_path)
                except PermissionError:
                    if os.path.exists(resume_state_path):
                        try:
                            os.remove(resume_state_path)
                        except OSError:
                            pass
                    os.replace(tmp_path, resume_state_path)
                return
            except Exception as exc:
                last_error = exc
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                time.sleep(0.1)

        raise RuntimeError(f"Failed to write resume state {resume_state_path}: {last_error}")

    done = 0
    miss_video = 0
    failed = 0
    skipped_existing = 0

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    total_tasks = len(all_tasks)
    task_index = start_task_index
    while task_index < total_tasks:
        batch_tasks = all_tasks[task_index : task_index + args.batch_size]
        batch_entries = []

        for offset, (folder, clip_name, frame_id) in enumerate(batch_tasks):
            global_index = task_index + offset
            out_dir = os.path.join(args.output_root, folder)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"frame_{frame_id}.npy")

            if os.path.exists(out_path):
                skipped_existing += 1
                save_resume_state(global_index + 1, folder, clip_name, frame_id)
                continue

            video_path = os.path.join(args.video_root, folder, clip_name)
            if not os.path.exists(video_path):
                miss_video += 1
                save_resume_state(global_index + 1, folder, clip_name, frame_id)
                continue

            try:
                frames = sample_frames(video_path, args.num_frames)
                feat = np.zeros((args.num_frames, 17, 5), dtype=np.float32)
                batch_entries.append(
                    {
                        "global_index": global_index,
                        "folder": folder,
                        "clip_name": clip_name,
                        "frame_id": frame_id,
                        "frames": frames,
                        "feat": feat,
                        "out_path": out_path,
                    }
                )
            except Exception as exc:
                failed += 1
                print(f"[WARN] pose failed (decode): {folder}/{clip_name} -> {exc}")

        if batch_entries:
            for frame_pos in range(args.num_frames):
                frame_batch = []
                owner_indices = []
                for entry_idx, entry in enumerate(batch_entries):
                    if frame_pos < len(entry["frames"]):
                        frame_batch.append(entry["frames"][frame_pos])
                        owner_indices.append(entry_idx)

                if not frame_batch:
                    continue

                start = 0
                current_bs = min(args.batch_size, len(frame_batch))
                while start < len(frame_batch):
                    end = min(start + current_bs, len(frame_batch))
                    sub_frames = frame_batch[start:end]
                    sub_owner_indices = owner_indices[start:end]
                    try:
                        batch_feats = infer_pose_batch_17x5(
                            sub_frames,
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
                        for bf, entry_idx in zip(batch_feats, sub_owner_indices):
                            batch_entries[entry_idx]["feat"][frame_pos] = bf
                        start = end
                    except Exception as exc:
                        if is_cuda_oom_error(exc) and current_bs > 1:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            next_bs = max(1, current_bs // 2)
                            print(
                                f"[WARN] OOM at frame_pos={frame_pos}, reduce batch {current_bs}->{next_bs}"
                            )
                            current_bs = next_bs
                            continue

                        # Fallback to single-frame inference so one bad frame does not kill the whole batch.
                        print(
                            f"[WARN] batched pose frame failed at pos={frame_pos}, "
                            f"fallback to single inference: {exc}"
                        )
                        for entry_idx in sub_owner_indices:
                            frame = batch_entries[entry_idx]["frames"][frame_pos]
                            try:
                                batch_entries[entry_idx]["feat"][frame_pos] = infer_pose_17x5(
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
                            except Exception as inner_exc:
                                print(
                                    f"[WARN] single pose fallback failed: "
                                    f"{batch_entries[entry_idx]['folder']}/{batch_entries[entry_idx]['clip_name']} -> {inner_exc}"
                                )
                        start = end

            for entry in batch_entries:
                try:
                    np.save(entry["out_path"], entry["feat"])
                    done += 1
                except Exception as exc:
                    failed += 1
                    print(f"[WARN] pose save failed: {entry['folder']}/{entry['clip_name']} -> {exc}")
                finally:
                    save_resume_state(entry["global_index"] + 1, entry["folder"], entry["clip_name"], entry["frame_id"])

        task_index += args.batch_size
        if (done + skipped_existing + miss_video + failed) % 200 == 0:
            print(
                f"[INFO] progress {min(task_index, total_tasks)}/{total_tasks} "
                f"saved={done} skipped_existing={skipped_existing} missing_videos={miss_video} failed={failed}"
            )

    print("\n=== Pose Extraction Summary ===")
    print(f"folders={len(folders)}")
    print(f"total_tasks={total_tasks}")
    print(f"saved={done}")
    print(f"skipped_existing={skipped_existing}")
    print(f"missing_videos={miss_video}")
    print(f"failed={failed}")
    print(f"resume_state={resume_state_path}")


if __name__ == "__main__":
    main()
