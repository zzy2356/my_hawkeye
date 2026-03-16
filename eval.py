import os
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import logging

from llava.constants import DEFAULT_X_TOKEN, X_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_X_token
from llava.model.builder import load_pretrained_model
from llava.train.qwen3vl_data import preprocess_qwen3vl_visual
from llava.utils import disable_torch_init

logging.set_verbosity_error()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

IASDIG_PROMPT = (
    "Please determine if the people in the video or the video itself show negative emotions."
)
UCF_PROMPT = (
    "Please determine whether the video is an anomalistic video that contains one of Abuse, Arrest, "
    "Arson, Assault, Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, "
    "Shoplifting, and Vandalism."
)


def _is_qwen3_vl_model_name(model_name_or_path: Optional[str]) -> bool:
    normalized = (model_name_or_path or "").lower().replace("-", "").replace("_", "")
    return "qwen3" in normalized and "vl" in normalized


def _pick_existing_path(*candidates: str) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _model_device(model) -> torch.device:
    return getattr(model, "device", next(model.parameters()).device)


def _resolve_scene_feature_path(base_dir: str, folder: str, filename: str) -> str:
    resolved = _pick_existing_path(
        os.path.join(base_dir, "graph_feat", folder, filename),
        os.path.join(base_dir, "rel_feat", folder, filename),
    )
    if resolved is None:
        raise FileNotFoundError(f"Scene feature not found for {folder}/{filename} under {base_dir}.")
    return resolved


def _load_temporal_feature(path: str, target_frames: int, feature_shape: Iterable[int]) -> torch.Tensor:
    feature = torch.from_numpy(np.load(path)).float()
    feature = feature[:target_frames]
    if feature.size(0) < target_frames:
        pad_shape = (target_frames - feature.size(0), *feature_shape)
        feature = torch.cat((feature, torch.zeros(pad_shape, dtype=feature.dtype)), dim=0)
    return feature


def _sorted_videos(folder_path: str):
    def _video_sort_key(filename: str):
        try:
            return int(filename.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return filename

    return sorted(
        [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))],
        key=_video_sort_key,
    )


def _run_qwen3_vl_inference(
    model,
    processor,
    video_path: str,
    prompt: str,
    pose_feature: torch.Tensor,
    scene_feature: torch.Tensor,
    max_new_tokens: int,
) -> str:
    conversations = [{"from": "human", "value": f"{prompt}\n<video>"}]
    model_inputs = preprocess_qwen3vl_visual(
        conversations=conversations,
        processor=processor,
        media_path=video_path,
        media_type="video",
        add_generation_prompt=True,
        include_labels=False,
    )
    input_len = model_inputs["input_ids"].shape[1]

    for key, value in list(model_inputs.items()):
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(_model_device(model))

    pose_values = pose_feature.unsqueeze(0).to(_model_device(model))
    scene_values = scene_feature.unsqueeze(0).to(_model_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            pose_values=pose_values,
            scene_values=scene_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Use the true prefix length stored by Qwen3VLHawkeyeAdapter.generate.
    # _splice_hawkeye_tokens inserts MoE tokens into the sequence, making the
    # real prefix longer than the original input_ids.  Using input_len here
    # would leave MoE dummy tokens in the decoded output (visible as garbage
    # characters like "!!!" before the answer).
    true_prefix_lens = getattr(model, "last_prefix_lens", None) or [getattr(model, "last_prefix_len", None) or input_len]
    trimmed_ids = [output_ids[prefix_len:] for output_ids, prefix_len in zip(generated_ids, true_prefix_lens)]
    outputs = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs[0].strip() if outputs else ""


def _run_legacy_hawkeye_inference(
    model,
    tokenizer,
    video_processor,
    video_path: str,
    prompt: str,
    pose_feature: torch.Tensor,
    scene_feature: torch.Tensor,
    max_new_tokens: int,
) -> str:
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_X_TOKEN["VIDEO"] + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_X_token(
        prompt_text,
        tokenizer,
        X_TOKEN_INDEX["VIDEO"],
        return_tensors="pt",
    ).unsqueeze(0).to(_model_device(model))

    video_tensor = video_processor(video_path, return_tensors="pt")["pixel_values"]
    if isinstance(video_tensor, list):
        video_tensor = [video.to(_model_device(model), dtype=torch.float16) for video in video_tensor]
    else:
        video_tensor = video_tensor.to(_model_device(model), dtype=torch.float16)

    pose_values = pose_feature.to(_model_device(model), dtype=torch.float16)
    scene_values = scene_feature.to(_model_device(model), dtype=torch.float16)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor, [pose_values], [scene_values], ["video"]],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def _evaluate_folder_dataset(
    root_dir: str,
    pose_root: str,
    scene_root: str,
    save_root: str,
    prompt: str,
    tokenizer,
    model,
    processor_dict,
    is_qwen3_vl: bool,
    max_new_tokens: int,
    skip_if_contains: Optional[str] = None,
) -> None:
    if not os.path.isdir(root_dir):
        print(f"Skip missing dataset root: {root_dir}")
        return

    os.makedirs(save_root, exist_ok=True)
    qwen_processor = processor_dict.get("qwen")
    video_processor = processor_dict.get("video")

    for folder_name in tqdm(sorted(os.listdir(root_dir))):
        if skip_if_contains and skip_if_contains in folder_name:
            continue

        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        outputs = []
        filenames = []
        for video_name in tqdm(_sorted_videos(folder_path), leave=False):
            video_path = os.path.join(folder_path, video_name)
            filenames.append(video_name)

            frame_file = f"frame_{int(video_name.split('.')[0])}.npy"
            pose_path = os.path.join(pose_root, folder_name, frame_file)
            scene_path = _resolve_scene_feature_path(scene_root, folder_name, frame_file)

            pose_feature = _load_temporal_feature(pose_path, target_frames=5, feature_shape=(17, 5))
            scene_feature = _load_temporal_feature(scene_path, target_frames=5, feature_shape=(353,))

            if is_qwen3_vl:
                if qwen_processor is None:
                    raise ValueError("Qwen3-VL inference requires processor['qwen'].")
                result = _run_qwen3_vl_inference(
                    model=model,
                    processor=qwen_processor,
                    video_path=video_path,
                    prompt=prompt,
                    pose_feature=pose_feature,
                    scene_feature=scene_feature,
                    max_new_tokens=max_new_tokens,
                )
            else:
                if video_processor is None:
                    raise ValueError("Legacy Hawkeye inference requires processor['video'].")
                result = _run_legacy_hawkeye_inference(
                    model=model,
                    tokenizer=tokenizer,
                    video_processor=video_processor,
                    video_path=video_path,
                    prompt=prompt,
                    pose_feature=pose_feature,
                    scene_feature=scene_feature,
                    max_new_tokens=max_new_tokens,
                )

            outputs.append(result)

        pd.DataFrame({"file": filenames, "output": outputs}).to_csv(
            os.path.join(save_root, f"{folder_name}.csv"),
            index=False,
        )


def main():
    disable_torch_init()

    model_path = os.environ.get("HAWKEYE_MODEL_PATH", "models/Qwen3-VL-8B-Instruct")
    model_base = os.environ.get("HAWKEYE_MODEL_BASE") or None
    device = os.environ.get("HAWKEYE_DEVICE", "cuda")
    load_4bit = os.environ.get("HAWKEYE_LOAD_4BIT", "0") == "1"
    load_8bit = os.environ.get("HAWKEYE_LOAD_8BIT", "0") == "1"
    model_name = get_model_name_from_path(model_base or model_path)

    tokenizer, model, processor_dict, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device,
    )

    is_qwen3_vl = _is_qwen3_vl_model_name(model_path) or _is_qwen3_vl_model_name(model_base)

    iasdig_root = _pick_existing_path("dataset/vid_split/test_new", "dataset/vid_noaudio_split/test_new")
    if iasdig_root is not None:
        _evaluate_folder_dataset(
            root_dir=iasdig_root,
            pose_root="dataset/pose_feat/test",
            scene_root="dataset",
            save_root="dataset/saved_result/test_res",
            prompt=IASDIG_PROMPT,
            tokenizer=tokenizer,
            model=model,
            processor_dict=processor_dict,
            is_qwen3_vl=is_qwen3_vl,
            max_new_tokens=16,
        )

    _evaluate_folder_dataset(
        root_dir="dataset/Ucf/Ucfcrime_split",
        pose_root="dataset/Ucf/pose_feat",
        scene_root="dataset/Ucf",
        save_root="dataset/saved_result/test_res",
        prompt=UCF_PROMPT,
        tokenizer=tokenizer,
        model=model,
        processor_dict=processor_dict,
        is_qwen3_vl=is_qwen3_vl,
        max_new_tokens=32,
        skip_if_contains="Normal",
    )


if __name__ == "__main__":
    main()
