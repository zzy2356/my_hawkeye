import argparse
import json
import os
import sys

# Ensure the project root (two levels up from scripts/qwen3vl/) is on sys.path
# so that the `llava` package can be imported regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.train.qwen3vl_data import preprocess_qwen3vl_visual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL smoke inference for Hawkeye pipeline")
    parser.add_argument("--model-path", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-base", default=None)
    parser.add_argument(
        "--video-path",
        default="Qwen3-VL/qwen-vl-finetune/demo/videos/v_7bUu05RIksU.mp4",
    )
    parser.add_argument(
        "--prompt",
        default="Please determine whether this video shows abnormal or negative events.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--pose-npy", default=None)
    parser.add_argument("--scene-npy", default=None)
    parser.add_argument("--print-shapes", action="store_true")
    return parser


def _load_feature(path: str, target_frames: int, feature_shape) -> torch.Tensor:
    feature = torch.from_numpy(np.load(path)).float()
    feature = feature[:target_frames]
    if feature.size(0) < target_frames:
        feature = torch.cat((feature, torch.zeros((target_frames - feature.size(0), *feature_shape))), dim=0)
    return feature


def _model_device(model) -> torch.device:
    return getattr(model, "device", next(model.parameters()).device)


def _device_map(model):
    base_model = getattr(model, "model", model)
    return getattr(base_model, "hf_device_map", None)


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video path not found: {args.video_path}")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor_dict, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device="cuda",
    )

    qwen_processor = processor_dict.get("qwen", None)
    if qwen_processor is None:
        raise RuntimeError("Qwen processor not found. Check transformers version and model path.")

    inputs = preprocess_qwen3vl_visual(
        conversations=[{"from": "human", "value": f"{args.prompt}\n<video>"}],
        processor=qwen_processor,
        media_path=args.video_path,
        media_type="video",
        add_generation_prompt=True,
        include_labels=False,
    )
    input_len = inputs["input_ids"].shape[1]

    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(_model_device(model))

    if args.pose_npy is not None:
        pose_values = _load_feature(args.pose_npy, target_frames=5, feature_shape=(17, 5))
    else:
        pose_values = torch.zeros((5, 17, 5), dtype=torch.float32)

    if args.scene_npy is not None:
        scene_values = _load_feature(args.scene_npy, target_frames=5, feature_shape=(353,))
    else:
        scene_values = torch.zeros((5, 353), dtype=torch.float32)

    if args.print_shapes:
        print("=== Hawkeye Multimodal Shapes ===")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"model_device: {_model_device(model)}")
        print(f"hf_device_map: {_device_map(model)}")
        print(f"input_ids: {tuple(inputs['input_ids'].shape)}")
        if "pixel_values_videos" in inputs:
            print(f"pixel_values_videos: {tuple(inputs['pixel_values_videos'].shape)}")
        if "video_grid_thw" in inputs:
            print(f"video_grid_thw: {tuple(inputs['video_grid_thw'].shape)}")
        print(f"pose_values: {tuple(pose_values.shape)}")
        print(f"scene_values: {tuple(scene_values.shape)}")
        print(f"hawkeye_scene_token_count: {getattr(model.config, 'hawkeye_scene_token_count', 'unknown')}")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            pose_values=pose_values.unsqueeze(0).to(_model_device(model)),
            scene_values=scene_values.unsqueeze(0).to(_model_device(model)),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    # Use the true prefix length stored by Qwen3VLHawkeyeAdapter.generate.
    # After _splice_hawkeye_tokens the sequence is longer than the original
    # input_ids (MoE tokens were inserted), so we must NOT use input_len here.
    true_prefix_lens = getattr(model, "last_prefix_lens", None) or [getattr(model, "last_prefix_len", None) or input_len]
    trimmed = [out[prefix_len:] for out, prefix_len in zip(output_ids, true_prefix_lens)]
    text = qwen_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    if args.print_shapes:
        print("=== Hawkeye Debug Info ===")
        print(f"true_prefix_lens: {true_prefix_lens}")
        debug_info = getattr(model, "last_debug_info", {})
        print(json.dumps(debug_info, ensure_ascii=False, indent=2))

    print("=== Smoke Inference Output ===")
    print(text[0] if text else "")


if __name__ == "__main__":
    main()
