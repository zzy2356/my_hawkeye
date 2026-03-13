import argparse
import os

import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL smoke inference for Hawkeye pipeline")
    parser.add_argument("--model-path", default="models/Qwen3-VL-8B-Instruct")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video path not found: {args.video_path}")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor_dict, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device="cuda",
    )

    qwen_processor = processor_dict.get("qwen", None)
    if qwen_processor is None:
        raise RuntimeError("Qwen processor not found. Check transformers version and model path.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video_path},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = qwen_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(model.device)

    pose_values = torch.zeros((1, 5, 17, 5), dtype=torch.float16, device=model.device)
    scene_values = torch.zeros((1, 5, 353), dtype=torch.float16, device=model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            pose_values=pose_values,
            scene_values=scene_values,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
    text = qwen_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print("=== Smoke Inference Output ===")
    print(text[0] if text else "")


if __name__ == "__main__":
    main()
