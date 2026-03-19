import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from llava.constants import IGNORE_INDEX


# Fallback constant kept for backward-compatibility.  Actual detection is
# done at runtime via _get_assistant_token_id() so that changes to the
# Qwen3/Qwen2 chat template or tokenizer vocabulary do not silently break
# label masking.
ASSISTANT_TOKEN_ID = 77091
# Template string used to locate the "assistant" token in the live tokenizer.
_ASSISTANT_TEMPLATE = "<|im_start|>assistant"


def _get_assistant_token_id(processor) -> int:
    """Detect the assistant-turn header token ID from the tokenizer at runtime.

    Qwen3/Qwen2 chat format uses ``<|im_start|>assistant\\n``.
    ``<|im_start|>`` is a single special token; ``assistant`` is the
    immediately following ordinary token whose ID we need.

    Falls back to the module-level ``ASSISTANT_TOKEN_ID`` constant if
    detection fails (e.g. during unit tests that use a stub processor).
    """
    try:
        tokenizer = getattr(processor, "tokenizer", processor)
        ids = tokenizer.encode(_ASSISTANT_TEMPLATE, add_special_tokens=False)
        # ids[0] is <|im_start|>, ids[1] is "assistant"
        if len(ids) >= 2:
            return int(ids[1])
        # Some tokenizers may merge them; in that case fall back.
    except Exception:
        pass
    return ASSISTANT_TOKEN_ID


def get_rope_index_3(
    spatial_merge_size: int = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del second_per_grid_ts

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
    return position_ids, mrope_position_deltas


def _build_messages(
    conversations: Sequence[Dict[str, str]],
    media_path: Optional[str] = None,
    media_type: Optional[str] = None,
) -> List[Dict]:
    media_inserted = False
    messages = []
    for sentence in conversations:
        from_str = sentence["from"].lower()
        role = "user" if from_str == "human" else "assistant"
        text = sentence["value"]
        if role == "assistant":
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
            continue

        content = []
        parts = re.split(r"(<video>|<image>)", text, flags=re.IGNORECASE)
        for part in parts:
            if not part:
                continue
            lowered = part.lower()
            if lowered in {"<video>", "<image>"} and media_path is not None and not media_inserted:
                if media_type == "image":
                    content.append({"type": "image", "image": media_path})
                else:
                    content.append({"type": "video", "video": media_path})
                media_inserted = True
            elif part.strip():
                content.append({"type": "text", "text": part.strip()})

        if media_path is not None and not media_inserted:
            if media_type == "image":
                content.insert(0, {"type": "image", "image": media_path})
            else:
                content.insert(0, {"type": "video", "video": media_path})
            media_inserted = True

        if not content:
            content = [{"type": "text", "text": text}]
        messages.append({"role": role, "content": content})
    return messages


def _build_labels(input_ids: torch.Tensor, eos_token_id: int, assistant_token_id: int = ASSISTANT_TOKEN_ID) -> torch.Tensor:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    flat_ids = input_ids[0].tolist()
    pos = 0
    seq_len = len(flat_ids)
    while pos < seq_len:
        if flat_ids[pos] == assistant_token_id:
            # Qwen3/Qwen2 chat format: <|im_start|> assistant \n {response} <|im_end|>
            # flat_ids[pos] is "assistant", pos+1 is the newline token, pos+2 is the
            # first actual response token.
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < seq_len and flat_ids[ans_end] != eos_token_id:
                ans_end += 1
            if ans_end < seq_len:
                # Include the response text plus the EOS token (<|im_end|>) that
                # closes the assistant turn.  The +2 accounts for EOS itself (at
                # ans_end) and the trailing newline that follows it in the template.
                labels[0, ans_start: ans_end + 2] = input_ids[0, ans_start: ans_end + 2]
                pos = ans_end
        pos += 1
    return labels


def preprocess_qwen3vl_visual(
    conversations: Sequence[Dict[str, str]],
    processor,
    media_path: Optional[str] = None,
    media_type: Optional[str] = None,
    add_generation_prompt: bool = False,
    include_labels: bool = True,
) -> Dict[str, torch.Tensor]:
    messages = _build_messages(conversations, media_path=media_path, media_type=media_type)
    data_dict = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = data_dict["input_ids"]
    attention_mask = data_dict.get("attention_mask")
    if attention_mask is None:
        pad_id = processor.tokenizer.pad_token_id
        attention_mask = input_ids.ne(pad_id)

    merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2)

    image_grid_thw = data_dict.get("image_grid_thw")
    video_grid_thw = data_dict.get("video_grid_thw")
    position_ids, _ = get_rope_index_3(
        spatial_merge_size=merge_size,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )

    data_dict["input_ids"] = input_ids
    data_dict["attention_mask"] = attention_mask
    data_dict["position_ids"] = position_ids
    if include_labels:
        assistant_token_id = _get_assistant_token_id(processor)
        data_dict["labels"] = _build_labels(input_ids, eos_token_id=processor.tokenizer.eos_token_id, assistant_token_id=assistant_token_id)
    return data_dict


def collate_qwen3vl_batch(instances: Sequence[Dict], tokenizer) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_size = len(instances)
    max_len = max(instance["input_ids"].shape[-1] for instance in instances)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=instances[0]["input_ids"].dtype)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=instances[0]["labels"].dtype)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    position_ids = torch.zeros((3, batch_size, max_len), dtype=instances[0]["position_ids"].dtype)

    for idx, instance in enumerate(instances):
        cur_input_ids = instance["input_ids"].squeeze(0)
        cur_labels = instance["labels"].squeeze(0)
        cur_attention = instance["attention_mask"].squeeze(0).long()
        cur_position_ids = instance["position_ids"].squeeze(1)
        seq_len = cur_input_ids.shape[0]

        input_ids[idx, :seq_len] = cur_input_ids
        labels[idx, :seq_len] = cur_labels
        attention_mask[idx, :seq_len] = cur_attention
        position_ids[:, idx, :seq_len] = cur_position_ids

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        values = [instance[key] for instance in instances if key in instance]
        if not values:
            continue
        batch[key] = torch.cat(values, dim=0)

    pose_values = [instance.get("pose_feat") for instance in instances]
    scene_values = [instance.get("scene_feat") for instance in instances]
    if all(value is not None for value in pose_values):
        batch["pose_values"] = torch.stack(pose_values)
    if all(value is not None for value in scene_values):
        batch["scene_values"] = torch.stack(scene_values)

    return batch
