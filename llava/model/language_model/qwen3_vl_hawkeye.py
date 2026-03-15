import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from packaging.version import InvalidVersion, Version
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llava.model.hawkeye_modules import (
    build_moe,
    build_moe_projector,
    build_pose_projector,
    build_pose_tower,
    build_scene_projector,
    build_scene_tower,
)

IGNORE_INDEX = -100

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None


def _get_transformers_version() -> Optional[Version]:
    try:
        import transformers

        return Version(transformers.__version__)
    except (ImportError, InvalidVersion):
        return None


def _resolve_hidden_size(config: Any, fallback_model: Optional[nn.Module] = None) -> int:
    for attr in ("hidden_size", "text_hidden_size"):
        value = getattr(config, attr, None)
        if value is not None:
            return value

    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None) is not None:
        return text_config.hidden_size

    if fallback_model is not None:
        embeddings = fallback_model.get_input_embeddings()
        embedding_dim = getattr(embeddings, "embedding_dim", None)
        if embedding_dim is not None:
            return embedding_dim

    raise ValueError("Unable to infer hidden size for the Qwen3-VL backbone.")


def _resolve_scene_token_count(config: Any) -> int:
    return int(getattr(config, "hawkeye_scene_token_count", 30))


def _resolve_dummy_token_id(config: Any) -> int:
    pad_id = getattr(config, "pad_token_id", None)
    eos_id = getattr(config, "eos_token_id", None)
    return int(pad_id if pad_id is not None else (eos_id if eos_id is not None else 0))


def _strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("base_model.model.", "base_model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def _read_adapter_base_path(model_path: str) -> Optional[str]:
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return None
    try:
        with open(adapter_config_path, "r", encoding="utf-8") as file:
            adapter_config = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return adapter_config.get("base_model_name_or_path")


def _choose_pretrained_source(model_path: str, backbone_path: str, filenames: List[str]) -> str:
    if all(os.path.exists(os.path.join(model_path, filename)) for filename in filenames):
        return model_path
    return backbone_path


class Qwen3VLHawkeyeAdapter(nn.Module):
    """Qwen3-VL wrapper that preserves Hawkeye's placeholder-to-embedding fusion flow."""

    def __init__(self, backbone_model: nn.Module, processor: Optional[Any] = None):
        super().__init__()
        self.model = backbone_model
        self.processor = processor
        self.config = backbone_model.config
        self._hidden_size = _resolve_hidden_size(self.config, fallback_model=backbone_model)
        self.scene_token_count = _resolve_scene_token_count(self.config)
        self.video_token_id = getattr(self.config, "video_token_id", 151656)
        self.image_token_id = getattr(self.config, "image_token_id", 151655)

        self.pose_tower = build_pose_tower(self._hidden_size)
        self.pose_projector = build_pose_projector(self._hidden_size)
        self.scene_tower = build_scene_tower(self._hidden_size)
        self.scene_projector = build_scene_projector(self._hidden_size)
        self.moe = build_moe(self._hidden_size, scene_token_count=self.scene_token_count)
        self.moe_projector = build_moe_projector(self._hidden_size)

        self.config.hawkeye_scene_token_count = self.scene_token_count
        self.config.hawkeye_hidden_size = self._hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def get_model(self) -> nn.Module:
        return self.model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = super().__getattr__("model")
            return getattr(model, name)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.model.resize_token_embeddings(*args, **kwargs)

    def _get_visual_module(self) -> Optional[nn.Module]:
        visual = getattr(self.model, "visual", None)
        if visual is not None:
            return visual
        inner_model = getattr(self.model, "model", None)
        return getattr(inner_model, "visual", None) if inner_model is not None else None

    def _get_visual_dtype(self) -> torch.dtype:
        visual = self._get_visual_module()
        if visual is None:
            return self.dtype
        if hasattr(visual, "get_dtype"):
            return visual.get_dtype()
        try:
            return next(visual.parameters()).dtype
        except StopIteration:
            return self.dtype

    def _run_visual_encoder(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor],
        modality: str,
    ) -> torch.Tensor:
        visual = self._get_visual_module()
        if visual is None:
            raise ValueError("Qwen3-VL visual module is unavailable; cannot materialize visual embeddings.")

        visual_inputs = pixel_values.to(device=self.device, dtype=self._get_visual_dtype())
        if grid_thw is not None:
            grid_thw = grid_thw.to(device=self.device)

        for kwargs in (
            {"grid_thw": grid_thw},
            {f"{modality}_grid_thw": grid_thw},
        ):
            try:
                return visual(visual_inputs, **kwargs)
            except TypeError:
                continue
        try:
            return visual(visual_inputs, grid_thw)
        except TypeError:
            return visual(visual_inputs)

    @staticmethod
    def _masked_scatter_multimodal_embeds(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        multimodal_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if multimodal_embeds is None:
            return inputs_embeds

        token_mask = input_ids == token_id
        token_count = int(token_mask.sum().item())
        if token_count == 0:
            return inputs_embeds

        multimodal_embeds = multimodal_embeds.reshape(-1, inputs_embeds.shape[-1]).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        if multimodal_embeds.shape[0] != token_count:
            raise ValueError(
                f"Mismatch between token count ({token_count}) and multimodal embeddings ({multimodal_embeds.shape[0]}) "
                f"for token_id={token_id}."
            )

        expanded_mask = token_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(expanded_mask, multimodal_embeds.reshape(-1))

    def _materialize_qwen_multimodal_embeds(
        self,
        input_ids: torch.Tensor,
        model_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any], bool]:
        prepared_kwargs = dict(model_kwargs)
        inputs_embeds = prepared_kwargs.pop("inputs_embeds", None)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        else:
            inputs_embeds = inputs_embeds.to(device=self.device, dtype=self.dtype)

        pixel_values = prepared_kwargs.pop("pixel_values", None)
        image_grid_thw = prepared_kwargs.pop("image_grid_thw", None)
        pixel_values_videos = prepared_kwargs.pop("pixel_values_videos", None)
        video_grid_thw = prepared_kwargs.pop("video_grid_thw", None)
        has_visual_inputs = any(value is not None for value in (pixel_values, pixel_values_videos))

        if pixel_values is not None:
            image_embeds = self._run_visual_encoder(pixel_values, image_grid_thw, modality="image")
            inputs_embeds = self._masked_scatter_multimodal_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                token_id=self.image_token_id,
                multimodal_embeds=image_embeds,
            )

        if pixel_values_videos is not None:
            video_embeds = self._run_visual_encoder(pixel_values_videos, video_grid_thw, modality="video")
            inputs_embeds = self._masked_scatter_multimodal_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                token_id=self.video_token_id,
                multimodal_embeds=video_embeds,
            )

        return inputs_embeds, prepared_kwargs, has_visual_inputs

    def encode_poses(self, pose_values: torch.Tensor) -> torch.Tensor:
        pose_tokens = self.pose_tower(pose_values.to(device=self.device, dtype=self.dtype))
        return self.pose_projector(pose_tokens)

    def encode_scenes(self, scene_values: torch.Tensor) -> torch.Tensor:
        scene_tokens = self.scene_tower(scene_values.to(device=self.device, dtype=self.dtype))
        return self.scene_projector(scene_tokens)

    def moe_route(self, pose_feat: torch.Tensor, scene_feat: torch.Tensor) -> torch.Tensor:
        if pose_feat.ndim == 2:
            pose_feat = pose_feat.unsqueeze(0)
        if scene_feat.ndim == 2:
            scene_feat = scene_feat.unsqueeze(0)
        moe_tokens = self.moe(pose_feat, scene_feat)
        moe_tokens = self.moe_projector(moe_tokens)
        return moe_tokens.squeeze(0)

    def _build_hawkeye_token_sequences(
        self,
        pose_values: Optional[torch.Tensor],
        scene_values: Optional[torch.Tensor],
        batch_size: int,
    ) -> List[Optional[torch.Tensor]]:
        if pose_values is None and scene_values is None:
            return [None] * batch_size

        token_sequences: List[Optional[torch.Tensor]] = []
        for batch_idx in range(batch_size):
            pose_sample = pose_values[batch_idx] if pose_values is not None else None
            scene_sample = scene_values[batch_idx] if scene_values is not None else None

            if pose_sample is None and scene_sample is None:
                token_sequences.append(None)
                continue

            if pose_sample is not None:
                pose_tokens = self.encode_poses(pose_sample)
            else:
                pose_tokens = torch.zeros(1, self._hidden_size, device=self.device, dtype=self.dtype)

            if scene_sample is not None:
                scene_tokens = self.encode_scenes(scene_sample)
            else:
                scene_tokens = torch.zeros(1, self._hidden_size, device=self.device, dtype=self.dtype)

            token_sequences.append(self.moe_route(pose_tokens, scene_tokens))

        return token_sequences

    @staticmethod
    def _find_contiguous_spans(token_ids: torch.Tensor, target_token_id: int) -> List[Tuple[int, int]]:
        positions = torch.where(token_ids == target_token_id)[0]
        if positions.numel() == 0:
            return []
        spans = []
        start = positions[0].item()
        prev = start
        for pos in positions[1:].tolist():
            if pos != prev + 1:
                spans.append((start, prev + 1))
                start = pos
            prev = pos
        spans.append((start, prev + 1))
        return spans

    def _splice_hawkeye_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        hawkeye_tokens: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size = input_ids.shape[0]
        dummy_token_id = _resolve_dummy_token_id(self.config)
        hidden_size = inputs_embeds.shape[-1]

        sample_payloads = []
        max_len = 0
        for batch_idx in range(batch_size):
            sample_input_ids = input_ids[batch_idx]
            sample_input_embeds = inputs_embeds[batch_idx]
            sample_attention = (
                attention_mask[batch_idx]
                if attention_mask is not None
                else torch.ones_like(sample_input_ids, dtype=torch.long)
            )
            valid_len = int(sample_attention.long().sum().item())
            sample_input_ids = sample_input_ids[:valid_len]
            sample_input_embeds = sample_input_embeds[:valid_len]
            sample_attention = sample_attention[:valid_len]
            sample_labels = labels[batch_idx, :valid_len] if labels is not None else None
            sample_position_ids = position_ids[:, batch_idx, :valid_len] if position_ids is not None else None
            sample_hawkeye = hawkeye_tokens[batch_idx]

            if sample_hawkeye is None:
                new_input_ids = sample_input_ids
                new_input_embeds = sample_input_embeds
                new_attention = sample_attention
                new_labels = sample_labels
                new_position_ids = sample_position_ids
            else:
                spans = self._find_contiguous_spans(sample_input_ids, self.video_token_id)
                if not spans:
                    new_input_ids = sample_input_ids
                    new_input_embeds = sample_input_embeds
                    new_attention = sample_attention
                    new_labels = sample_labels
                    new_position_ids = sample_position_ids
                else:
                    insert_at = spans[-1][1]
                    num_tokens = sample_hawkeye.shape[0]
                    dummy_ids = torch.full(
                        (num_tokens,),
                        dummy_token_id,
                        device=sample_input_ids.device,
                        dtype=sample_input_ids.dtype,
                    )
                    new_input_ids = torch.cat([sample_input_ids[:insert_at], dummy_ids, sample_input_ids[insert_at:]], dim=0)
                    hawkeye_embed_block = sample_hawkeye.to(device=sample_input_embeds.device, dtype=sample_input_embeds.dtype)
                    new_input_embeds = torch.cat(
                        [sample_input_embeds[:insert_at], hawkeye_embed_block, sample_input_embeds[insert_at:]],
                        dim=0,
                    )
                    new_attention = torch.cat(
                        [
                            sample_attention[:insert_at],
                            torch.ones(num_tokens, device=sample_attention.device, dtype=sample_attention.dtype),
                            sample_attention[insert_at:],
                        ],
                        dim=0,
                    )
                    if sample_labels is not None:
                        ignore_tokens = torch.full(
                            (num_tokens,),
                            IGNORE_INDEX,
                            device=sample_labels.device,
                            dtype=sample_labels.dtype,
                        )
                        new_labels = torch.cat([sample_labels[:insert_at], ignore_tokens, sample_labels[insert_at:]], dim=0)
                    else:
                        new_labels = None

                    if sample_position_ids is not None:
                        start_position = int(sample_position_ids[:, insert_at - 1].max().item() + 1) if insert_at > 0 else 0
                        hawkeye_pos = (
                            torch.arange(
                                start_position,
                                start_position + num_tokens,
                                device=sample_position_ids.device,
                                dtype=sample_position_ids.dtype,
                            )
                            .view(1, -1)
                            .expand(3, -1)
                        )
                        tail_position_ids = sample_position_ids[:, insert_at:] + num_tokens
                        new_position_ids = torch.cat(
                            [sample_position_ids[:, :insert_at], hawkeye_pos, tail_position_ids],
                            dim=1,
                        )
                    else:
                        new_position_ids = None

            max_len = max(max_len, new_input_ids.shape[0])
            sample_payloads.append((new_input_ids, new_input_embeds, new_attention, new_labels, new_position_ids))

        batch_input_ids = torch.full(
            (batch_size, max_len),
            dummy_token_id,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        batch_inputs_embeds = torch.zeros(
            (batch_size, max_len, hidden_size),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        batch_attention = torch.zeros((batch_size, max_len), device=input_ids.device, dtype=torch.long)
        batch_labels = None
        if labels is not None:
            batch_labels = torch.full((batch_size, max_len), IGNORE_INDEX, device=input_ids.device, dtype=labels.dtype)
        batch_position_ids = None
        if position_ids is not None:
            batch_position_ids = torch.zeros((3, batch_size, max_len), device=input_ids.device, dtype=position_ids.dtype)

        for batch_idx, (sample_ids, sample_embeds, sample_attention, sample_labels, sample_pos) in enumerate(sample_payloads):
            seq_len = sample_ids.shape[0]
            batch_input_ids[batch_idx, :seq_len] = sample_ids
            batch_inputs_embeds[batch_idx, :seq_len] = sample_embeds
            batch_attention[batch_idx, :seq_len] = sample_attention.long()
            if batch_labels is not None and sample_labels is not None:
                batch_labels[batch_idx, :seq_len] = sample_labels
            if batch_position_ids is not None and sample_pos is not None:
                batch_position_ids[:, batch_idx, :seq_len] = sample_pos

        return batch_input_ids, batch_inputs_embeds, batch_attention, batch_labels, batch_position_ids

    def _prepare_qwen_hawkeye_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        model_kwargs = dict(kwargs)
        legacy_images = model_kwargs.pop("images", None)
        if legacy_images is not None:
            model_kwargs.update(self._normalize_legacy_multimodal_inputs(legacy_images))

        pose_values = model_kwargs.pop("pose_values", None)
        scene_values = model_kwargs.pop("scene_values", None)
        for key in ("keys", "video_label"):
            model_kwargs.pop(key, None)

        input_ids = model_kwargs.get("input_ids")
        if input_ids is None:
            return model_kwargs

        input_ids = input_ids.to(device=self.device)
        attention_mask = model_kwargs.get("attention_mask")
        labels = model_kwargs.get("labels")
        position_ids = model_kwargs.get("position_ids")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        if labels is not None:
            labels = labels.to(device=self.device)
        if position_ids is not None:
            position_ids = position_ids.to(device=self.device)
        inputs_embeds, model_kwargs, has_visual_inputs = self._materialize_qwen_multimodal_embeds(input_ids, model_kwargs)
        hawkeye_sequences = self._build_hawkeye_token_sequences(
            pose_values=pose_values,
            scene_values=scene_values,
            batch_size=input_ids.shape[0],
        )
        has_hawkeye = any(sequence is not None for sequence in hawkeye_sequences)
        if not has_visual_inputs and not has_hawkeye:
            model_kwargs["input_ids"] = input_ids
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            if labels is not None:
                model_kwargs["labels"] = labels
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids
            return model_kwargs

        new_input_ids, new_inputs_embeds, new_attention_mask, new_labels, new_position_ids = self._splice_hawkeye_tokens(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            hawkeye_tokens=hawkeye_sequences,
        )

        model_kwargs["input_ids"] = new_input_ids
        model_kwargs["inputs_embeds"] = new_inputs_embeds
        model_kwargs["attention_mask"] = new_attention_mask
        if new_labels is not None:
            model_kwargs["labels"] = new_labels
        if new_position_ids is not None:
            model_kwargs["position_ids"] = new_position_ids
        return model_kwargs

    def _normalize_legacy_multimodal_inputs(self, legacy_images: Any) -> Dict[str, Any]:
        if not isinstance(legacy_images, (list, tuple)) or len(legacy_images) != 4:
            return {"images": legacy_images}

        video_values, pose_values, scene_values, keys = legacy_images
        normalized: Dict[str, Any] = {}
        key_list = list(keys) if isinstance(keys, (list, tuple)) else [keys]
        if any(str(key).lower() == "video" for key in key_list):
            normalized["pixel_values_videos"] = video_values
        else:
            normalized["pixel_values"] = video_values

        if isinstance(pose_values, (list, tuple)) and len(pose_values) == 1:
            pose_values = pose_values[0]
        if isinstance(scene_values, (list, tuple)) and len(scene_values) == 1:
            scene_values = scene_values[0]

        normalized["pose_values"] = pose_values
        normalized["scene_values"] = scene_values
        return normalized

    def forward(self, *args, **kwargs):
        model_kwargs = self._prepare_qwen_hawkeye_inputs(kwargs)
        return self.model(*args, **model_kwargs)

    def generate(self, *args, **kwargs):
        model_kwargs = self._prepare_qwen_hawkeye_inputs(kwargs)
        return self.model.generate(*args, **model_kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)


def load_pretrained_qwen3vl_hawkeye_model(
    model_path: str,
    model_base: Optional[str] = None,
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map: str = "auto",
    device: str = "cuda",
) -> Tuple[Any, Qwen3VLHawkeyeAdapter, Dict[str, Any], int]:
    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "cache_dir": r"./",
        "trust_remote_code": True,
    }

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    detected_base_path = _read_adapter_base_path(model_path)
    backbone_path = model_base or detected_base_path or model_path
    if detected_base_path is not None and model_base is None and not os.path.exists(detected_base_path):
        warnings.warn(
            f"Adapter checkpoint points to base model '{detected_base_path}', but that path is unavailable. "
            "Pass model_base explicitly if running offline."
        )
        backbone_path = model_path

    version = _get_transformers_version()
    if version is not None and version < Version("4.57.0"):
        warnings.warn(
            "The current transformers version is too old for Qwen3-VL. "
            "Please upgrade to transformers >= 4.57.0 before running."
        )

    config_source = _choose_pretrained_source(model_path, backbone_path, ["config.json"])
    tokenizer_source = _choose_pretrained_source(model_path, backbone_path, ["tokenizer_config.json"])
    processor_source = _choose_pretrained_source(model_path, backbone_path, ["preprocessor_config.json"])

    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False, trust_remote_code=True)

    processor = None
    if AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
        except Exception:
            processor = None

    if AutoModelForImageTextToText is not None:
        backbone_model = AutoModelForImageTextToText.from_pretrained(
            backbone_path,
            low_cpu_mem_usage=True,
            config=config,
            **kwargs,
        )
    else:
        backbone_model = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            low_cpu_mem_usage=True,
            config=config,
            **kwargs,
        )
    if device_map != "auto" and not load_8bit and not load_4bit:
        backbone_model.to(device=device)

    model: nn.Module = Qwen3VLHawkeyeAdapter(backbone_model=backbone_model, processor=processor)

    non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
        non_lora_trainables = _strip_state_dict_prefixes(non_lora_trainables)
        model.load_state_dict(non_lora_trainables, strict=False)

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_path)
            if not load_4bit and not load_8bit and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
        except Exception as exc:
            warnings.warn(f"Failed to load or merge LoRA adapter from {model_path}: {exc}")

    processor_dict: Dict[str, Any] = {"qwen": processor}
    if processor is not None:
        processor_dict["image"] = getattr(processor, "image_processor", processor)
        processor_dict["video"] = getattr(processor, "video_processor", processor)

    context_len = getattr(config, "max_position_embeddings", None)
    if context_len is None:
        text_config = getattr(config, "text_config", None)
        context_len = getattr(text_config, "max_position_embeddings", 32768) if text_config is not None else 32768

    return tokenizer, model, processor_dict, context_len
