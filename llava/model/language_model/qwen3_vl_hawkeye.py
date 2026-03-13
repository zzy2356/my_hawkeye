import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from packaging.version import InvalidVersion, Version
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


class Qwen3VLHawkeyeAdapter(nn.Module):
    """Minimal wrapper that keeps the current Hawkeye entry points stable while a Qwen3-VL backend is introduced."""

    def __init__(self, backbone_model: nn.Module, processor: Optional[Any] = None):
        super().__init__()
        self.model = backbone_model
        self.processor = processor
        self.config = backbone_model.config
        self._hidden_size = _resolve_hidden_size(self.config, fallback_model=backbone_model)

        self.pose_projector = nn.Linear(85, self._hidden_size)
        self.scene_projector = nn.Linear(353, self._hidden_size)
        self.moe_projector = nn.Linear(self._hidden_size * 2, self._hidden_size)
        self._last_aux_features = None
        self._warned_aux_passthrough = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def get_model(self) -> nn.Module:
        return self.model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.model.resize_token_embeddings(*args, **kwargs)

    def _pool_feature(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 1:
            return values
        return values.reshape(-1, values.shape[-1]).mean(dim=0)

    def encode_aux_features(
        self,
        pose_values: Optional[torch.Tensor] = None,
        scene_values: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if pose_values is None and scene_values is None:
            return None

        features = []
        if pose_values is not None:
            pose_values = self._pool_feature(pose_values).to(device=self.device, dtype=self.dtype)
            features.append(self.pose_projector(pose_values))

        if scene_values is not None:
            scene_values = self._pool_feature(scene_values).to(device=self.device, dtype=self.dtype)
            features.append(self.scene_projector(scene_values))

        if len(features) == 1:
            return features[0]

        return self.moe_projector(torch.cat(features, dim=-1))

    def _strip_hawkeye_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        model_kwargs = dict(kwargs)
        legacy_images = model_kwargs.pop("images", None)
        if legacy_images is not None:
            model_kwargs.update(self._normalize_legacy_multimodal_inputs(legacy_images))

        pose_values = model_kwargs.pop("pose_values", None)
        scene_values = model_kwargs.pop("scene_values", None)
        self._last_aux_features = self.encode_aux_features(pose_values=pose_values, scene_values=scene_values)

        if self._last_aux_features is not None and not self._warned_aux_passthrough:
            warnings.warn(
                "Qwen3-VL Hawkeye skeleton received pose/scene features, but the current minimal wrapper only stores "
                "their projected representation. The features are not injected into the backbone yet."
            )
            self._warned_aux_passthrough = True

        # Keep the wrapper resilient if upstream code passes data that this backend does not consume.
        for key in ("keys", "video_label"):
            if key in model_kwargs:
                model_kwargs.pop(key)

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
        model_kwargs = self._strip_hawkeye_kwargs(kwargs)
        return self.model(*args, **model_kwargs)

    def generate(self, *args, **kwargs):
        model_kwargs = self._strip_hawkeye_kwargs(kwargs)
        return self.model.generate(*args, **model_kwargs)


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
        kwargs["torch_dtype"] = torch.float16

    backbone_path = model_path
    if model_base is not None:
        warnings.warn(
            "The Qwen3-VL Hawkeye skeleton does not merge adapter weights from model_base yet. "
            "Loading model_path directly."
        )
    version = _get_transformers_version()
    if version is not None and version < Version("4.37.0"):
        warnings.warn(
            "The current transformers version is old for recent Qwen-VL backbones. "
            "This skeleton uses trust_remote_code, but upgrading transformers is recommended before training."
        )

    config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, use_fast=False, trust_remote_code=True)

    processor = None
    if AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(backbone_path, trust_remote_code=True)
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

    model = Qwen3VLHawkeyeAdapter(backbone_model=backbone_model, processor=processor)

    processor_dict: Dict[str, Any] = {"qwen": processor}
    if processor is not None:
        processor_dict["image"] = getattr(processor, "image_processor", processor)
        processor_dict["video"] = getattr(processor, "video_processor", processor)

    context_len = getattr(config, "max_position_embeddings", None)
    if context_len is None:
        text_config = getattr(config, "text_config", None)
        context_len = getattr(text_config, "max_position_embeddings", 32768) if text_config is not None else 32768

    return tokenizer, model, processor_dict, context_len