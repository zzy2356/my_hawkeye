# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Only apply LLaMA flash attention monkey patch for non-Qwen3-VL models.
_model_path = ""
for _i, _arg in enumerate(sys.argv):
    if _arg == "--model_name_or_path" and _i + 1 < len(sys.argv):
        _model_path = sys.argv[_i + 1]
        break

_is_qwen3_vl = "qwen3-vl" in _model_path.lower() or ("qwen3" in _model_path.lower() and "vl" in _model_path.lower())
if not _is_qwen3_vl:
    try:
        from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    except Exception:
        pass

from llava.train.train import train

if __name__ == "__main__":
    train()
