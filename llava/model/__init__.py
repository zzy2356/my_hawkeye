# Lazy model init.
# Import each class explicitly where needed rather than loading everything
# at package import time.  This avoids the LanguageBind / transformers
# version conflict when running under the Qwen3-VL environment.
#
# Legacy LLaVA classes (require old transformers):
#   from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
#   from llava.model.language_model.llava_mpt  import LlavaMPTForCausalLM, LlavaMPTConfig
#
# Qwen3-VL Hawkeye classes (work with new transformers):
#   from llava.model.language_model.qwen3_vl_hawkeye import (
#       Qwen3VLHawkeyeAdapter,
#       load_pretrained_qwen3vl_hawkeye_model,
#   )
