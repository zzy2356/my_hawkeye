# Lazy top-level init: do NOT import LlavaLlamaForCausalLM here.
# Importing it unconditionally pulls in LanguageBind, which depends on
# transformers internals (_expand_mask) removed in transformers >= 4.36.
# Users who need the legacy LLaVA class should import it directly:
#   from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
