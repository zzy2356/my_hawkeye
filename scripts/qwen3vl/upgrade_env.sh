#!/usr/bin/env bash
set -euo pipefail

# ── Core Qwen3-VL runtime requirements ──────────────────────────────────────
pip install -U "transformers>=4.57.0" "accelerate>=0.34.0" "tokenizers>=0.21.0"

# Optional but recommended for faster video processing.
pip install -U "qwen-vl-utils[decord]==0.0.14" || true

# ── Hawkeye auxiliary dependencies ───────────────────────────────────────────
# torch-geometric: required for SceneGraphTower (GTN graph network).
pip install torch-geometric || true

# fairscale: required for memory-efficient optimizers (used by llava_trainer).
pip install fairscale || true

# deepspeed: required for ZeRO-2 distributed training.
pip install deepspeed || true

# peft: LoRA adapter support.
pip install -U "peft>=0.9.0" || true

# Miscellaneous utilities used across the codebase.
pip install -U mmengine || true
pip install -U fvcore iopath || true
pip install -U decord av || true
pip install -U opencv-python-headless || true
pip install -U pandas tqdm || true

echo "Environment upgrade finished."
