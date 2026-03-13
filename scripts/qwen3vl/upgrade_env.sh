#!/usr/bin/env bash
set -euo pipefail

# Upgrade runtime dependencies required by Qwen3-VL.
pip install -U "transformers>=4.57.0" "accelerate>=0.34.0" "tokenizers>=0.21.0"

# Optional but recommended for faster video processing.
pip install -U "qwen-vl-utils[decord]==0.0.14" || true

echo "Environment upgrade finished."
