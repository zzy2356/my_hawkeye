#!/usr/bin/env bash
set -euo pipefail

export HAWKEYE_MODEL_PATH="${HAWKEYE_MODEL_PATH:-models/Qwen3-VL-8B-Instruct}"
export HAWKEYE_MODEL_BASE="${HAWKEYE_MODEL_BASE:-}"

python eval.py

echo "Evaluation finished."
