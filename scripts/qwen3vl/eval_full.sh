#!/usr/bin/env bash
set -euo pipefail

# eval.py now defaults to models/Qwen3-VL-8B-Instruct.
python eval.py

echo "Evaluation finished."
