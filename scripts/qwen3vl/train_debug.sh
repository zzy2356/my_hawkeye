#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${HAWKEYE_WORK_ROOT:-$(pwd)}"
DATA_ROOT="${WORK_ROOT}/dataset/vid_noaudio_split/train_new"
IMAGE_ROOT="${WORK_ROOT}/dataset"
MODEL_PATH="${HAWKEYE_MODEL_PATH:-${WORK_ROOT}/models/Qwen3-VL-8B-Instruct}"
MODEL_BASE="${HAWKEYE_MODEL_BASE:-${MODEL_PATH}}"
OUT_DIR="${WORK_ROOT}/output_folder/Hawkeye-Qwen3VL-debug"

# Minimal debug run: verify data/collator/model path with tiny steps.
deepspeed --master_port=29503 train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --lora_enable True \
  --model_name_or_path "${MODEL_PATH}" \
  --model_base "${MODEL_BASE}" \
  --version v1 \
  --data_path "${WORK_ROOT}/dataset/new_train.json" \
  --video_folder "${DATA_ROOT}" \
  --image_folder "${IMAGE_ROOT}" \
  --bf16 True \
  --output_dir "${OUT_DIR}" \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 2 \
  --lazy_preprocess True \
  --report_to none \
  --cache_dir "./cache_dir"

echo "Debug training finished."
