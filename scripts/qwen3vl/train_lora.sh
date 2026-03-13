#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="dataset/vid_noaudio_split/train_new"
IMAGE_ROOT="dataset"
MODEL_PATH="models/Qwen3-VL-8B-Instruct"
OUT_DIR="output_folder/Hawkeye-Qwen3VL"

# Full training entry for Qwen3-VL + Hawkeye pipeline.
deepspeed --master_port=29501 train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --lora_enable True \
  --model_name_or_path "${MODEL_PATH}" \
  --version v1 \
  --data_path dataset/new_train.json \
  --video_folder "${DATA_ROOT}" \
  --image_folder "${IMAGE_ROOT}" \
  --bf16 True \
  --output_dir "${OUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to tensorboard \
  --cache_dir "./cache_dir"

echo "Qwen3-VL training finished."
