# Qwen3-VL End-to-End Workflow

This folder contains runnable scripts for the replaced Qwen3-VL pipeline.

## Prerequisites

- Model directory exists: models/Qwen3-VL-8B-Instruct
- Dataset files exist: dataset/new_train.json and related video/feature folders
- CUDA + deepspeed available in current environment

## 1) Upgrade environment

### Bash

bash scripts/qwen3vl/upgrade_env.sh

### PowerShell

pip install -U "transformers>=4.57.0" "accelerate>=0.34.0" "tokenizers>=0.21.0"
pip install -U "qwen-vl-utils[decord]==0.0.14"

## 2) Smoke inference (single video)

### Bash

python scripts/qwen3vl/smoke_infer.py \
  --model-path models/Qwen3-VL-8B-Instruct \
  --video-path Qwen3-VL/qwen-vl-finetune/demo/videos/v_7bUu05RIksU.mp4

### PowerShell

python scripts/qwen3vl/smoke_infer.py --model-path models/Qwen3-VL-8B-Instruct --video-path Qwen3-VL/qwen-vl-finetune/demo/videos/v_7bUu05RIksU.mp4

Expected result:
- Console prints "=== Smoke Inference Output ==="
- Non-empty text output after generation

## 3) Debug training (2 steps)

Use this first to validate data loader + collator + model wiring.

### Bash

bash scripts/qwen3vl/train_debug.sh

### PowerShell

deepspeed --master_port=29503 train_mem.py --deepspeed ./scripts/zero2.json --lora_enable True --model_name_or_path models/Qwen3-VL-8B-Instruct --version v1 --data_path dataset/new_train.json --video_folder dataset/vid_noaudio_split/train_new --image_folder dataset --bf16 True --output_dir output_folder/Hawkeye-Qwen3VL-debug --max_steps 2 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --evaluation_strategy no --save_strategy steps --save_steps 2 --save_total_limit 1 --learning_rate 1e-5 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 True --model_max_length 4096 --gradient_checkpointing True --dataloader_num_workers 2 --lazy_preprocess True --report_to none --cache_dir ./cache_dir

Expected result:
- Training starts
- Global step reaches 2
- Checkpoint saved in output_folder/Hawkeye-Qwen3VL-debug

## 4) Full LoRA training

### Bash

bash scripts/qwen3vl/train_lora.sh

### PowerShell

deepspeed --master_port=29501 train_mem.py --deepspeed ./scripts/zero2.json --lora_enable True --model_name_or_path models/Qwen3-VL-8B-Instruct --version v1 --data_path dataset/new_train.json --video_folder dataset/vid_noaudio_split/train_new --image_folder dataset --bf16 True --output_dir output_folder/Hawkeye-Qwen3VL --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy no --save_strategy steps --save_steps 200 --save_total_limit 2 --learning_rate 2e-5 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 True --model_max_length 4096 --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --report_to tensorboard --cache_dir ./cache_dir

## 5) Evaluation

### Bash

bash scripts/qwen3vl/eval_full.sh

### PowerShell

python eval.py

Expected result:
- CSV outputs are generated under dataset/saved_result/test_res

## Recommended validation checklist

1. Smoke inference returns text.
2. Debug training reaches step 2 and writes checkpoint.
3. Full training writes periodic checkpoints.
4. Evaluation generates CSV result files.

## Common issues

1. transformers too old:
- Symptom: cannot import Qwen3-VL model class.
- Fix: run upgrade step again.

2. Out of memory:
- Reduce per_device_train_batch_size to 1.
- Increase gradient_accumulation_steps.
- Add --bits 4 for QLoRA if needed.

3. Missing video decoder dependency:
- Install qwen-vl-utils with decord extra.
