# Qwen3-VL Migration Guide

## Purpose

This document describes the current migration status of Hawkeye from the original
`Vicuna + LanguageBind / Video-LLaVA` stack to the new `Qwen3-VL` stack.

The goal of the migration is:

1. Replace the old multimodal backbone with `Qwen3-VL`.
2. Keep Hawkeye's core pipeline:
   `text tokens + VIDEO placeholder -> internal multimodal prepare/splice -> LLM`.
3. Preserve Hawkeye's task-specific `pose / scene GNN / MoE` enhancement for
   IasDig.

This document is intentionally explicit about what is already aligned and what is
not yet fully equivalent to the original paper implementation.

---

## Current Status

### What is already aligned

- `Qwen3-VL` is now the primary multimodal backbone for the migration path.
- The Qwen path no longer injects pose/scene information into prompt text.
- Input processing now preserves the original Hawkeye idea:
  a text sequence contains the video placeholder first, then multimodal features
  are spliced inside the model forward path.
- The Qwen wrapper now inserts Hawkeye-enhanced tokens after the contiguous
  `video_token_id` span, which matches the original "same-level internal fusion"
  direction.
- Training, smoke inference, and evaluation scripts now support
  `model_path + model_base` loading for `base model / LoRA adapter` workflows.
- Save outputs now include:
  `adapter weights + non_lora_trainables.bin + hawkeye_config.json`.

### What is not yet fully equivalent

- The current MoE routing is not yet a byte-for-byte reproduction of the old
  Hawkeye route.
  The old code used `sigmoid -> normalize`; the new Qwen path currently uses
  `softmax`.
- The current MoE output length is configured by
  `hawkeye_scene_token_count` and defaults to `30`.
  The old code declared resample tokens of length `30`, but the effective old
  sequence behavior in code was not a clean 30-token implementation either.
- The current label mask for Qwen SFT is much better than the old
  `labels = input_ids.clone()` shortcut, but it still relies on the current
  assistant sentinel token pattern and is not yet as robust as a role-span mask
  derived from the processor itself.

### Practical conclusion

If your question is "Has Qwen3-VL already completed a strict full replacement of
the original Hawkeye paper code?" the answer is:

- No, not yet in the strict sense.
- Yes, in the architectural sense that the migration path now keeps the key
  Hawkeye logic:
  internal placeholder-based fusion, pose tower, scene tower, and MoE-enhanced
  token insertion.

---

## Model Requirements

### Original Hawkeye route

If you run the original scripts under `scripts/v1_5/` or the original LLaVA
pipeline, you still need the original model stack:

1. `lmsys/vicuna-7b-v1.5`
2. `LanguageBind / LanguageBind_Video_merge`
3. `checkpoints/Video-LLaVA-Pretrain-7B/mm_projector.bin`

That route is still present in the repository.

### Qwen-Hawkeye migration route

If you run the migrated Qwen route, you do not need the old `Vicuna`,
`LanguageBind`, or `Video-LLaVA mm_projector` stack.

You need:

1. `models/Qwen3-VL-8B-Instruct`
2. Dataset video files
3. Pose features under `dataset/pose_feat/...`
4. Scene features under either `dataset/graph_feat/...` or `dataset/rel_feat/...`
5. If evaluating trained Qwen checkpoints:
   - `adapter_model.*`
   - `adapter_config.json`
   - `non_lora_trainables.bin`
   - `hawkeye_config.json`

### Rule of thumb

- Running `scripts/v1_5/*`: old model stack is still required.
- Running `scripts/qwen3vl/*`: only Qwen3-VL and Hawkeye auxiliary features are
  required.

---

## Current Qwen-Hawkeye Flow

### 1. Input and preprocessing

The migrated Qwen path keeps the original Hawkeye idea:

1. User text still contains the video placeholder.
2. Qwen chat-template preprocessing produces `input_ids`,
   `attention_mask`, `position_ids`, and visual tensors.
3. Pose and scene features are passed separately as tensors and are not
   serialized into the prompt.
4. Inside the model wrapper, Hawkeye tokens are generated and spliced into the
   embedding sequence.

### 2. Multimodal encoding

The Qwen route currently contains:

1. Qwen3-VL visual encoding for video tokens
2. Pose tower
3. Scene graph tower
4. MoE fusion
5. Same-level insertion of Hawkeye tokens into the Qwen input embedding stream

### 3. Token splice point

The splice point is currently:

1. Find the contiguous `video_token_id` span inside `input_ids`
2. Insert Hawkeye tokens immediately after that span
3. Mark inserted positions as valid in `attention_mask`
4. Mask inserted positions with `IGNORE_INDEX` in `labels`
5. Shift `position_ids` so the inserted Hawkeye sequence becomes part of the
   same LLM timeline

This is the current closest equivalent to the original
`prepare_inputs_labels_for_multimodal` behavior.

### 4. Save / load rules

For the Qwen route:

- Base-only inference:
  - `HAWKEYE_MODEL_PATH == HAWKEYE_MODEL_BASE == Qwen3-VL-8B-Instruct`
- Adapter inference:
  - `HAWKEYE_MODEL_PATH = checkpoint dir`
  - `HAWKEYE_MODEL_BASE = Qwen3-VL-8B-Instruct`

The loader now supports both cases.

---

## Important Alignment Notes

### MoE routing

Current state:

- Kept: expert routing over fused pose/scene sequence and generation of a
  Hawkeye token sequence for internal insertion.
- Not fully aligned yet:
  the routing formula is still not identical to the original code.

If strict parity with the old implementation is required, the next code change
should be:

1. Reconcile the routing activation with the old route
2. Reconcile the effective output token count with the old code and the paper's
   intended behavior
3. Decide whether to keep the current 30-token sequence or restore the old
   shorter effective sequence exactly

### Token splice location

Current state:

- Architecturally aligned.
- Hawkeye tokens are inserted after the video token span instead of being
  prepended as one auxiliary token.

This is a major correction relative to the earlier minimal adapter.

### Label mask strategy

Current state:

- Better than the old broken Qwen branch.
- Still not perfect.

Current mask behavior:

1. Padding tokens are masked
2. Non-assistant spans are masked
3. Inserted Hawkeye tokens are masked with `IGNORE_INDEX`

Remaining risk:

- The assistant-span detection still depends on the current processor template
  and assistant token sentinel layout.
- If the tokenizer or chat template changes, mask alignment must be rechecked.

---

## Recommended Test Order

Do not start with full training.

### Step 1: Environment import check

```bash
python -c "import torch, transformers, deepspeed, peft, torch_geometric; print('ok')"
python -m py_compile \
  llava/model/hawkeye_modules.py \
  llava/train/qwen3vl_data.py \
  llava/model/language_model/qwen3_vl_hawkeye.py \
  llava/model/builder.py \
  llava/train/train.py \
  eval.py \
  scripts/qwen3vl/smoke_infer.py
```

### Step 2: Single-sample smoke inference

```bash
python scripts/qwen3vl/smoke_infer.py \
  --model-path /path/to/models/Qwen3-VL-8B-Instruct \
  --model-base /path/to/models/Qwen3-VL-8B-Instruct \
  --video-path /path/to/video.mp4 \
  --pose-npy /path/to/frame_xxx.npy \
  --scene-npy /path/to/frame_xxx.npy \
  --print-shapes \
  --max-new-tokens 16
```

### Step 3: Two-step debug training

```bash
export HAWKEYE_WORK_ROOT=/path/to/Hawkeye
export HAWKEYE_MODEL_PATH=/path/to/models/Qwen3-VL-8B-Instruct
export HAWKEYE_MODEL_BASE=/path/to/models/Qwen3-VL-8B-Instruct
bash scripts/qwen3vl/train_debug.sh
```

### Step 4: Evaluation

```bash
export HAWKEYE_MODEL_PATH=/path/to/models/Qwen3-VL-8B-Instruct
export HAWKEYE_MODEL_BASE=/path/to/models/Qwen3-VL-8B-Instruct
bash scripts/qwen3vl/eval_full.sh
```

### Step 5: Adapter evaluation

```bash
export HAWKEYE_MODEL_PATH=/path/to/output_folder/Hawkeye-Qwen3VL/checkpoint-xxx
export HAWKEYE_MODEL_BASE=/path/to/models/Qwen3-VL-8B-Instruct
bash scripts/qwen3vl/eval_full.sh
```

---

## Data Expectations

The migrated Qwen route expects:

- training annotation file:
  `dataset/new_train.json`
- videos:
  `dataset/vid_noaudio_split/train_new/...`
  or matching test directories
- pose features:
  `dataset/pose_feat/...`
- scene features:
  either `dataset/graph_feat/...` or `dataset/rel_feat/...`

The code now accepts both `graph_feat` and `rel_feat` for scene features.

---

## Current Output Artifacts

After Qwen LoRA training, the output directory should contain:

1. `adapter_model.safetensors` or equivalent PEFT adapter file
2. `adapter_config.json`
3. `non_lora_trainables.bin`
4. `hawkeye_config.json`
5. tokenizer / processor files

These are all required for stable adapter reloads.

---

## Before Uploading To A Server

Please confirm the following:

1. You are using the Qwen route, not the old `scripts/v1_5/*` route
2. Your server has `torch`, `transformers`, `deepspeed`, `peft`,
   `torch_geometric`
3. `Qwen3-VL-8B-Instruct` exists on the server
4. Dataset paths on the server match the script expectations
5. You understand that current Qwen-Hawkeye is close in architecture, but not
   yet a strict paper-code clone in MoE routing and label-mask details

If you need strict original-code parity, treat the current code as the new
Qwen migration baseline, not as the final exact reproduction.
