# Hawkeye × Qwen3-VL — Project Logic Flow

This document gives a precise description of every stage in the Qwen3-VL migration
of Hawkeye: data organisation, model interfaces, adapter injection, and the
training / inference pipelines.

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Hawkeye × Qwen3-VL System                           │
├──────────────┬──────────────────────────────┬──────────────────────────────┤
│  Video Input │  Pose Features  (5,17,5)      │  Scene Features  (5,353)     │
│  .mp4 / .avi │  pose_feat/…/frame_N.npy      │  graph_feat/ or rel_feat/    │
└──────┬───────┴──────────────┬───────────────┴──────────────┬───────────────┘
       │                      │                              │
       ▼                      ▼                              ▼
┌─────────────┐      ┌─────────────────┐          ┌──────────────────────┐
│  Qwen3-VL   │      │  PoseTower      │          │  SceneGraphTower     │
│  Visual     │      │  (flatten→85d   │          │  (GTN, 2 layers,     │
│  Encoder    │      │  → Linear→H)    │          │   node_dim=151,      │
│  (ViT)      │      └────────┬────────┘          │   edge_attr=51→H)    │
└──────┬──────┘               │                   └──────────┬───────────┘
       │ video tokens          │                              │
       ▼                      ▼  pose_projector               ▼  scene_projector
  [V₁…Vₙ] ∈ Rⁿˣᴴ       [P₁] ∈ R¹ˣᴴ                  [S₁…S₃₀] ∈ R³⁰ˣᴴ
                              │                              │
                              └──────────┬───────────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  HawkeyeMoE          │
                              │  (2 experts, 30 out) │
                              └──────────┬───────────┘
                                         │ moe_projector
                                         ▼
                              [M₁…M₃₀] ∈ R³⁰ˣᴴ   ← Hawkeye tokens
                                         │
                    ┌────────────────────┘
                    │  _splice_hawkeye_tokens
                    │  inserts M immediately after video_token_id span
                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  Full input_embeds sequence sent to Qwen3-VL LLM backbone:               │
│  [text₁ … text_k | vid_tok₁…vid_tokN | M₁…M₃₀ | text_k+1 … text_end]  │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
         Qwen3-VL LLM  (8B, bfloat16)
                    │
                    ▼
         Decoded text answer
```

### Modules and their role

| Module | Class | Input → Output |
|--------|-------|----------------|
| Qwen3-VL Visual | `backbone.visual` | `pixel_values_videos (B,T,C,H,W)` → `(N_video_tokens, H)` |
| PoseTower | `PoseTower` | `(B, 5, 17, 5)` → reshape `(B, 85)` → `Linear(85, H)` → `(B, 1, H)` |
| PoseProjector | `nn.Linear(H, H)` | `(B, 1, H)` → `(B, 1, H)` |
| SceneGraphTower | `SceneGraphTower` | `(B, 5, 353)` → graph walk → `(B, 30, H)` |
| SceneProjector | `nn.Linear(H, H)` | `(B, 30, H)` → `(B, 30, H)` |
| HawkeyeMoE | `HawkeyeMoE` | `pose (B,1,H)` + `scene (B,30,H)` → routed → `(B, 30, H)` |
| MoE Projector | `nn.Linear(H, H)` | `(B, 30, H)` → `(B, 30, H)` |

---

## 2. Data Organisation

### 2.1 Training JSON (`dataset/new_train.json`)

```json
[
  {
    "id": "sample_001",
    "mode": "video",
    "path": "vid_noaudio_split/train_new/Normal/0.mp4",
    "conversations": [
      {"from": "human",    "value": "<video>\nPlease determine if …"},
      {"from": "gpt",      "value": "No, the video does not show …"}
    ]
  },
  …
]
```

### 2.2 Feature directory layout

```
dataset/
  new_train.json
  vid_noaudio_split/
    train_new/        ← training clips  (mp4)
    test_new/         ← test clips
  pose_feat/
    train/            ← pose features per video folder
      <category>/
        frame_N.npy   ← shape (T, 17, 5), T ≤ 5
    test/
  graph_feat/         ← scene graph features (preferred)
    train/
      <category>/
        frame_N.npy   ← shape (T, 353), T ≤ 5
    test/
  rel_feat/           ← fallback if graph_feat absent
    train/ / test/
  saved_result/
    test_res/         ← CSV output from eval.py
```

### 2.3 Feature shapes and padding

| Feature | Raw shape | Padded to | Flatten / pool |
|---------|-----------|-----------|----------------|
| pose | `(T, 17, 5)`, T ≤ 5 | `(5, 17, 5)` | `reshape(-1)` → 85-dim |
| scene | `(T, 353)`, T ≤ 5 | `(5, 353)` | passed to GTN as nodes |

---

## 3. Model Interface — `Qwen3VLHawkeyeAdapter`

File: `llava/model/language_model/qwen3_vl_hawkeye.py`

### 3.1 Constructor

```
Qwen3VLHawkeyeAdapter(backbone_model, processor)
├── self.model          ← Qwen3-VL backbone (AutoModelForImageTextToText)
├── self.pose_tower     ← PoseTower(H, pose_dim=85)
├── self.pose_projector ← nn.Linear(H, H)
├── self.scene_tower    ← SceneGraphTower(H)
├── self.scene_projector← nn.Linear(H, H)
├── self.moe            ← HawkeyeMoE(H, scene_token_count=30)
└── self.moe_projector  ← nn.Linear(H, H)
```

### 3.2 Keyword arguments consumed by the adapter

The following extra kwargs are stripped before the call is forwarded to the backbone:

| kwarg | Type | Purpose |
|-------|------|---------|
| `pose_values` | `Tensor (B, 5, 17, 5)` | Pose skeleton features |
| `scene_values` | `Tensor (B, 5, 353)` | Scene graph features |
| `images` | legacy list | Compatibility shim for old LLaVA callers |
| `keys` | list | Modality key (stripped, not used) |
| `video_label` | any | Training label (stripped) |

### 3.3 `forward()` pipeline (training)

```
kwargs
  │
  ├─ _strip_hawkeye_kwargs()          pop pose/scene/legacy fields
  │
  ├─ _materialize_qwen_multimodal_embeds()
  │    ├─ get_input_embeddings()(input_ids) → inputs_embeds
  │    ├─ run ViT on pixel_values / pixel_values_videos
  │    └─ scatter video/image embeddings into inputs_embeds
  │
  ├─ _build_hawkeye_token_sequences()
  │    per batch item:
  │    ├─ encode_poses(pose_values[i])    → pose_tokens  (1, H)
  │    ├─ encode_scenes(scene_values[i])  → scene_tokens (30, H)
  │    └─ moe_route(pose_tokens, scene_tokens) → hawkeye_tokens (30, H)
  │
  ├─ _splice_hawkeye_tokens()
  │    per batch item:
  │    ├─ find contiguous video_token_id span in input_ids
  │    ├─ insert hawkeye_tokens immediately after that span
  │    ├─ pad attention_mask with 1s for new tokens
  │    ├─ pad labels with IGNORE_INDEX for new tokens
  │    └─ shift position_ids for tokens after insertion point
  │
  └─ self.model(**prepared_kwargs)      forward to Qwen3-VL backbone
```

### 3.4 `generate()` pipeline (inference)

Same as `forward()` except:
1. Calls `self.model.generate()` instead of `self.model()`
2. Stores `last_prefix_lens` = per-sample sum of attention_mask
   (callers trim generated tokens using `output_ids[i, prefix_lens[i]:]`)

### 3.5 `load_pretrained_qwen3vl_hawkeye_model()` (builder)

```
load_pretrained_qwen3vl_hawkeye_model(model_path, model_base, …)
  │
  ├─ AutoConfig.from_pretrained(config_source)
  ├─ AutoTokenizer.from_pretrained(tokenizer_source)
  ├─ AutoProcessor.from_pretrained(processor_source)
  ├─ AutoModelForImageTextToText.from_pretrained(backbone_path, torch_dtype=bfloat16)
  │    (cache_dir from HAWKEYE_CACHE_DIR env var if set, else HF default)
  ├─ wrap in Qwen3VLHawkeyeAdapter(backbone_model, processor)
  │
  ├─ if non_lora_trainables.bin exists:           ← end-of-training Hawkeye weights
  │    load_state_dict(non_lora_trainables, strict=False)
  │
  ├─ if hawkeye_non_lora.bin exists:              ← intermediate checkpoint Hawkeye weights
  │    load_state_dict(hawkeye_non_lora, strict=False)
  │
  ├─ if adapter_config.json exists:
  │    PeftModel.from_pretrained → merge_and_unload          ← apply LoRA weights
  │
  └─ return (tokenizer, adapter, processor_dict, context_len)
```

---

## 4. Training Pipeline

### 4.1 Entry points

```
train_mem.py           ← recommended entry point
  │  (conditionally skips LLaMA flash-attn monkey patch for Qwen3-VL)
  └─ calls llava/train/train.py::train()

llava/train/train.py::train()
  ├─ parse (ModelArguments, DataArguments, TrainingArguments)
  ├─ is_qwen3_vl = _is_qwen3_vl_model_name(model_name_or_path)
  │
  ├─ [Qwen3-VL branch]
  │    ├─ load_pretrained_qwen3vl_hawkeye_model(model_path, …)
  │    ├─ LoRA via peft.get_peft_model
  │    │    target_modules: ["q_proj","k_proj","v_proj","o_proj",
  │    │                     "gate_proj","up_proj","down_proj"]
  │    ├─ _set_qwen_hawkeye_modules_trainable(model)
  │    │    (Hawkeye sub-modules always trained even under LoRA)
  │    └─ data_args.qwen_multimodal = True
  │
  ├─ LazySupervisedDataset(data_args, tokenizer)
  │    __getitem__:
  │    ├─ load conversation JSON
  │    ├─ np.load pose_feat → pad to (5,17,5) → torch.Tensor
  │    ├─ np.load scene_feat → pad to (5,353)  → torch.Tensor
  │    ├─ [Qwen path] preprocess_qwen3vl_visual(conversations, processor, media_path)
  │    │    → {input_ids, attention_mask, position_ids, labels,
  │    │        pixel_values_videos, video_grid_thw}
  │    └─ add pose_feat + scene_feat to data_dict
  │
  ├─ DataCollatorForSupervisedDataset(qwen_multimodal=True)
  │    → collate_qwen3vl_batch(instances, tokenizer)
  │         pads input_ids/labels/attention_mask/position_ids
  │         cats pixel_values_videos / video_grid_thw
  │         stacks pose_values / scene_values
  │
  └─ Qwen3VLHawkeyeTrainer.train()       ← used for Qwen3-VL path
       ├─ batch contains pose_values, scene_values → passed to model.forward()
       ├─ model.forward() runs the full adapter pipeline
       └─ deepspeed ZeRO-2 + gradient checkpointing + LoRA
```

### 4.2 Checkpoint saving

After training completes, the output directory contains:

| File | Content |
|------|---------|
| `adapter_model.safetensors` | LoRA delta weights |
| `adapter_config.json` | LoRA configuration |
| `non_lora_trainables.bin` | Hawkeye modules (pose/scene/moe towers) saved at end of training |
| `hawkeye_config.json` | Adapter config snapshot |
| `tokenizer_config.json` | Tokenizer files |

Intermediate `checkpoint-N/` directories (written at `save_steps`) additionally contain:

| File | Content |
|------|---------|
| `hawkeye_non_lora.bin` | Hawkeye module states at step N (written by `Qwen3VLHawkeyeTrainer`) |

This ensures Hawkeye sub-module weights are never lost when resuming training from an intermediate checkpoint.

---

## 5. Inference Pipeline

### 5.1 Evaluation (`eval.py`)

```
eval.py::main()
  │
  ├─ load_pretrained_model(model_path, model_base, model_name)
  │    → for Qwen3-VL: load_pretrained_qwen3vl_hawkeye_model(…)
  │
  ├─ for each video in dataset/vid_split/test_new/ (or Ucf/):
  │    ├─ np.load pose_feat → pad to (5,17,5)
  │    ├─ np.load scene_feat → pad to (5,353)
  │    │
  │    └─ _run_qwen3_vl_inference(model, processor, video_path, prompt,
  │                                pose_feature, scene_feature, max_new_tokens)
  │         ├─ preprocess_qwen3vl_visual(conversations, processor, video_path)
  │         │    → {input_ids, attention_mask, position_ids,
  │         │        pixel_values_videos, video_grid_thw}
  │         ├─ model.generate(**model_inputs,
  │         │                  pose_values=pose_feature.unsqueeze(0),
  │         │                  scene_values=scene_feature.unsqueeze(0),
  │         │                  max_new_tokens=16, do_sample=False)
  │         ├─ trim output: output_ids[prefix_len:]  (using last_prefix_lens)
  │         └─ processor.batch_decode(trimmed) → answer string
  │
  └─ pd.DataFrame(filenames, outputs).to_csv(save_root/folder.csv)
```

### 5.2 Smoke inference (`scripts/qwen3vl/smoke_infer.py`)

```
smoke_infer.py::main()
  ├─ load_pretrained_model(model_path, …)
  ├─ preprocess_qwen3vl_visual(conversations, processor, video_path)
  ├─ model.generate(…, pose_values=zeros(1,5,17,5) or loaded,
  │                     scene_values=zeros(1,5,353) or loaded)
  ├─ trim with last_prefix_lens
  └─ print decoded text
```

---

## 6. Key Design Decisions

### 6.1 Hawkeye token injection point

Hawkeye MoE tokens are inserted **immediately after** the contiguous
`video_token_id` span in the embedding sequence.  This mirrors the original
Hawkeye paper's internal multimodal fusion principle (same-level embedding
fusion, not external prompt injection).

### 6.2 dtype

The backbone loads in **bfloat16** (`torch_dtype=torch.bfloat16`).
All Hawkeye sub-modules are cast to the backbone dtype before each forward pass
via `_ensure_hawkeye_modules_dtype()` to prevent mixed-dtype matmul errors.

### 6.3 LoRA training scope

```
Trainable under LoRA:
  Qwen3-VL attention:    q_proj, k_proj, v_proj, o_proj
  Qwen3-VL MLP:          gate_proj, up_proj, down_proj
  Hawkeye sub-modules (always unfrozen via _set_qwen_hawkeye_modules_trainable):
                         pose_tower, pose_projector,
                         scene_tower, scene_projector,
                         moe, moe_projector
Frozen:
  Qwen3-VL ViT (visual encoder)
  Embedding table
```

### 6.4 Position IDs for spliced tokens

After inserting Hawkeye tokens at position `insert_at`, all position IDs at
`insert_at` and beyond are shifted by `num_hawkeye_tokens` (30).  The new
tokens receive sequential position IDs starting from `insert_at`.  This
ensures the 3-dimensional RoPE position encoding remains valid.

### 6.5 Label masking

Inserted Hawkeye tokens always receive `IGNORE_INDEX` in labels so they do not
contribute to the SFT cross-entropy loss.

---

## 7. File Map

```
my_hawkeye/
├── train_mem.py                              ← training entry point
├── eval.py                                   ← evaluation entry point
├── requirements.txt                          ← main pip dependencies (Qwen3VL-first)
├── environment.qwen3vl.yml                   ← conda environment spec
├── llava/
│   ├── model/
│   │   ├── builder.py                        ← load_pretrained_model dispatcher
│   │   ├── hawkeye_modules.py                ← PoseTower, SceneGraphTower, HawkeyeMoE
│   │   └── language_model/
│   │       └── qwen3_vl_hawkeye.py           ← Qwen3VLHawkeyeAdapter (core)
│   └── train/
│       ├── train.py                          ← training loop + LazySupervisedDataset
│       ├── llava_trainer.py                  ← LLaVATrainer + Qwen3VLHawkeyeTrainer
│       └── qwen3vl_data.py                   ← preprocess_qwen3vl_visual, collate_qwen3vl_batch
├── scripts/
│   └── qwen3vl/
│       ├── smoke_infer.py                    ← single-video inference test
│       ├── train_debug.sh                    ← 2-step training sanity check
│       ├── train_lora.sh                     ← full LoRA training
│       ├── eval_full.sh                      ← full evaluation
│       ├── upgrade_env.sh                    ← install/upgrade dependencies
│       └── README.md                         ← per-script quick reference
├── dataset/
│   ├── new_train.json                        ← training annotation
│   ├── vid_noaudio_split/train_new/          ← training video clips
│   ├── vid_split/test_new/                   ← test video clips
│   ├── pose_feat/                            ← skeleton features
│   ├── graph_feat/ or rel_feat/              ← scene graph features
│   └── saved_result/test_res/               ← evaluation CSV outputs
└── models/
    └── Qwen3-VL-8B-Instruct/                ← Qwen3-VL pretrained weights
```
