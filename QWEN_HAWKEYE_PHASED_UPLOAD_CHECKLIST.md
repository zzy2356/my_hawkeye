# Qwen-Hawkeye Phased Upload Checklist

## 1. Roadmap

### Stage A: Data readiness
- Align dataset layout with the current Qwen-Hawkeye training and evaluation entrypoints.
- Confirm real pose and scene `.npy` features exist and match expected shapes.
- Verify `new_train.json`, train/test video folders, and train/test feature folders.

### Stage B: Base-model smoke validation
- Run Qwen3-VL base model with real `pose/scene` features.
- Confirm Qwen visual token materialization works.
- Confirm Hawkeye `pose -> scene graph -> MoE -> splice` path works.

### Stage C: Two-step debug training
- Run 2 training steps to validate dataset loader, collator, forward, backward, loss, and checkpoint saving.
- Confirm Hawkeye non-LoRA modules are trainable and saved.

### Stage D: Full LoRA training
- Train LoRA on top of Qwen3-VL while keeping Hawkeye modules trainable.
- Save LoRA adapter + `non_lora_trainables.bin` + `hawkeye_config.json`.

### Stage E: Reload and inference
- Reload from `checkpoint + model_base`.
- Re-run smoke inference and compare debug outputs with base-only inference.

### Stage F: Small-sample evaluation
- Run task evaluation on IASDig folders with real pose/scene features.
- Check CSV outputs and output format stability.

### Stage G: Full integration hardening
- Reduce "adapter/wrapper" feel by making Qwen-Hawkeye the primary multimodal path.
- Decide whether to retire or isolate the legacy LLaVA path.
- If strict paper parity is required, migrate routing semantics and deeper fusion behavior that are still simplified.


## 2. Current Integration Status

### What is already using Qwen3-VL
- `llava/model/language_model/qwen3_vl_hawkeye.py`
  - Loads Qwen3-VL as the multimodal backbone.
  - Materializes Qwen visual embeddings from `<video>` placeholders.
  - Owns the Hawkeye pose tower, scene graph tower, and MoE modules.
  - Splices Hawkeye tokens into the embedding stream before the Qwen text backbone.
- `llava/train/qwen3vl_data.py`
  - Builds Qwen chat-template inputs.
  - Computes `attention_mask` and `position_ids`.
  - Builds label masks for Qwen-style SFT.
- `llava/train/train.py`
  - Detects Qwen3-VL mode.
  - Loads the Qwen-Hawkeye wrapper.
  - Enables LoRA on Qwen attention layers.
  - Keeps Hawkeye non-LoRA modules trainable.
- `llava/model/builder.py`
  - Routes Qwen3-VL model names into the Qwen-Hawkeye loader.
- `scripts/qwen3vl/*`
  - Qwen-specific smoke, debug train, full train, and eval wrappers.

### What is still "wrapper-like"
- The old LLaVA/Hawkeye path still exists in parallel.
- Hawkeye modules live in the wrapper, not inside Qwen3-VL transformer blocks.
- Qwen visual embeddings are materialized first, and Hawkeye tokens are then inserted externally.
- The MoE route is computed from pose/scene tokens only; it is not conditioned on Qwen visual hidden states.


## 3. B-H MoE and Dynamic Routing Status

### Is the original B-H MoE used?
- Partially, structurally yes.
- The current `HawkeyeMoE` in `llava/model/hawkeye_modules.py` still receives `pose_tokens + scene_tokens`, routes them across experts, and outputs a fixed Hawkeye token sequence.
- The current route is produced by `routers(fused)` over fused pose/scene tokens.

### Is it aligned with Qwen3-VL DeepStack ViT for dynamic routing?
- No, not in the current code.
- Qwen3-VL visual features are used only to materialize the video token embeddings.
- The MoE router does not currently consume Qwen visual features, Qwen vision intermediate layers, or any DeepStack-style multilevel visual state.
- Therefore there is no current "Qwen visual feature guided dynamic routing alignment" between DeepStack ViT outputs and the B-H MoE route.

### Practical conclusion
- Current route source:
  - `pose_values -> PoseTower`
  - `scene_values -> SceneGraphTower`
  - `concat(pose_tokens, scene_tokens) -> HawkeyeMoE`
- Current Qwen visual role:
  - `pixel_values_videos -> Qwen visual encoder -> video embeddings`
- Current fusion:
  - `video embeddings + Hawkeye MoE tokens -> same external embedding stream`
- Not yet present:
  - `Qwen visual hidden states -> router/gating/alignment signal`


## 4. Fine-Grained Logic Model

### 4.1 Data preparation flow
1. `scripts/prepare_dataset.py`
   - Reads `dataset/split_train.txt` and `dataset/split_test.txt`.
   - Organizes IASDig videos into:
     - `dataset/vid_noaudio_split/train_new/<folder>/<index>.mp4`
     - `dataset/vid_noaudio_split/test_new/<folder>/<index>.mp4`
   - Creates the expected feature directory layout for:
     - `dataset/pose_feat/train|test`
     - `dataset/rel_feat/train|test`
   - Generates `dataset/new_train.json`.
2. External feature extraction tools
   - HigherHRNet writes `frame_<index>.npy` with shape `(5, 17, 5)`.
   - RelTR writes `frame_<index>.npy` with shape `(5, 353)`.
3. `scripts/verify_dataset.py`
   - Checks train/test video folders.
   - Checks pose feature shape.
   - Checks scene feature shape.
   - Checks JSON/video path consistency.

### 4.2 Training data flow
1. `train_mem.py`
   - Calls `llava.train.train.train()`.
2. `llava/train/train.py`
   - Parses args.
   - Detects Qwen3-VL mode.
   - Loads Qwen-Hawkeye wrapper.
3. `LazySupervisedDataset.__getitem__`
   - Reads one item from `dataset/new_train.json`.
   - Resolves video file from `data_args.video_folder`.
   - Resolves pose file from `dataset/pose_feat/train/<folder>/frame_<index>.npy`.
   - Resolves scene file from `dataset/graph_feat/train/...` or `dataset/rel_feat/train/...`.
   - Calls `preprocess_qwen3vl_visual(...)`.
4. `llava/train/qwen3vl_data.py`
   - Converts conversation into Qwen chat template.
   - Inserts `<video>` as Qwen multimodal message content.
   - Produces:
     - `input_ids`
     - `attention_mask`
     - `position_ids`
     - `labels`
     - `pixel_values_videos`
     - `video_grid_thw`
5. `collate_qwen3vl_batch(...)`
   - Pads text tensors.
   - Concatenates video tensors.
   - Stacks `pose_values` and `scene_values`.
6. `Qwen3VLHawkeyeAdapter.forward(...)`
   - Calls `_prepare_qwen_hawkeye_inputs(...)`.
   - Materializes Qwen visual embeddings from the `<video>` placeholder span.
   - Encodes pose features with `PoseTower`.
   - Encodes scene relation features with `SceneGraphTower`.
   - Routes `pose_tokens + scene_tokens` through `HawkeyeMoE`.
   - Inserts Hawkeye tokens into the embedding stream after the video span.
   - Sends the final `inputs_embeds` into the Qwen text backbone.

### 4.3 Inference flow
1. `scripts/qwen3vl/smoke_infer.py`
   - Loads model through `llava/model/builder.py`.
   - Builds Qwen chat-template input from one video.
   - Optionally loads `pose_npy` and `scene_npy`.
   - Calls `model.generate(...)`.
2. `Qwen3VLHawkeyeAdapter.generate(...)`
   - Uses the same `_prepare_qwen_hawkeye_inputs(...)`.
   - Stores the real post-splice prefix lengths.
3. `smoke_infer.py`
   - Trims generated tokens using `last_prefix_lens`.
   - Prints debug info from `last_debug_info`.

### 4.4 Evaluation flow
1. `eval.py`
   - Loads Qwen-Hawkeye or legacy Hawkeye depending on model path.
   - Iterates over IASDig test folders.
   - Loads:
     - video file
     - pose feature from `dataset/pose_feat/test`
     - scene feature from `dataset/graph_feat/test` or `dataset/rel_feat/test`
   - Runs Qwen or legacy inference branch.
   - Writes folder-level CSV outputs.


## 5. Step-by-Step Upload Checklist

### Step 0: Environment bootstrap
Upload:
```text
environment.qwen3vl.yml
scripts/qwen3vl/README.md
scripts/zero2.json
train_mem.py
```

Call tree:
```text
server shell
└── conda env create/update
    └── train_mem.py
```

### Step 1: Data preparation
Upload:
```text
scripts/
├── prepare_dataset.py
└── verify_dataset.py

dataset/
├── split_train.txt
└── split_test.txt
```

Generated on server:
```text
dataset/
├── new_train.json
├── vid_noaudio_split/
│   ├── train_new/<folder>/<index>.mp4
│   └── test_new/<folder>/<index>.mp4
├── pose_feat/
│   ├── train/<folder>/frame_<index>.npy
│   └── test/<folder>/frame_<index>.npy
└── rel_feat/
    ├── train/<folder>/frame_<index>.npy
    └── test/<folder>/frame_<index>.npy
```

Call tree:
```text
scripts/prepare_dataset.py
├── dataset/split_train.txt
├── dataset/split_test.txt
├── raw TSL/UCF roots
└── dataset/{vid_noaudio_split,pose_feat,rel_feat,new_train.json}

scripts/verify_dataset.py
├── dataset/new_train.json
├── dataset/vid_noaudio_split/train_new
├── dataset/pose_feat/train
└── dataset/{graph_feat|rel_feat}/train
```

### Step 2: Base-model smoke inference with real features
Upload:
```text
scripts/qwen3vl/
└── smoke_infer.py

llava/
├── model/
│   ├── builder.py
│   ├── hawkeye_modules.py
│   └── language_model/qwen3_vl_hawkeye.py
└── train/qwen3vl_data.py
```

Required on server:
```text
models/Qwen3-VL-8B-Instruct/
dataset/vid_noaudio_split/test_new/<folder>/<index>.mp4
dataset/pose_feat/test/<folder>/frame_<index>.npy
dataset/rel_feat/test/<folder>/frame_<index>.npy
```

Call tree:
```text
scripts/qwen3vl/smoke_infer.py
├── llava/model/builder.py
│   └── llava/model/language_model/qwen3_vl_hawkeye.py
│       ├── Qwen3-VL backbone
│       └── llava/model/hawkeye_modules.py
└── llava/train/qwen3vl_data.py
```

### Step 3: Two-step debug training
Upload:
```text
train_mem.py
scripts/qwen3vl/train_debug.sh

llava/
├── train/train.py
├── train/qwen3vl_data.py
└── model/
    ├── builder.py
    ├── hawkeye_modules.py
    └── language_model/qwen3_vl_hawkeye.py
```

Required on server:
```text
models/Qwen3-VL-8B-Instruct/
dataset/new_train.json
dataset/vid_noaudio_split/train_new/
dataset/pose_feat/train/
dataset/rel_feat/train/
```

Call tree:
```text
scripts/qwen3vl/train_debug.sh
└── train_mem.py
    └── llava/train/train.py
        ├── LazySupervisedDataset
        ├── collate_qwen3vl_batch
        ├── load_pretrained_qwen3vl_hawkeye_model
        └── Qwen3VLHawkeyeAdapter.forward
```

### Step 4: Full LoRA training
Upload:
```text
scripts/qwen3vl/train_lora.sh
train_mem.py
llava/train/train.py
llava/train/qwen3vl_data.py
llava/model/builder.py
llava/model/hawkeye_modules.py
llava/model/language_model/qwen3_vl_hawkeye.py
```

Generated on server:
```text
output_folder/Hawkeye-Qwen3VL/
├── checkpoint-*/
│   ├── adapter_config.json
│   ├── adapter_model.*
│   ├── non_lora_trainables.bin
│   └── hawkeye_config.json
└── ...
```

Call tree:
```text
scripts/qwen3vl/train_lora.sh
└── train_mem.py
    └── llava/train/train.py
        ├── LoRA on Qwen attention modules
        ├── trainable Hawkeye modules
        └── save adapter + non_lora_trainables.bin + hawkeye_config.json
```

### Step 5: Reload + smoke inference on checkpoint
Upload:
```text
scripts/qwen3vl/smoke_infer.py
llava/model/builder.py
llava/model/language_model/qwen3_vl_hawkeye.py
llava/model/hawkeye_modules.py
llava/train/qwen3vl_data.py
```

Required on server:
```text
models/Qwen3-VL-8B-Instruct/
output_folder/Hawkeye-Qwen3VL/checkpoint-*/
dataset/vid_noaudio_split/test_new/
dataset/pose_feat/test/
dataset/rel_feat/test/
```

Call tree:
```text
smoke_infer.py
└── builder.py
    └── qwen3_vl_hawkeye.py
        ├── load base model
        ├── load non_lora_trainables.bin
        └── load LoRA adapter
```

### Step 6: Small-sample evaluation
Upload:
```text
eval.py
scripts/qwen3vl/eval_full.sh

llava/
├── model/builder.py
├── model/language_model/qwen3_vl_hawkeye.py
├── model/hawkeye_modules.py
└── train/qwen3vl_data.py
```

Required on server:
```text
models/Qwen3-VL-8B-Instruct/
or checkpoint + base model
dataset/vid_noaudio_split/test_new/
dataset/pose_feat/test/
dataset/rel_feat/test/
```

Call tree:
```text
scripts/qwen3vl/eval_full.sh
└── eval.py
    ├── builder.py
    ├── qwen3vl_data.py
    ├── qwen3_vl_hawkeye.py
    └── CSV outputs under dataset/saved_result/test_res
```


## 6. Where the New Qwen3-VL Modules Are Actually Used

### Qwen visual backbone usage
- `qwen3_vl_hawkeye.py::_run_visual_encoder`
- `qwen3_vl_hawkeye.py::_materialize_qwen_multimodal_embeds`

### New Hawkeye-on-Qwen fusion usage
- `qwen3_vl_hawkeye.py::encode_poses`
- `qwen3_vl_hawkeye.py::encode_scenes`
- `qwen3_vl_hawkeye.py::moe_route`
- `qwen3_vl_hawkeye.py::_build_hawkeye_token_sequences`
- `qwen3_vl_hawkeye.py::_splice_hawkeye_tokens`
- `qwen3_vl_hawkeye.py::_prepare_qwen_hawkeye_inputs`

### Qwen training path usage
- `train.py` Qwen branch
- `qwen3vl_data.py` preprocess + collator

### Qwen inference path usage
- `smoke_infer.py`
- `eval.py`


## 7. Legacy vs Current Fusion Difference

### Legacy LLaVA path
- Video features are encoded by the external LLaVA/LanguageBind video tower.
- Pose/scene MoE tokens are concatenated with video features before placeholder replacement.
- Key reference:
  - `llava/model/llava_arch.py::prepare_inputs_labels_for_multimodal`

### Current Qwen path
- Qwen3-VL internally defines the `<video>` placeholder semantics.
- The wrapper first materializes the video placeholder span into real Qwen visual embeddings.
- Hawkeye MoE tokens are then inserted into the already-materialized embedding stream.
- This preserves the external placeholder-to-embedding fusion pattern, but it is still wrapper-level fusion, not native Qwen block-level fusion.


## 8. What Must Change For "Fully Integrated" Qwen-Hawkeye

- Make Qwen-Hawkeye the default multimodal path and demote legacy LLaVA to a compatibility path.
- Move from "wrapper owns Hawkeye modules" toward a first-class model package/config.
- Decide whether the MoE route should be conditioned on Qwen visual hidden states.
- If strict paper parity is required, revisit the routing math and fusion semantics that remain simplified in `hawkeye_modules.py`.
- Add a true merged checkpoint/export path if deployment should not depend on `base + adapter + non_lora_trainables.bin`.
