# Hawkeye 架构与 IasDig 流程符合性分析

本文档对照原论文 Hawkeye 的精华架构，逐项检查当前代码是否达到：**输入与预处理**、**多模态编码层**、**场景增强 MoE 模块**、**训练与推理脚本**的完整功能，以及是否保留「文本 Token + VIDEO 占位符 → prepare_inputs_labels_for_multimodal → 拼接 embedding 序列 → 多模态模型」这一套流程与同级内部融合机制。

---

## 1. 输入与预处理

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 视频/帧输入 | ✅ | `LazySupervisedDataset` 通过 `path` 加载视频；Qwen3-VL 路径用 `preprocess_qwen3vl_visual` + `media_path` 处理视频。 |
| 姿态特征 (pose) | ✅ | 从 `dataset/pose_feat/train/{folder}/frame_{id}.npy` 加载，形状 `(T, 17, 5)`，不足 5 帧则 pad 到 5 帧，写入 `data_dict['pose_feat']`。 |
| 场景图特征 (scene) | ✅ | 从 `dataset/graph_feat` 或 `dataset/rel_feat`（`_resolve_scene_feature_path`）加载，形状 `(T, 353)`，同样 pad 到 5 帧，写入 `data_dict['scene_feat']`。 |
| 无 pose/scene 的样本 | ✅ | `mode == 'motion'` 或 `'image'` 时 `pose_feat`/`scene_feat` 置为 `None`，后续编码分支会跳过 MoE，仅用视频特征。 |

**结论**：输入与预处理逻辑完整，支持 video + pose + scene 三路输入及缺省情况。

---

## 2. 多模态编码层

### 2.1 视频编码器

- **LLaVA 路径**：`llava_arch.py` 中 `build_video_tower` → LanguageBind 视频塔；`encode_videos` 使用 `mm_projector` 映射到 LLM 空间。
- **Qwen3-VL 路径**：视频由 Qwen 骨干的视觉编码器处理（`pixel_values_videos` 经 processor 送入 backbone），不经过 `llava_arch` 的 video_tower。

### 2.2 姿态编码器

- **LLaVA**：`llava_arch.py` 中 `pose_feat`（`Linear(85, 4096)`）+ `build_pose_projector`（`Linear(4096, 4096)`）；输入 `(5, 17, 5)` 经 `view(5, -1)` 变为 `(5, 85)` 后编码。
- **Qwen3-VL**：`hawkeye_modules.py` 中 `PoseTower`（`Linear(pose_dim, hidden_size)`）+ `build_pose_projector`，由 `qwen3_vl_hawkeye.py` 的 `encode_poses` 调用。

### 2.3 场景图编码器

- **LLaVA**：`llava_arch.py` 中 `GTN`（`GTNLayer` + MessagePassing）+ `build_scene_tower` / `build_scene_projector`，输入 353 维（probas 51 + sub 151 + obj 151），输出 4096 维。
- **Qwen3-VL**：`hawkeye_modules.py` 中 `SceneGraphTower`（GTN 结构）+ `build_scene_tower` / `build_scene_projector`，在 `qwen3_vl_hawkeye.py` 的 `encode_scenes` 中使用。

**结论**：视频、姿态、场景图三类编码器在两条路径中均存在且被调用，多模态编码层功能达标。

---

## 3. 场景增强 MoE 模块

### 3.1 LLaVA 路径（`llava_arch.py`）

- **MOE 结构**：`MOE` 接收 pose 与 scene 的编码特征，2 个 expert（各 1 层 `TransformerBlock`），router 为 MLP(4096, 4096*4, 2)，输出经 `moe_projector` 映射到 4096 维。
- **融合方式**：在 `prepare_inputs_labels_for_multimodal` 中：
  - `X_features_pose = encode_poses(poses[i])`，`X_features_scene = encode_scenes(scenes[i])`
  - `X_moe_feat = moe_route(X_features_pose, X_features_scene)`
  - **关键**：`cur_X_features = torch.cat((X_features_video[i], X_moe_feat), dim=0)` —— **先视频 token 序列，再 MoE token，在 VIDEO 占位符处整体替换**，实现「视频 + 场景增强」同级融合。

### 3.2 Qwen3-VL 路径（`qwen3_vl_hawkeye.py` + `hawkeye_modules.py`）

- **MoE 结构**：`build_moe`（hidden_size, scene_token_count）与 `build_moe_projector`，`moe_route(pose_tokens, scene_tokens)` 得到场景增强 token 序列。
- **融合方式**：`_build_hawkeye_token_sequences` 为每个样本生成 MoE token；`_splice_hawkeye_tokens` 在 `input_ids` 中定位 `video_token_id` 的连续区间，在 **inputs_embeds 的对应位置上用 MoE token 覆盖**。视频由 Qwen 骨干内部视觉分支处理（`pixel_values_videos`），**不在同一占位符处与 MoE 做 concat**。

**结论**：  
- **Pose + Scene 的 GNN + MoE 场景增强**：两条路径都具备，且都用于 IasDig 的隐式场景建模，同级内部融合机制在「姿态+场景」侧成立。  
- **与论文完全一致的「视频 token 与 MoE token 在占位符处拼接」**：仅在 **LLaVA 路径** 实现（`torch.cat((X_features_video[i], X_moe_feat), dim=0)`）；Qwen3-VL 路径是在占位符处**仅注入 MoE token**，视频走骨干自带视觉通路。若需与论文 100% 一致，Qwen 路径可考虑在占位符处改为「Qwen 视频 token + MoE token」拼接。

---

## 4. 训练与推理脚本

### 4.1 训练

- **数据**：`llava/train/train.py` 中 `LazySupervisedDataset` 提供 `video`/`image`、`pose_feat`、`scene_feat`、`input_ids`、`labels` 等。
- **LLaVA 分支**：`DataCollatorForSupervisedDataset`（`qwen_multimodal=False`）构造  
  `batch['images'] = [Xs, poses, scenes, keys]`，与 `prepare_inputs_labels_for_multimodal` 的 `X_modalities = (Xs, poses, scenes, keys)` 约定一致。
- **Qwen3-VL 分支**：`qwen_multimodal=True` 时使用 `collate_qwen3vl_batch`，产出 `input_ids`、`labels`、`attention_mask`、`position_ids`、`pixel_values_videos`、`pose_values`、`scene_values`；`Qwen3VLHawkeyeAdapter.forward(**batch)` 通过 `_prepare_qwen_hawkeye_inputs` 使用 `pose_values`/`scene_values` 并注入 MoE。
- **模型选择**：`train.py` 中根据 `_is_qwen3_vl_model_name` 选择加载 LLaVA 或 Qwen3-VL Hawkeye，并设置 `data_args.qwen_multimodal` 与对应 data collator。

### 4.2 推理

- **LLaVA（原版 Hawkeye）**：`eval.py` 中 `_run_legacy_hawkeye_inference`：
  - 使用 `DEFAULT_X_TOKEN["VIDEO"] + "\n" + prompt` 与 `X_TOKEN_INDEX["VIDEO"]` 得到「文本 Token + VIDEO 占位符」；
  - `model.generate(..., images=[video_tensor, [pose_values], [scene_values], ["video"]])`，与训练时 `batch['images']` 格式一致，走 `prepare_inputs_labels_for_multimodal`。
- **Qwen3-VL**：`_run_qwen3_vl_inference` 使用 `preprocess_qwen3vl_visual` 得到输入，`model.generate(..., pose_values=..., scene_values=...)`，由 adapter 在占位符处注入 MoE token。

**结论**：训练与推理脚本均支持 LLaVA 与 Qwen3-VL 两套流程，且与各自模型的输入约定一致。

---

## 5. 核心流程保留情况：「文本 Token + VIDEO 占位符 → prepare → 拼接 embedding → LLM」

### 5.1 LLaVA 路径（完全保留）

1. **文本 Token + VIDEO 占位符**：  
   `tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], ...)`，prompt 中含 `DEFAULT_X_TOKEN['VIDEO']`（`<video>`），对应 `input_ids` 中 -201 的 VIDEO 占位符。
2. **prepare_inputs_labels_for_multimodal**：  
   接收 `X_modalities = (Xs, poses, scenes, keys)`（即 `batch['images']`）：
   - 对每个样本：`encode_videos(Xs[i])` → `X_features_video[i]`；若 `poses[i] is not None`，则 `encode_poses`、`encode_scenes`、`moe_route`，得到 `X_moe_feat`，并 **`cur_X_features = cat(X_features_video[i], X_moe_feat)`**。
   - 在 `cur_input_ids` 中定位 VIDEO 占位符位置，用 `cur_X_features` **整体替换**该位置对应的 embedding，得到 `new_input_embeds`。
3. **拼接 embedding 序列**：  
   `new_input_embeds` 为「文本段 embedding + 多模态 token（视频+MoE）+ 后续文本 embedding」的拼接序列。
4. **多模态模型**：  
   `LlavaLlamaForCausalLM` 的 `self.model(..., inputs_embeds=new_input_embeds, ...)`，即 Vicuna 骨干接收拼接后的序列。

**结论**：LLaVA 路径完整保留「文本 + VIDEO 占位符 → prepare → 拼接 embedding → LLM」的精华流程，且实现了**视频特征与 MoE 特征在占位符处的同级拼接**，符合原论文设计。

### 5.2 Qwen3-VL 路径（占位符处仅 MoE，视频走骨干)

1. **文本 Token + 视频占位符**：  
   Qwen processor 生成的 `input_ids` 中含 `video_token_id`（如 151656），即视频占位符。
2. **_prepare_qwen_hawkeye_inputs**：  
   根据 `pose_values`、`scene_values` 构建 `hawkeye_sequences`（每样本的 MoE token）；`_splice_hawkeye_tokens` 在 `input_ids` 中找占位符区间，扩展序列并得到 `hawkeye_mask`；先 `inputs_embeds = get_input_embeddings()(new_input_ids)`，再在 `hawkeye_mask` 为 True 的位置 **覆盖为 MoE token**。
3. **视频**：  
   通过 `pixel_values_videos` 等由 Qwen 骨干内部视觉编码并融合到序列，**不在同一占位符与 MoE 做 concat**。

**结论**：Qwen3-VL 路径保留了「占位符 → 注入场景增强 token → LLM」的逻辑，且 pose/scene GNN + MoE 对 IasDig 有效；但与论文在一点上不同：**占位符处仅为 MoE token，视频与 MoE 未在同一位置拼接**。若需与论文完全一致，需在 Qwen 路径中在占位符处改为「Qwen 视频 token + MoE token」的拼接。

---

## 6. 小结与建议

| 模块 | 是否达到原论文/所需功能 | 备注 |
|------|--------------------------|------|
| 输入与预处理 | ✅ | 视频、pose、scene 加载与 pad 一致；支持无 pose/scene 样本。 |
| 多模态编码层 | ✅ | 视频 / 姿态 / 场景图编码器在 LLaVA 与 Qwen3-VL 路径均存在且被调用。 |
| 场景增强 MoE | ✅ | Pose+Scene 的 GNN+MoE 两条路径均生效；与论文完全一致的「视频+MoE 在占位符处拼接」仅 LLaVA 路径。 |
| 训练与推理脚本 | ✅ | 两套脚本分别支持 LLaVA 与 Qwen3-VL，数据格式与模型入参一致。 |
| 核心流程保留 | ✅（LLaVA）/ ⚠️（Qwen） | LLaVA 完整保留「文本+VIDEO 占位符 → prepare → 拼接 embedding → LLM」；Qwen 保留占位符处 MoE 注入，但视频与 MoE 未在同一占位符拼接。 |

**建议**：  
- 若以**严格复现原论文**为主：优先使用 **LLaVA 路径**训练与推理，当前实现已满足「同级内部融合」与完整输入方法逻辑。  
- 若以 **Qwen3-VL 为骨干**且希望与论文一致：可在 `Qwen3VLHawkeyeAdapter._splice_hawkeye_tokens`（或等价处）改为在占位符处拼接「Qwen 视觉编码器在该位置的 token + MoE token」，而不是仅写入 MoE token。

---

## 7. 小问题（可选修复）

- **调试输出**：`llava_arch.py` 中 `prepare_inputs_labels_for_multimodal` 内有 `print(Xs[0].shape)`，建议改为 `logging` 或删除，避免污染训练日志。
- **LazySupervisedDataset 路径**：pose/scene 的路径写死为 `dataset/pose_feat/train/...`、`dataset/graph_feat` 等，若数据目录结构调整需同步修改或改为可配置。

以上为对当前代码是否达到原论文 Hawkeye 所需功能及流程保留情况的完整分析。
