# Hawkeye 架构对比 - 可视化总结（上）

## 一、整体架构对比

### 原始架构（LLaVA 版本）

```
┌─────────────────────────────────────────────────────────────────┐
│                    原始 Hawkeye 架构流程图                        │
└─────────────────────────────────────────────────────────────────┘

输入数据
├─ 视频帧 (T, H, W, 3)
├─ 姿态特征 (T, 17, 5)
└─ 场景特征 (T, 353)
    │
    ├─────────────────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         ▼
┌──────────────────────┐                          ┌──────────────────────┐
│  视频编码器          │                          │  姿态编码器          │
│ LanguageBind Video   │                          │  PoseTower           │
│ Tower                │                          │  (Linear 85→4096)    │
│                      │                          │                      │
│ 输出: video_features │                          │ 输出: pose_features  │
│ (T, hidden_size)     │                          │ (T, hidden_size)     │
└──────────────────────┘                          └──────────────────────┘
    │                                                         │
    │                    ┌──────────────────────┐             │
    │                    │  场景编码器          │             │
    │                    │  SceneGraphTower     │             │
    │                    │  (GTN)               │             │
    │                    │                      │             │
    │                    │ 输出: scene_features │             │
    │                    │ (T, hidden_size)     │             │
    │                    └──────────────────────┘             │
    │                                 │                       │
    └─────────────────┬───────────────┴───────────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  MoE 融合模块        │
            │  HawkeyeMoE          │
            │                      │
            │ 输入:                │
            │ - pose_features      │
            │ - scene_features     │
            │                      │
            │ 输出: moe_features   │
            │ (T, hidden_size)     │
            └──────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  占位符融合                  │
        │ prepare_inputs_labels_for_  │
        │ multimodal                  │
        │                             │
        │ 逻辑:                       │
        │ cur_X_features =            │
        │   cat(video_features,       │
        │       moe_features)         │
        │                             │
        │ 在 [VIDEO] 位置替换为:      │
        │ [video_tokens; moe_tokens]  │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  语言模型骨干               │
        │  Vicuna-7B (via LLaVA)      │
        │                             │
        │ 输入: 拼接后的 embedding    │
        │ 输出: logits / 生成文本     │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  IasDig 任务输出            │
        │ - 异常情绪标签              │
        │ - 异常定位信息              │
        └─────────────────────────────┘
```

---

### 当前架构（Qwen3-VL 版本）

```
┌─────────────────────────────────────────────────────────────────┐
│                   当前 Hawkeye 架构流程图                        │
└─────────────────────────────────────────────────────────────────┘

输入数据
├─ 视频帧 (T, H, W, 3)
├─ 姿态特征 (T, 17, 5)
└─ 场景特征 (T, 353)
    │
    ├─────────────────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         ▼
┌──────────────────────┐                          ┌──────────────────────┐
│  Qwen 视觉编码器     │                          │  姿态编码器          │
│ (Qwen3-VL 自带)      │                          │  PoseTower           │
│                      │                          │  (Linear 85→qwen_dim)│
│ 输入: pixel_values   │                          │                      │
│ 输出: video_embeds   │                          │ 输出: pose_tokens    │
│ (T, qwen_hidden_size)│                          │ (T, hidden_size)     │
└──────────────────────┘                          └──────────────────────┘
    │                                                         │
    │                    ┌──────────────────────┐             │
    │                    │  场景编码器          │             │
    │                    │  SceneGraphTower     │             │
    │                    │  (GTN)               │             │
    │                    │                      │             │
    │                    │ 输出: scene_tokens   │             │
    │                    │ (T, hidden_size)     │             │
    │                    └──────────────────────┘             │
    │                                 │                       │
    └─────────────────┬───────────────┴───────────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  MoE 融合模块        │
            │  HawkeyeMoE          │
            │                      │
            │ 输入:                │
            │ - pose_tokens        │
            │ - scene_tokens       │
            │                      │
            │ 输出: hawkeye_tokens │
            │ (T, hidden_size)     │
            └──────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  Qwen3-VL 占位符融合                │
        │  (Qwen3VLHawkeyeAdapter)            │
        │                                     │
        │ 步骤 1: _materialize_qwen_...       │
        │ ├─ 调用 Qwen 视觉编码器             │
        │ └─ 在 video_token_id 位置填入       │
        │    video_embeds                     │
        │                                     │
        │ 步骤 2: _build_hawkeye_...          │
        │ ├─ encode_poses(pose_values)        │
        │ ├─ encode_scenes(scene_values)      │
        │ └─ moe_route(...) → hawkeye_tokens  │
        │                                     │
        │ 步骤 3: _splice_hawkeye_tokens      │
        │ ├─ 找到 video_token_id 的末尾       │
        │ └─ 在该位置后插入 hawkeye_tokens    │
        │                                     │
        │ 最终序列:                           │
        │ [text] + [video_embeds] +           │
        │ [hawkeye_tokens] + [text]           │
        └─────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  语言模型骨干                       │
        │  Qwen3-VL                           │
        │                                     │
        │ 输入: 拼接后的 embedding            │
        │       + attention_mask              │
        │       + position_ids (mRoPE)        │
        │ 输出: logits / 生成文本             │
        └─────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  IasDig 任务输出                    │
        │ - 异常情绪标签                      │
        │ - 异常定位信息                      │
        └─────────────────────────────────────┘
```

---

## 二、占位符融合机制对比

### 原始架构的融合

```
文本 Token 序列
│
├─ [CLS] [text_token_1] [text_token_2] ... [VIDEO] ... [text_token_N] [SEP]
│                                           ↑
│                                    单个占位符
│
▼
prepare_inputs_labels_for_multimodal
│
├─ 编码视频: video_features = video_tower(video_frames)
├─ 编码姿态: pose_features = pose_tower(pose_feat)
├─ 编码场景: scene_features = scene_tower(scene_feat)
├─ MoE 融合: moe_features = moe(pose_features, scene_features)
├─ 拼接: cur_X_features = cat(video_features, moe_features)
│
└─ 替换 [VIDEO] 为 cur_X_features
   │
   ▼
   [CLS] [text_emb_1] [text_emb_2] ... [video_emb_1] [video_emb_2] ... 
                                        [moe_emb_1] [moe_emb_2] ...
                                        [text_emb_N] [SEP]
                                        ↑
                                  一个连续块
```

### 当前架构的融合

```
文本 Token 序列（含 Qwen video_token_id）
│
├─ [CLS] [text_token_1] [text_token_2] ... [video_token_id] [video_token_id] ...
│                                           ↑
│                                    多个连续占位符
│
▼
_materialize_qwen_multimodal_embeds
│
├─ 调用 Qwen 视觉编码器: video_embeds = qwen_visual(pixel_values_videos)
│
└─ 在 video_token_id 位置填入 video_embeds
   │
   ▼
   [CLS] [text_emb_1] [text_emb_2] ... [video_emb_1] [video_emb_2] ...
                                        ↑
                                  Qwen 视觉 embedding
│
▼
_build_hawkeye_token_sequences
│
├─ pose_tokens = encode_poses(pose_values)
├─ scene_tokens = encode_scenes(scene_values)
├─ hawkeye_tokens = moe_route(pose_tokens, scene_tokens)
│
▼
_splice_hawkeye_tokens
│
├─ 找到 video_token_id 的连续段末尾: insert_at = spans[-1][1]
│
└─ 在 insert_at 位置后插入 hawkeye_tokens
   │
   ▼
   [CLS] [text_emb_1] [text_emb_2] ... [video_emb_1] [video_emb_2] ...
                                        [moe_emb_1] [moe_emb_2] ...
                                        [text_emb_N] [SEP]
                                        ↑
                                  一个连续块
```

### 融合同构性验证

```
原始架构占位符处的内容:
┌─────────────────────────────────────┐
│ [video_tokens] + [moe_tokens]       │
│ ↑                  ↑                │
│ 视频特征        场景增强 MoE        │
└─────────────────────────────────────┘

当前架构占位符处的内容:
┌─────────────────────────────────────┐
│ [qwen_video_embeds] + [hawkeye_moe] │
│ ↑                      ↑            │
│ 视频特征            场景增强 MoE    │
└─────────────────────────────────────┘

结论: ✅ 语义完全一致
```

---

## 三、数据流对比

### 原始架构的数据流

```
dataset/new_train.json
    ↓
LazySupervisedDataset
    ├─ 加载视频路径
    ├─ 加载 pose_feat (T, 17, 5)
    ├─ 加载 scene_feat (T, 353)
    └─ 返回 data_dict
    ↓
DataCollatorForSupervisedDataset
    ├─ 构建 input_ids (含 VIDEO 占位符)
    ├─ 构建 batch['images'] = [Xs, poses, scenes, keys]
    └─ 返回 batch
    ↓
LlavaLlamaForCausalLM.forward(images=batch['images'])
    ├─ 调用 prepare_inputs_labels_for_multimodal
    ├─ 融合视频 + MoE
    ├─ 返回 new_input_embeds, new_labels
    └─ 送入 Vicuna 进行 forward
    ↓
Trainer.backward() & optimizer.step()
```

### 当前架构的数据流

```
dataset/new_train.json
    ↓
LazySupervisedDataset (qwen_multimodal=True)
    ├─ 加载视频路径
    ├─ 加载 pose_feat (T, 17, 5)
    ├─ 加载 scene_feat (T, 353)
    ├─ 调用 preprocess_qwen3vl_visual
    │  └─ 返回 pixel_values_videos, video_grid_thw
    └─ 返回 data_dict
    ↓
collate_qwen3vl_batch
    ├─ 构建 input_ids (含 video_token_id)
    ├─ 构建 position_ids (mRoPE)
    ├─ 堆叠 pose_values 和 scene_values
    └─ 返回 batch
    ↓
Qwen3VLHawkeyeAdapter.forward(
    input_ids, pixel_values_videos, pose_values, scene_values, ...
)
    ├─ 调用 _materialize_qwen_multimodal_embeds
    ├─ 调用 _build_hawkeye_token_sequences
    ├─ 调用 _splice_hawkeye_tokens
    ├─ 融合视频 + MoE
    ├─ 返回 loss
    └─ 送入 Qwen 进行 forward
    ↓
Trainer.backward() & optimizer.step()
```

---

## 四、模块对应关系

```
原始架构                          当前架构
─────────────────────────────────────────────────────

LanguageBind Video Tower    →    Qwen3-VL 视觉编码器
Vicuna-7B                   →    Qwen3-VL-8B
LLaVA mm_projector          →    (Qwen 内部处理)

PoseTower                   →    PoseTower (维度适配)
SceneGraphTower             →    SceneGraphTower (保留)
HawkeyeMoE                  →    HawkeyeMoE (保留)

prepare_inputs_labels_for_  →    _materialize_qwen_multimodal_embeds
multimodal                       + _build_hawkeye_token_sequences
                                 + _splice_hawkeye_tokens

LLaVA Trainer               →    Hugging Face Trainer
                                 (标准化)
```

---

## 五、关键改动点

### 1. 视频编码器替换

```
原始:
video_features = LanguageBind_video_tower(video_frames)
                 ↓
                 (T, video_dim) → mm_projector → (T, 4096)

当前:
video_embeds = Qwen_visual_encoder(pixel_values_videos)
               ↓
               (T, qwen_hidden_size)
```

### 2. 占位符融合重构

```
原始:
单个 [VIDEO] 占位符 → 替换为 [video_tokens; moe_tokens]

当前:
多个 [video_token_id] 占位符 → 填充 video_embeds → 在末尾插入 moe_tokens
```

### 3. 数据预处理新增

```
原始:
LLaVA 标准处理

当前:
+ Qwen Chat Template 处理
+ mRoPE position_ids 计算
+ video_grid_thw 处理
```

### 4. 训练框架标准化

```
原始:
LLaVA 自定义 Trainer

当前:
Hugging Face 标准 Trainer
(更易维护和扩展)
```

