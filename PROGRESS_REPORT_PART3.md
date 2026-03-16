# Hawkeye 项目进度报告（第三部分）

## 八、当前架构图和原始架构的框架对比图

### 8.1 原始 Hawkeye 架构（LLaVA 版本）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         原始 Hawkeye 架构（LLaVA 版本）                        │
└─────────────────────────────────────────────────────────────────────────────┘

输入层
├── 视频帧序列 (T, H, W, 3)
├── 姿态特征 (T, 17, 5)  ← HigherHRNet 提取
└── 场景关系特征 (T, 353)  ← RelTR 提取

                    ↓

多模态编码层
├─ 视频编码器（LanguageBind Video Tower）
│  └─ 输出: (T, video_dim) → mm_projector → (T, hidden_size)
│
├─ 姿态编码器（PoseTower）
│  └─ 输入: (T, 17, 5) → flatten → (T, 85) → Linear(85, 4096) → (T, hidden_size)
│
└─ 场景编码器（SceneGraphTower + GTN）
   └─ 输入: (T, 353) → GTN message passing → (T, hidden_size)

                    ↓

场景增强 MoE 模块
├─ 输入: pose_tokens (T, hidden_size) + scene_tokens (T, hidden_size)
├─ MoE 路由: 计算 pose 和 scene 的混合权重
├─ 专家混合: 多个专家块进行融合
└─ 输出: moe_tokens (T, hidden_size)

                    ↓

占位符融合（prepare_inputs_labels_for_multimodal）
├─ 文本 Token: input_ids 中的文本部分
├─ VIDEO 占位符: 单个特殊 token (-201)
├─ 融合逻辑:
│  ├─ 编码视频: video_features = video_tower(video_frames)
│  ├─ 编码姿态: pose_features = pose_tower(pose_feat)
│  ├─ 编码场景: scene_features = scene_tower(scene_feat)
│  ├─ MoE 融合: moe_features = moe(pose_features, scene_features)
│  ├─ 拼接: cur_X_features = cat(video_features, moe_features)
│  └─ 替换: 在 VIDEO 占位符位置替换为 cur_X_features
└─ 输出: [text_tokens] + [video_tokens; moe_tokens] + [text_tokens]

                    ↓

语言模型骨干（Vicuna-7B via LLaVA）
├─ 输入: 拼接后的 embedding 序列
├─ 处理: Transformer 层进行上下文理解
└─ 输出: logits / 生成的文本

                    ↓

任务输出（IasDig）
├─ 异常情绪标签
└─ 异常定位信息

```

---

### 8.2 当前 Hawkeye 架构（Qwen3-VL 版本）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      当前 Hawkeye 架构（Qwen3-VL 版本）                       │
└─────────────────────────────────────────────────────────────────────────────┘

输入层
├── 视频帧序列 (T, H, W, 3)
├── 姿态特征 (T, 17, 5)  ← HigherHRNet 提取
└── 场景关系特征 (T, 353)  ← RelTR 提取

                    ↓

数据预处理（qwen3vl_data.py）
├─ Qwen Chat Template 处理
├─ 生成 input_ids（含 video_token_id）
├─ 计算 position_ids（mRoPE）
└─ 输出: input_ids, attention_mask, position_ids, pixel_values_videos, video_grid_thw

                    ↓

多模态编码层
├─ 视频编码器（Qwen3-VL 自带视觉编码器）
│  ├─ 输入: pixel_values_videos (T, H, W, 3)
│  ├─ 处理: Qwen 视觉塔 → 视频 embedding
│  └─ 输出: video_embeds (T, qwen_hidden_size)
│
├─ 姿态编码器（PoseTower）
│  └─ 输入: (T, 17, 5) → flatten → (T, 85) → Linear(85, qwen_hidden_size) → (T, hidden_size)
│
└─ 场景编码器（SceneGraphTower + GTN）
   └─ 输入: (T, 353) → GTN message passing → (T, hidden_size)

                    ↓

场景增强 MoE 模块（hawkeye_modules.py）
├─ 输入: pose_tokens (T, hidden_size) + scene_tokens (T, hidden_size)
├─ MoE 路由: 计算 pose 和 scene 的混合权重
├─ 专家混合: 多个专家块进行融合
└─ 输出: moe_tokens (T, hidden_size)

                    ↓

占位符融合（Qwen3VLHawkeyeAdapter）
├─ 步骤 1: _materialize_qwen_multimodal_embeds
│  ├─ 调用 Qwen 视觉编码器处理 pixel_values_videos
│  ├─ 在 input_ids 中 video_token_id 的位置填入视频 embedding
│  └─ 输出: inputs_embeds（含视频 embedding）
│
├─ 步骤 2: _build_hawkeye_token_sequences
│  ├─ encode_poses(pose_values) → pose_tokens
│  ├─ encode_scenes(scene_values) → scene_tokens
│  ├─ moe_route(pose_tokens, scene_tokens) → hawkeye_tokens
│  └─ 输出: hawkeye_tokens (T, hidden_size)
│
└─ 步骤 3: _splice_hawkeye_tokens
   ├─ 找到 input_ids 中 video_token_id 的连续段
   ├─ 在该段末尾插入 hawkeye_tokens
   ├─ 扩展 inputs_embeds 序列
   └─ 输出: [text] + [video_embeds] + [hawkeye_tokens] + [text]

                    ↓

语言模型骨干（Qwen3-VL）
├─ 输入: 拼接后的 embedding 序列 + attention_mask + position_ids
├─ 处理: Qwen Transformer 层进行上下文理解
└─ 输出: logits / 生成的文本

                    ↓

任务输出（IasDig）
├─ 异常情绪标签
└─ 异常定位信息

```

---

### 8.3 架构对比表

| 维度 | 原始架构（LLaVA） | 当前架构（Qwen3-VL） | 变化 |
|------|------------------|-------------------|------|
| **视频编码器** | LanguageBind Video Tower | Qwen3-VL 自带视觉编码器 | 更新 |
| **语言模型** | Vicuna-7B | Qwen3-VL-8B | 升级 |
| **姿态编码** | PoseTower (Linear 85→4096) | PoseTower (Linear 85→qwen_hidden_size) | 维度适配 |
| **场景编码** | SceneGraphTower (GTN) | SceneGraphTower (GTN) | 保留 |
| **MoE 模块** | HawkeyeMoE | HawkeyeMoE | 保留 |
| **占位符融合** | prepare_inputs_labels_for_multimodal | _materialize + _splice | 重构 |
| **融合语义** | [video] + [MoE] 在单个占位符 | [video] + [MoE] 在视频 token 段后 | 同构 |
| **数据预处理** | LLaVA 标准处理 | Qwen Chat Template + mRoPE | 新增 |
| **训练框架** | LLaVA Trainer | Hugging Face Trainer | 标准化 |
| **推理方式** | model.generate(images=...) | model.generate(pose_values=..., scene_values=...) | 接口更新 |

---

### 8.4 关键融合机制对比

#### 原始架构（LLaVA）的融合流程

```
input_ids: [CLS] [text] [VIDEO] [text] [SEP]
                         ↓
                    单个占位符
                         ↓
prepare_inputs_labels_for_multimodal:
  1. video_features = video_tower(video_frames)
  2. pose_features = pose_tower(pose_feat)
  3. scene_features = scene_tower(scene_feat)
  4. moe_features = moe(pose_features, scene_features)
  5. cur_X_features = cat(video_features, moe_features)
  6. 在 [VIDEO] 位置替换为 cur_X_features
                         ↓
new_input_embeds: [CLS] [text] [video_tokens; moe_tokens] [text] [SEP]
                                 ↑
                         一个连续的 embedding 块
```

#### 当前架构（Qwen3-VL）的融合流程

```
input_ids: [CLS] [text] [video_token_id] [video_token_id] ... [text] [SEP]
                         ↓
                    多个连续的占位符
                         ↓
_materialize_qwen_multimodal_embeds:
  1. video_embeds = qwen_visual_encoder(pixel_values_videos)
  2. 在 input_ids 中 video_token_id 的位置填入 video_embeds
                         ↓
inputs_embeds: [CLS] [text] [video_emb_1] [video_emb_2] ... [text] [SEP]
                             ↑
                      Qwen 视觉 embedding
                         ↓
_build_hawkeye_token_sequences:
  1. pose_tokens = encode_poses(pose_values)
  2. scene_tokens = encode_scenes(scene_values)
  3. hawkeye_tokens = moe_route(pose_tokens, scene_tokens)
                         ↓
_splice_hawkeye_tokens:
  1. 找到 video_token_id 的连续段末尾
  2. 在该位置后插入 hawkeye_tokens
  3. 扩展 inputs_embeds 序列
                         ↓
new_input_embeds: [CLS] [text] [video_emb_1] ... [video_emb_N] [hawkeye_tokens] [text] [SEP]
                                ↑                                ↑
                         Qwen 视觉 embedding              Hawkeye MoE tokens
                                └────────────────────────────────┘
                                    占位符处的融合内容
```

---

### 8.5 融合同构性验证

**定义**：两个架构在占位符处的融合内容在语义上是否等价。

**原始架构**：
- 占位符处 = [video_features] + [moe_features]
- 一个连续的 embedding 块

**当前架构**：
- 占位符处 = [qwen_video_embeds] + [hawkeye_moe_tokens]
- 在视频 token 段后拼接 MoE token

**同构性结论**：✅ **是的，两个架构在占位符处的融合内容在语义上是等价的**

两者都实现了：
1. 视频特征的编码和填充
2. 姿态 + 场景的 MoE 融合
3. 在占位符处将两者拼接
4. 送入语言模型进行理解

区别仅在于：
- 原始架构：单个占位符 → 替换为 [video + MoE]
- 当前架构：多个占位符 → 填充视频 → 在末尾插入 MoE

**结论**：当前架构保留了原论文的核心融合机制，只是实现方式因 Qwen3-VL 的架构特点而有所调整。

---

### 8.6 数据流对比

#### 原始架构的数据流

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
    └─ 返回 loss
    ↓
Trainer.train()
```

#### 当前架构的数据流

```
dataset/new_train.json
    ↓
LazySupervisedDataset (qwen_multimodal=True)
    ├─ 加载视频路径
    ├─ 加载 pose_feat (T, 17, 5)
    ├─ 加载 scene_feat (T, 353)
    ├─ 调用 preprocess_qwen3vl_visual
    └─ 返回 data_dict (含 pixel_values_videos, video_grid_thw)
    ↓
collate_qwen3vl_batch
    ├─ 构建 input_ids (含 video_token_id)
    ├─ 构建 position_ids (mRoPE)
    ├─ 堆叠 pose_values 和 scene_values
    └─ 返回 batch
    ↓
Qwen3VLHawkeyeAdapter.forward(input_ids, pixel_values_videos, pose_values, scene_values)
    ├─ 调用 _materialize_qwen_multimodal_embeds
    ├─ 调用 _build_hawkeye_token_sequences
    ├─ 调用 _splice_hawkeye_tokens
    ├─ 融合视频 + MoE
    └─ 返回 loss
    ↓
Trainer.train()
```

---

### 8.7 模块依赖关系图

#### 原始架构的模块依赖

```
llava_arch.py
├─ LlavaMetaModel
│  ├─ video_tower (LanguageBind)
│  ├─ mm_projector
│  ├─ pose_feat (Linear)
│  ├─ pose_projector
│  ├─ scene_tower (GTN)
│  ├─ scene_projector
│  └─ moe (HawkeyeMoE)
│
└─ LlavaMetaForCausalLM
   ├─ prepare_inputs_labels_for_multimodal
   └─ forward

llava_llama.py
└─ LlavaLlamaForCausalLM
   ├─ 继承 LlavaMetaForCausalLM
   └─ 使用 Vicuna 语言模型
```

#### 当前架构的模块依赖

```
hawkeye_modules.py
├─ PoseTower
├─ SceneGraphTower
├─ HawkeyeMoE
├─ build_pose_tower
├─ build_scene_tower
└─ build_moe

qwen3_vl_hawkeye.py
└─ Qwen3VLHawkeyeAdapter
   ├─ self.model (Qwen3VLForConditionalGeneration)
   ├─ self.pose_tower (PoseTower)
   ├─ self.scene_tower (SceneGraphTower)
   ├─ self.moe (HawkeyeMoE)
   ├─ encode_poses
   ├─ encode_scenes
   ├─ moe_route
   ├─ _materialize_qwen_multimodal_embeds
   ├─ _build_hawkeye_token_sequences
   ├─ _splice_hawkeye_tokens
   ├─ _prepare_qwen_hawkeye_inputs
   ├─ forward
   └─ generate

train.py
├─ LazySupervisedDataset
├─ DataCollatorForSupervisedDataset
└─ train (函数)

qwen3vl_data.py
├─ preprocess_qwen3vl_visual
├─ collate_qwen3vl_batch
└─ get_rope_index_3
```

---

### 8.8 性能对比预期

| 指标 | 原始架构（LLaVA） | 当前架构（Qwen3-VL） | 预期变化 |
|------|------------------|-------------------|---------|
| **FNR (TSL)** | 35.82% | ≤ 40% | ±5% |
| **F2 (TSL)** | 38.09 | ≥ 36 | ±2 |
| **mAP@0.1 (TSL)** | 35.24 | ≥ 33 | ±2 |
| **FNR (UCF-Crime)** | 45.66% | ≤ 50% | ±5% |
| **F2 (UCF-Crime)** | 45.03 | ≥ 43 | ±2 |
| **mAP@0.1 (UCF-Crime)** | 34.41 | ≥ 32 | ±2 |
| **推理速度** | 基准 | 预期更快 | +10-20% |
| **显存占用** | 基准 | 预期相近或更低 | ±10% |

---

### 8.9 架构演进总结

```
原始 Hawkeye (LLaVA)
    ↓
    ├─ 优点: 论文原始实现，性能已验证
    ├─ 缺点: 基础模型较旧，视觉编码能力有限
    └─ 局限: 难以升级到更新的 LLM
    
    ↓ 迁移
    
当前 Hawkeye (Qwen3-VL)
    ├─ 优点: 
    │  ├─ 使用更强大的 Qwen3-VL 多模态模型
    │  ├─ 保留原论文的场景增强 MoE 机制
    │  ├─ 占位符融合在语义上同构
    │  └─ 支持更灵活的多模态输入处理
    │
    ├─ 缺点:
    │  ├─ 需要在 GPU 服务器上进行完整验证
    │  ├─ 性能指标需要通过实验确认
    │  └─ 两个环境配置增加维护成本
    │
    └─ 前景:
       ├─ 为后续模型升级奠定基础
       ├─ 支持更多 LLM 骨干的快速迁移
       └─ 可进一步优化场景增强机制

```

