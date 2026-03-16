# Hawkeye 项目 - 快速参考卡

## 📋 项目信息速查

| 项目 | 内容 |
|------|------|
| **项目名称** | Hawkeye - 隐式异常情绪发现与定位（IasDig） |
| **论文** | ACM MM 2024 |
| **当前阶段** | 架构迁移完成，待 GPU 服务器验证 |
| **主要改动** | LLaVA → Qwen3-VL 迁移 |
| **核心保留** | 场景增强 MoE 机制 + 占位符融合逻辑 |

---

## 🎯 核心指标

### 原始论文性能（LLaVA 版本）

**TSL-300 数据集**
- FNR: 35.82%
- F2: 38.09
- mAP@0.1: 35.24

**UCF-Crime 数据集**
- FNR: 45.66%
- F2: 45.03
- mAP@0.1: 34.41

### 当前版本预期（Qwen3-VL 版本）

**TSL-300 数据集**
- FNR: ≤ 40%（±5%）
- F2: ≥ 36（±2）
- mAP@0.1: ≥ 33（±2）

**UCF-Crime 数据集**
- FNR: ≤ 50%（±5%）
- F2: ≥ 43（±2）
- mAP@0.1: ≥ 32（±2）

---

## 📁 文件导航

### 进度文档

| 文档 | 用途 | 阅读时间 |
|------|------|---------|
| `PROGRESS_REPORT_INDEX.md` | 总索引和快速查阅 | 5 分钟 |
| `PROGRESS_REPORT_BRIEF.md` | 小组内部通知版 | 10 分钟 |
| `PROGRESS_REPORT_PART1.md` | 论文目标和开题报告 | 20 分钟 |
| `PROGRESS_REPORT_PART2.md` | 已修改文件和接口 | 15 分钟 |
| `PROGRESS_REPORT_PART3.md` | 架构对比和融合机制 | 25 分钟 |
| `ARCHITECTURE_COMPARISON_VISUAL.md` | 可视化架构对比 | 15 分钟 |
| `HAWKEYE_ARCHITECTURE_COMPLIANCE.md` | 架构合规性验证 | 20 分钟 |

### 核心代码文件

| 文件 | 说明 | 状态 |
|------|------|------|
| `llava/model/hawkeye_modules.py` | Hawkeye 模块集合 | ✨ 新增 |
| `llava/model/language_model/qwen3_vl_hawkeye.py` | Qwen3-VL 适配器 | ✨ 新增 |
| `llava/train/qwen3vl_data.py` | Qwen 数据预处理 | ✨ 新增 |
| `llava/train/train.py` | 训练脚本 | 🔧 修改 |
| `eval.py` | 推理脚本 | 🔧 修改 |
| `llava/model/llava_arch.py` | LLaVA 架构 | 🔧 修改 |
| `llava/model/builder.py` | 模型加载 | 🔧 修改 |

---

## 🔑 关键接口

### Qwen3VLHawkeyeAdapter（核心类）

```python
class Qwen3VLHawkeyeAdapter(PreTrainedModel):
    # 初始化
    def __init__(self, config, model_args)
    
    # 编码方法
    def encode_poses(self, pose_values) -> torch.Tensor
    def encode_scenes(self, scene_values) -> torch.Tensor
    def moe_route(self, pose_tokens, scene_tokens) -> torch.Tensor
    
    # 融合方法
    def _materialize_qwen_multimodal_embeds(...) -> torch.Tensor
    def _build_hawkeye_token_sequences(...) -> torch.Tensor
    def _splice_hawkeye_tokens(...) -> torch.Tensor
    def _prepare_qwen_hawkeye_inputs(batch) -> dict
    
    # 前向传播
    def forward(input_ids, attention_mask, labels, ...) -> CausalLMOutput
    def generate(input_ids, attention_mask, ...) -> torch.Tensor
```

### 数据预处理函数

```python
# qwen3vl_data.py
def preprocess_qwen3vl_visual(sources, processor, model_args)
def collate_qwen3vl_batch(instances, processor, model_args)
def get_rope_index_3(input_ids, video_grid_thw, processor)
```

### Hawkeye 模块

```python
# hawkeye_modules.py
class PoseTower(nn.Module)
class SceneGraphTower(nn.Module)
class HawkeyeMoE(nn.Module)

def build_pose_tower(model_args) -> PoseTower
def build_scene_tower(model_args) -> SceneGraphTower
def build_moe(model_args) -> HawkeyeMoE
```

---

## 🚀 快速开始

### 1. 环境配置（5 分钟）

```bash
# 创建 Qwen 环境
conda env create -f environment.qwen3vl.yml
conda activate hawkeye-qwen3vl

# 或创建 LLaVA 环境（用于对比）
conda env create -f environment.yml
conda activate hawkeye-llava
```

### 2. 数据准备（已有）

```
dataset/
├── new_train.json
├── vid_split/...
├── pose_feat/...
└── rel_feat/ 或 graph_feat/...
```

### 3. 快速验证（10 分钟）

```bash
# Smoke test - 验证推理
python scripts/qwen3vl/smoke_infer.py

# 或 2-5 步训练调试
bash scripts/qwen3vl/train_debug.sh
```

### 4. 完整训练（2-4 周）

```bash
# TSL-300 数据集
bash scripts/qwen3vl/train_lora.sh --dataset tsl

# UCF-Crime 数据集
bash scripts/qwen3vl/train_lora.sh --dataset ucf
```

### 5. 评估（1 周）

```bash
# 推理和评估
python eval.py --model_path <checkpoint_path>
```

---

## 📊 架构对比一览

### 占位符融合机制

```
原始架构:
[text] + [VIDEO] + [text]
         ↓
    替换为 [video_tokens; moe_tokens]

当前架构:
[text] + [video_token_id]... + [text]
         ↓
    填充 video_embeds，在末尾插入 moe_tokens
    结果: [text] + [video_embeds] + [moe_tokens] + [text]

同构性: ✅ 语义完全一致
```

### 模块对应

```
原始                    当前
─────────────────────────────────
LanguageBind    →    Qwen3-VL 视觉编码器
Vicuna-7B       →    Qwen3-VL-8B
PoseTower       →    PoseTower (维度适配)
SceneGraphTower →    SceneGraphTower (保留)
HawkeyeMoE      →    HawkeyeMoE (保留)
```

---

## ✅ 功能检查清单

### 输入与预处理
- [x] 视频加载和预处理
- [x] 姿态特征加载（5 帧 padding）
- [x] 场景特征加载（5 帧 padding）
- [x] Qwen Chat Template 处理
- [x] mRoPE position_ids 计算

### 多模态编码层
- [x] 视频编码器（Qwen3-VL 自带）
- [x] 姿态编码器（PoseTower）
- [x] 场景编码器（SceneGraphTower + GTN）
- [x] 维度映射和对齐

### 场景增强 MoE
- [x] Pose + Scene 的 GNN 融合
- [x] MoE 路由和专家混合
- [x] 场景增强 token 生成

### 占位符融合
- [x] 视频 embedding 物化（_materialize_qwen_multimodal_embeds）
- [x] MoE token 序列构建（_build_hawkeye_token_sequences）
- [x] MoE token 拼接（_splice_hawkeye_tokens）
- [x] 最终 embedding 序列正确性

### 训练与推理
- [x] 训练脚本适配（train.py）
- [x] 数据 collate 适配（qwen3vl_data.py）
- [x] 推理脚本适配（eval.py）
- [x] 模型加载适配（builder.py）

### 向后兼容性
- [x] LLaVA 链路保留
- [x] 自动路由逻辑
- [x] 两个环境配置

---

## 🔍 常见问题速答

| 问题 | 答案 |
|------|------|
| **原 LLaVA 链路还能用吗？** | ✅ 是的，完全向后兼容 |
| **需要两个环境吗？** | 建议先用两个，后续可合并 |
| **占位符融合与原论文一致吗？** | ✅ 是的，语义完全一致 |
| **性能会下降吗？** | 预期持平或更优，需实验验证 |
| **什么时候能看到结果？** | Smoke test: 1-2 天；完整训练: 2-4 周 |
| **如何快速验证？** | 运行 `python scripts/qwen3vl/smoke_infer.py` |

---

## 📈 进度时间表

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|---------|------|
| 1 | 架构设计与代码框架 | 2 周 | ✅ 完成 |
| 2 | GPU 服务器 smoke test | 1-2 天 | 🔄 进行中 |
| 2 | 数据加载与 forward 调试 | 1 周 | 🔄 进行中 |
| 3 | TSL-300 完整训练 | 2 周 | ⏳ 待进行 |
| 3 | UCF-Crime 完整训练 | 2 周 | ⏳ 待进行 |
| 3 | 性能对比与分析 | 1 周 | ⏳ 待进行 |
| 4 | 优化与发布 | 1 周 | ⏳ 待进行 |

---

## 🎓 相关资源

### 论文
- **Hawkeye**: https://openreview.net/pdf?id=ys3V4jiENk
- **LLaVA**: https://github.com/haotian-liu/LLaVA
- **Qwen3-VL**: https://github.com/QwenLM/Qwen-VL

### 数据集
- **TSL-300**: https://github.com/nku-zhichengzhang/TSL300
- **UCF-Crime**: https://github.com/WaqasSultani/AnomalyDetectionCVPR2018

### 工具
- **HigherHRNet**: 姿态估计
- **RelTR**: 场景图生成

---

## 📞 技术支持

### 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `ModuleNotFoundError: qwen_vl_utils` | 环境缺少依赖 | `pip install qwen-vl-utils` |
| `CUDA out of memory` | 显存不足 | 减小 batch_size 或使用 LoRA |
| `Shape mismatch in pose encoding` | 姿态维度错误 | 检查 pose_feat 的 padding 逻辑 |
| `video_token_id not found` | Qwen tokenizer 问题 | 检查 processor 的初始化 |

### 调试技巧

```python
# 打印 tensor 形状
print(f"video_embeds shape: {video_embeds.shape}")
print(f"moe_tokens shape: {moe_tokens.shape}")

# 检查占位符位置
print(f"video_token_id positions: {(input_ids == video_token_id).nonzero()}")

# 验证融合结果
print(f"final embedding shape: {inputs_embeds.shape}")
```

---

## 📝 文档版本

| 版本 | 日期 | 内容 |
|------|------|------|
| v1.0 | 2024-03-16 | 初始版本 |

---

## 🎯 下一步行动

### 立即行动（今天）
1. 阅读 `PROGRESS_REPORT_BRIEF.md`（10 分钟）
2. 理解占位符融合机制（见 `ARCHITECTURE_COMPARISON_VISUAL.md`）
3. 准备 GPU 服务器环境

### 短期行动（1-2 天）
1. 创建两个 conda 环境
2. 运行 smoke test
3. 进行 2-5 步训练调试

### 中期行动（1-2 周）
1. 完整训练 TSL-300
2. 完整训练 UCF-Crime
3. 收集性能指标

### 长期行动（2-4 周）
1. 性能对比分析
2. 模型优化
3. 发布最终版本

---

**项目状态**: 🟡 进行中（待 GPU 服务器验证）  
**最后更新**: 2024-03-16  
**维护者**: Hawkeye 项目组

