# Hawkeye 项目进度报告（第二部分）

## 五、目前已经修改的文件和接口

### 5.1 核心模型文件

#### 1. `llava/model/hawkeye_modules.py`（新增）
**目的**：集中管理 Hawkeye 的姿态、场景、MoE 模块

**主要接口**：
```python
class PoseTower(nn.Module):
    """姿态编码塔"""
    def forward(self, pose_feat: torch.Tensor) -> torch.Tensor
    # 输入: (batch, 5, 17, 5) → 输出: (batch, 5, hidden_size)

class SceneGraphTower(nn.Module):
    """场景图编码塔（GTN 结构）"""
    def forward(self, scene_feat: torch.Tensor) -> torch.Tensor
    # 输入: (batch, 5, 353) → 输出: (batch, 5, hidden_size)

class HawkeyeMoE(nn.Module):
    """场景增强 MoE 模块"""
    def forward(self, pose_tokens, scene_tokens) -> torch.Tensor
    # 输入: 编码后的姿态和场景 token
    # 输出: 场景增强 token 序列

def build_pose_tower(model_args) -> PoseTower
def build_scene_tower(model_args) -> SceneGraphTower
def build_moe(model_args) -> HawkeyeMoE
```

**修改点**：
- 从 llava_arch.py 中提取并独立实现
- 支持 Qwen3-VL 的 hidden_size 配置
- 保留原 GTN 和 MoE 的核心逻辑

---

#### 2. `llava/model/language_model/qwen3_vl_hawkeye.py`（新增）
**目的**：Qwen3-VL 与 Hawkeye 的适配器

**主要接口**：
```python
class Qwen3VLHawkeyeAdapter(PreTrainedModel):
    """Qwen3-VL + Hawkeye 融合模型"""
    
    def __init__(self, config, model_args):
        # 初始化 Qwen 骨干 + Hawkeye 模块
        self.model = Qwen3VLForConditionalGeneration(...)
        self.pose_tower = build_pose_tower(model_args)
        self.scene_tower = build_scene_tower(model_args)
        self.moe = build_moe(model_args)
    
    def encode_poses(self, pose_values) -> torch.Tensor
        # 编码姿态特征
    
    def encode_scenes(self, scene_values) -> torch.Tensor
        # 编码场景特征
    
    def moe_route(self, pose_tokens, scene_tokens) -> torch.Tensor
        # MoE 路由和融合
    
    def _materialize_qwen_multimodal_embeds(self, input_ids, pixel_values_videos, ...) -> torch.Tensor
        # 显式物化 Qwen 视频 embedding
        # 在 video_token_id 位置填入视频 embedding
    
    def _build_hawkeye_token_sequences(self, pose_values, scene_values) -> torch.Tensor
        # 构建 Hawkeye MoE token 序列
    
    def _splice_hawkeye_tokens(self, inputs_embeds, hawkeye_tokens, ...) -> torch.Tensor
        # 在视频 token 后插入 MoE token
        # 实现占位符处的 [video] + [MoE] 拼接
    
    def _prepare_qwen_hawkeye_inputs(self, batch) -> dict
        # 准备 Qwen + Hawkeye 的输入
        # 调用上述三个方法进行融合
    
    def forward(self, input_ids, attention_mask, labels, position_ids, 
                pixel_values_videos, video_grid_thw, pose_values, scene_values, **kwargs) -> CausalLMOutput
        # 完整的 forward pass
    
    def generate(self, input_ids, attention_mask, position_ids, 
                 pixel_values_videos, video_grid_thw, pose_values, scene_values, **kwargs) -> torch.Tensor
        # 生成接口
```

**关键融合逻辑**：
```
1. 输入: input_ids (含 video_token_id), pixel_values_videos, pose_values, scene_values
2. 步骤 1: _materialize_qwen_multimodal_embeds
   - 调用 Qwen 视觉编码器处理 pixel_values_videos
   - 在 input_ids 中 video_token_id 的位置填入视频 embedding
3. 步骤 2: _build_hawkeye_token_sequences
   - encode_poses(pose_values) → pose_tokens
   - encode_scenes(scene_values) → scene_tokens
   - moe_route(pose_tokens, scene_tokens) → hawkeye_tokens
4. 步骤 3: _splice_hawkeye_tokens
   - 在视频 token 块的末尾插入 hawkeye_tokens
   - 最终序列: [text] + [video_embeds] + [hawkeye_tokens] + [text]
5. 输出: 送入 Qwen 语言骨干进行 forward
```

**修改点**：
- 新增文件，实现 Qwen3-VL 与 Hawkeye 的完整融合
- 占位符处实现"视频特征 + 场景增强 MoE"的同构拼接

---

#### 3. `llava/model/llava_arch.py`（修改）
**修改内容**：
- 移除 `print(Xs[0].shape)` 调试语句（第 ~850 行）
- 保留原有的 `prepare_inputs_labels_for_multimodal` 逻辑，用于 LLaVA 路径

**保留接口**：
```python
def prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, 
                                         past_key_values, labels, images, 
                                         image_sizes=None, X_modalities=None):
    """原 LLaVA 融合逻辑（保留用于对比）"""
    # 输入: X_modalities = (Xs, poses, scenes, keys)
    # 输出: new_input_embeds, new_labels
    # 逻辑: [text] + [video_embeds] + [moe_embeds] + [text]
```

**修改点**：
- 仅移除调试代码，核心逻辑保持不变
- 用于 LLaVA 路径的训练和推理

---

### 5.2 训练数据处理文件

#### 4. `llava/train/qwen3vl_data.py`（新增）
**目的**：Qwen3-VL 的数据预处理和 collate 逻辑

**主要接口**：
```python
def preprocess_qwen3vl_visual(sources, processor, model_args):
    """Qwen3-VL 视觉预处理"""
    # 输入: 对话数据 + 视频路径
    # 输出: input_ids, attention_mask, position_ids, pixel_values_videos, video_grid_thw

def collate_qwen3vl_batch(instances, processor, model_args):
    """Qwen3-VL batch collate"""
    # 输入: 多个样本实例
    # 输出: batch dict 包含:
    #   - input_ids, attention_mask, labels, position_ids
    #   - pixel_values_videos, video_grid_thw
    #   - pose_values, scene_values

def get_rope_index_3(input_ids, video_grid_thw, processor):
    """计算 Qwen3-VL 的 mRoPE 位置索引"""
```

**修改点**：
- 新增文件，实现 Qwen chat template 的处理
- 支持 pose_values 和 scene_values 的 batch 拼接

---

#### 5. `llava/train/train.py`（修改）
**修改内容**：

1. **LazySupervisedDataset 类**（第 ~600-800 行）
   - 新增 `qwen_multimodal` 参数判断
   - 当 `qwen_multimodal=True` 时，调用 `preprocess_qwen3vl_visual`
   - 保留 pose_feat 和 scene_feat 的加载逻辑

2. **DataCollatorForSupervisedDataset 类**（第 ~900-1100 行）
   - 新增 `qwen_multimodal` 分支
   - 当 `qwen_multimodal=True` 时，调用 `collate_qwen3vl_batch`
   - 否则调用原 LLaVA 的 collate 逻辑

3. **train 函数**（第 ~1200+ 行）
   - 新增 `_is_qwen3_vl_model_name` 判断
   - 根据模型名称选择 Qwen 或 LLaVA 路径

**关键修改点**：
```python
# 在 train 函数中
if _is_qwen3_vl_model_name(model_args.model_name_or_path):
    # Qwen3-VL 路径
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        processor=processor,
        qwen_multimodal=True,
        model_args=model_args
    )
else:
    # LLaVA 路径
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        qwen_multimodal=False,
        model_args=model_args
    )
```

---

### 5.3 推理文件

#### 6. `eval.py`（修改）
**修改内容**：

1. **新增 `_run_qwen3_vl_inference` 函数**
   - 使用 Qwen3VLHawkeyeAdapter 进行推理
   - 接收 pose_values 和 scene_values
   - 调用 model.generate(..., pose_values=..., scene_values=...)

2. **保留 `_run_legacy_hawkeye_inference` 函数**
   - 原 LLaVA 推理路径
   - 使用 images=[video_tensor, [pose_values], [scene_values], ["video"]]

3. **main 函数中的路由逻辑**
   ```python
   if model_type == "qwen3vl":
       results = _run_qwen3_vl_inference(...)
   else:
       results = _run_legacy_hawkeye_inference(...)
   ```

**修改点**：
- 新增 Qwen3-VL 推理路径
- 保留 LLaVA 推理路径用于对比

---

#### 7. `scripts/qwen3vl/smoke_infer.py`（新增）
**目的**：快速验证 Qwen3-VL + Hawkeye 的推理

**主要功能**：
```python
def smoke_infer():
    """Smoke test for Qwen3-VL Hawkeye"""
    # 1. 加载模型和 processor
    # 2. 加载一个视频样本
    # 3. 加载对应的 pose 和 scene 特征
    # 4. 调用 model.generate(...)
    # 5. 打印输出和 tensor 形状
```

**修改点**：
- 新增文件，用于快速调试

---

### 5.4 模型加载和构建文件

#### 8. `llava/model/builder.py`（修改）
**修改内容**：

1. **新增 `load_pretrained_model_qwen3vl` 函数**
   ```python
   def load_pretrained_model_qwen3vl(model_path, model_base, model_args):
       """加载 Qwen3-VL Hawkeye 模型"""
       # 1. 加载 Qwen3-VL 基础模型
       # 2. 加载 adapter 权重（如果有）
       # 3. 加载 Hawkeye 模块权重
       # 4. 返回 model, processor, tokenizer
   ```

2. **修改 `load_pretrained_model` 函数**
   - 新增 Qwen3-VL 模型名称判断
   - 根据模型类型调用不同的加载函数

**修改点**：
- 支持 Qwen3-VL 模型的加载
- 保留 LLaVA 模型的加载逻辑

---

### 5.5 配置和环境文件

#### 9. `environment.qwen3vl.yml`（新增）
**内容**：
```yaml
name: hawkeye-qwen3vl
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch::pytorch::2.1.0
  - pytorch::pytorch-cuda=11.8
  - pytorch::torchvision
  - pytorch::torchaudio
  - pip
  - pip:
    - transformers>=4.40.0
    - qwen-vl-utils
    - deepspeed>=0.12.0
    - flash-attn>=2.4.0
    - peft>=0.7.0
    - bitsandbytes>=0.41.0
    - wandb
    - opencv-python
    - numpy
    - scipy
    - scikit-learn
```

**修改点**：
- 新增文件，用于 Qwen3-VL 环境配置

---

### 5.6 文档文件

#### 10. `HAWKEYE_ARCHITECTURE_COMPLIANCE.md`（新增）
**内容**：
- 完整的架构合规性分析
- 数据流图和融合逻辑说明
- 视频 token 组织同构性验证

#### 11. `PROGRESS_REPORT_PART1.md` 和 `PROGRESS_REPORT_PART2.md`（新增）
**内容**：
- 论文目标和设计内容
- 开题报告各部分
- 已修改文件和接口清单

---

## 六、修改接口总结表

| 文件 | 类/函数 | 修改类型 | 说明 |
|------|--------|---------|------|
| hawkeye_modules.py | PoseTower | 新增 | 姿态编码塔 |
| hawkeye_modules.py | SceneGraphTower | 新增 | 场景图编码塔 |
| hawkeye_modules.py | HawkeyeMoE | 新增 | MoE 融合模块 |
| qwen3_vl_hawkeye.py | Qwen3VLHawkeyeAdapter | 新增 | Qwen3-VL 适配器 |
| qwen3_vl_hawkeye.py | _materialize_qwen_multimodal_embeds | 新增 | 视频 embedding 物化 |
| qwen3_vl_hawkeye.py | _splice_hawkeye_tokens | 新增 | MoE token 拼接 |
| llava_arch.py | prepare_inputs_labels_for_multimodal | 修改 | 移除调试代码 |
| qwen3vl_data.py | preprocess_qwen3vl_visual | 新增 | Qwen 视觉预处理 |
| qwen3vl_data.py | collate_qwen3vl_batch | 新增 | Qwen batch collate |
| train.py | LazySupervisedDataset | 修改 | 支持 Qwen 路径 |
| train.py | DataCollatorForSupervisedDataset | 修改 | 支持 Qwen collate |
| eval.py | _run_qwen3_vl_inference | 新增 | Qwen 推理路径 |
| builder.py | load_pretrained_model_qwen3vl | 新增 | Qwen 模型加载 |
| smoke_infer.py | smoke_infer | 新增 | 快速验证脚本 |

---

## 七、代码修改的向后兼容性

### 保留的接口
- ✅ `llava_arch.prepare_inputs_labels_for_multimodal`：原 LLaVA 路径保持不变
- ✅ `train.py` 中的 LLaVA 分支：通过 `_is_qwen3_vl_model_name` 判断
- ✅ `eval.py` 中的 LLaVA 推理：保留 `_run_legacy_hawkeye_inference`

### 新增的接口
- 🆕 Qwen3-VL 专用的数据处理、模型、推理流程
- 🆕 两个环境配置文件（environment.yml 和 environment.qwen3vl.yml）

### 兼容性结论
- **完全向后兼容**：原 LLaVA 链路不受影响
- **并行可用**：两条链路可在不同环境中独立运行
- **可选合并**：如果依赖版本统一，可合并为单一环境

