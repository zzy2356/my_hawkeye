# Qwen3-VL Migration — 完整运行指南 / Full Run-Book

## 概述 / Overview

本文档是 Hawkeye 项目从 `Vicuna + LanguageBind / Video-LLaVA` 迁移到
`Qwen3-VL` 的完整操作指南，涵盖每一步的命令、预期输出和验证方法。

> **工作区约定** — 本文档中的命令使用以下环境变量，请根据实际服务器路径替换：
> ```bash
> export WORK_ROOT=/path/to/my_hawkeye   # 项目根目录，如 /home/djingwang/zyzhu/my_hawkeye
> export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct  # 模型目录，如 /home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct
> ```
> 若模型在其他位置，只需修改 `MODEL_PATH` 即可，无需改动脚本。

---

## Step 0 — 环境安装与验证 (Environment Setup)

### 0.1 安装依赖

```bash
cd $WORK_ROOT

# Qwen3-VL 核心依赖
bash scripts/qwen3vl/upgrade_env.sh
```

脚本内容等价于：

```bash
pip install -U "transformers>=4.57.0" "accelerate>=0.34.0" "tokenizers>=0.21.0"
pip install -U "qwen-vl-utils[decord]==0.0.14"   # 视频解码加速（可选但推荐）
pip install torch-geometric                       # SceneGraphTower GTN 必需
pip install fairscale                             # LLaVA trainer 内存优化
pip install deepspeed                             # ZeRO-2 分布式训练
pip install -U "peft>=0.9.0"                      # LoRA
pip install mmengine fvcore iopath decord av opencv-python-headless pandas tqdm
```

### 0.2 验证方法

```bash
python -c "
import torch, transformers, deepspeed, peft, torch_geometric
print('transformers:', transformers.__version__)
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('GPU memory (GB):', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1) if torch.cuda.is_available() else 'N/A')
print('all imports OK')
"
```

**期望输出：**
```
transformers: 4.57.x  (≥4.57.0)
torch: 2.6.x
CUDA available: True
GPU: NVIDIA A100 (或其他 49GB 显卡)
GPU memory (GB): 49.x
all imports OK
```

**如果 transformers 版本过低：**
```bash
pip install -U "transformers>=4.57.0"
```

### 0.3 语法检查

```bash
python -m py_compile \
  llava/model/hawkeye_modules.py \
  llava/train/qwen3vl_data.py \
  llava/model/language_model/qwen3_vl_hawkeye.py \
  llava/model/builder.py \
  llava/train/train.py \
  eval.py \
  scripts/qwen3vl/smoke_infer.py && echo "All files compile OK"
```

**期望输出：**`All files compile OK`

---

## Step 1 — 模型加载验证 (Model Load Test)

### 1.1 命令

```bash
cd $WORK_ROOT

python -c "
import sys, torch
sys.path.insert(0, '.')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = '$MODEL_PATH'
model_name = get_model_name_from_path(model_path)
print('Loading model:', model_name)

tokenizer, model, processor_dict, ctx_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device='cuda',
)
print('Model type:', type(model).__name__)
print('Processor keys:', list(processor_dict.keys()))
print('Context len:', ctx_len)
print('Hawkeye modules:')
for name in ['pose_tower', 'pose_projector', 'scene_tower', 'scene_projector', 'moe', 'moe_projector']:
    m = getattr(model, name, None)
    print(f'  {name}: {type(m).__name__ if m else \"MISSING\"}')
print(f'GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB')
"
```

### 1.2 期望输出

```
Loading model: Qwen3-VL-8B-Instruct
Model type: Qwen3VLHawkeyeAdapter
Processor keys: ['qwen', 'image', 'video']
Context len: 32768
Hawkeye modules:
  pose_tower: PoseTower
  pose_projector: Linear
  scene_tower: SceneGraphTower
  scene_projector: Linear
  moe: HawkeyeMoE
  moe_projector: Linear
GPU memory after load: ~16.x GB
```

### 1.3 失败排查

| 症状 | 原因 | 修复 |
|------|------|------|
| `cannot import AutoModelForImageTextToText` | transformers 版本过低 | `pip install -U "transformers>=4.57.0"` |
| `FileNotFoundError: config.json` | 模型路径错误 | 确认路径存在 `ls $MODEL_PATH/` |
| `Model type: Qwen3VLForConditionalGeneration` (非 Adapter) | builder 未走 Qwen3 分支 | 检查 model_name 是否包含 "qwen3" 和 "vl" |
| `MISSING` in Hawkeye modules | 模块未初始化 | 检查 qwen3_vl_hawkeye.py 中 `__init__` |

---

## Step 2 — Smoke 单样本推理 (Smoke Inference)

### 2.1 命令（使用样例视频）

> `Qwen3-VL/qwen-vl-finetune/demo/videos/` 是仓库中随附的示例目录；
> 也可替换为任意本地 mp4 文件路径。

```bash
cd $WORK_ROOT

python scripts/qwen3vl/smoke_infer.py \
  --model-path "$MODEL_PATH" \
  --video-path Qwen3-VL/qwen-vl-finetune/demo/videos/v_7bUu05RIksU.mp4 \
  --print-shapes \
  --max-new-tokens 32
```

### 2.2 命令（使用真实数据集特征）

> 示例使用 `Normal` 类别（IasDig 数据集）。
> 请将类别名替换为实际目录名（如 `Normal`、`Fighting`、`Arrest` 等），
> 帧编号替换为实际存在的 npy 文件编号（如 `frame_1.npy`）。

```bash
# 示例：Normal 类别第 1 个视频
python scripts/qwen3vl/smoke_infer.py \
  --model-path "$MODEL_PATH" \
  --video-path dataset/vid_split/test_new/Normal/1.mp4 \
  --pose-npy  dataset/pose_feat/test/Normal/frame_1.npy \
  --scene-npy dataset/graph_feat/test/Normal/frame_1.npy \
  --print-shapes \
  --max-new-tokens 16
```

### 2.3 期望输出

```
=== Hawkeye Multimodal Shapes ===
torch.cuda.is_available: True
model_device: cuda:0
input_ids: (1, XXX)
pixel_values_videos: (T, 3, H, W)
video_grid_thw: (1, 3)
pose_values: (1, 5, 17, 5)
scene_values: (1, 5, 353)
hawkeye_scene_token_count: 30
=== Hawkeye Debug Info ===
{
  "visual": {"video_placeholder_count": T, ...},
  "hawkeye": [{"status": "hawkeye_spliced", ...}]
}
=== Smoke Inference Output ===
<non-empty generated text>
```

### 2.4 关键检查点

1. `hawkeye_scene_token_count: 30` ← Hawkeye MoE 正确初始化
2. `status: "hawkeye_spliced"` ← Hawkeye tokens 成功注入 embedding 序列
3. 输出文本非空 ← 生成正常
4. 显存占用约 18-22 GB（bfloat16 模型 + KV cache）

### 2.5 失败排查

| 症状 | 原因 | 修复 |
|------|------|------|
| `FileNotFoundError: video` | 视频路径不存在 | 检查路径或用 `ls` 确认 |
| `status: "no_hawkeye_tokens"` | pose/scene 输入为 None | 检查 kwargs 传递 |
| `status: "no_video_span_found"` | input_ids 中无 video_token_id | 检查 chat template 是否包含 `<video>` |
| 输出全为乱码 / 特殊符号 | `last_prefix_lens` 取值错误 | 检查 `model.last_prefix_lens` 是否正确 |
| CUDA OOM | 显存不足 | 添加 `--load-4bit` 参数 |

---

## Step 3 — 评估验证 (Evaluation Test)

### 3.1 命令

```bash
cd $WORK_ROOT

# 方法 A：通过环境变量指定模型路径
export HAWKEYE_MODEL_PATH=$MODEL_PATH
bash scripts/qwen3vl/eval_full.sh

# 方法 B：直接运行
HAWKEYE_MODEL_PATH=$MODEL_PATH python eval.py
```

### 3.2 期望输出

```
100%|██████████| N/N [XX:XX<00:00, ...]
# CSV 文件生成在 dataset/saved_result/test_res/<category>.csv
```

### 3.3 验证方法

```bash
# 检查 CSV 文件是否生成
ls -la dataset/saved_result/test_res/*.csv

# 检查内容格式
head -5 dataset/saved_result/test_res/<first_category>.csv
# 期望：
# file,output
# 0.mp4,No, the video does not show ...
# 1.mp4,...
```

### 3.4 失败排查

| 症状 | 原因 | 修复 |
|------|------|------|
| `Skip missing dataset root` | 测试集路径不存在 | 确认 `dataset/vid_split/test_new/` 或 `dataset/vid_noaudio_split/test_new/` |
| `FileNotFoundError: Scene feature not found` | scene feature 文件缺失 | 检查 `dataset/graph_feat/` 或 `dataset/rel_feat/` |
| CSV 输出为空字符串 | 模型未生成文本 | 检查 `max_new_tokens` 和 `do_sample` 参数 |
| `ValueError: Qwen3-VL inference requires processor['qwen']` | processor 未加载 | 确认 AutoProcessor 可用（transformers>=4.57.0）|

---

## Step 4 — Debug 训练验证（2步）

### 4.1 前置检查

```bash
# 确认训练数据存在
ls dataset/new_train.json
ls dataset/vid_noaudio_split/train_new/ | head -5
ls dataset/pose_feat/train/ | head -3

# 确认 deepspeed 配置存在
ls scripts/zero2.json
```

### 4.2 命令

```bash
cd $WORK_ROOT

export HAWKEYE_WORK_ROOT=$WORK_ROOT
export HAWKEYE_MODEL_PATH=$MODEL_PATH
export HAWKEYE_MODEL_BASE=$MODEL_PATH
bash scripts/qwen3vl/train_debug.sh
```

等价的直接命令：

```bash
deepspeed --master_port=29503 train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --lora_enable True \
  --model_name_or_path $MODEL_PATH \
  --model_base $MODEL_PATH \
  --version v1 \
  --data_path dataset/new_train.json \
  --video_folder dataset/vid_noaudio_split/train_new \
  --image_folder dataset \
  --bf16 True \
  --output_dir output_folder/Hawkeye-Qwen3VL-debug \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 2 \
  --learning_rate 1e-5 \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --report_to none
```

### 4.3 期望输出

```
Loading model: Qwen3-VL-8B-Instruct
Adding LoRA adapters...
trainable params: ...
...
{"loss": X.XX, "learning_rate": ..., "epoch": ..., "step": 1}
{"loss": X.XX, "learning_rate": ..., "epoch": ..., "step": 2}
***** train metrics *****
  train_loss: X.XX
Saving model checkpoint to output_folder/Hawkeye-Qwen3VL-debug/checkpoint-2
```

### 4.4 验证方法

```bash
# 检查 checkpoint 是否生成
ls output_folder/Hawkeye-Qwen3VL-debug/checkpoint-2/
# 期望包含：adapter_model.safetensors, adapter_config.json,
#           non_lora_trainables.bin, tokenizer_config.json

# 检查 non_lora_trainables.bin 中包含 Hawkeye 模块
python -c "
import torch
ckpt = torch.load('output_folder/Hawkeye-Qwen3VL-debug/checkpoint-2/non_lora_trainables.bin', map_location='cpu')
print('Hawkeye keys in non_lora_trainables:')
for k in sorted(ckpt.keys()):
    if any(name in k for name in ['pose', 'scene', 'moe']):
        print(' ', k, ckpt[k].shape)
"
```

### 4.5 失败排查

| 症状 | 原因 | 修复 |
|------|------|------|
| `ModuleNotFoundError: torch_geometric` | 缺少依赖 | `pip install torch-geometric` |
| `RuntimeError: mat1 and mat2 must have the same dtype` | Hawkeye 模块 dtype 不匹配 | 已由 `_ensure_hawkeye_modules_dtype()` 自动修复，检查版本 |
| `CUDA OOM` | 显存不足 | `--gradient_checkpointing True` 已开启；可降 `--model_max_length 2048` |
| `loss = nan` | 学习率过大或数据问题 | 降低 `--learning_rate 5e-6`；检查训练 JSON 格式 |
| deepspeed 进程挂起 | 端口冲突 | 更换 `--master_port` |
| `non_lora_trainables.bin` 不存在 | 保存逻辑未执行 | 确认 `--save_steps 2 --max_steps 2` 对齐 |

---

## Step 5 — 完整 LoRA 训练

### 5.1 前置确认

```bash
# 显存估算（49GB 卡）
# Qwen3-VL-8B weights (bfloat16): ~16 GB
# LoRA adapters + gradients:       ~8  GB
# Optimizer states (ZeRO-2):       ~8  GB
# KV cache + activations:          ~8  GB
# Hawkeye modules:                 ~0.2 GB
# Total estimate:                 ~40 GB ✅ 49GB 卡可运行

nvidia-smi  # 确认 GPU 空闲显存 ≥ 40 GB
```

### 5.2 命令

```bash
cd $WORK_ROOT

export HAWKEYE_WORK_ROOT=$WORK_ROOT
export HAWKEYE_MODEL_PATH=$MODEL_PATH
export HAWKEYE_MODEL_BASE=$MODEL_PATH
bash scripts/qwen3vl/train_lora.sh
```

等价的直接命令：

```bash
deepspeed --master_port=29501 train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --lora_enable True \
  --model_name_or_path $MODEL_PATH \
  --model_base $MODEL_PATH \
  --version v1 \
  --data_path dataset/new_train.json \
  --video_folder dataset/vid_noaudio_split/train_new \
  --image_folder dataset \
  --bf16 True \
  --output_dir output_folder/Hawkeye-Qwen3VL \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --report_to tensorboard
```

### 5.3 训练过程监控

```bash
# 另开终端，实时监控 GPU 使用
watch -n 5 nvidia-smi

# 查看 TensorBoard 日志
tensorboard --logdir output_folder/Hawkeye-Qwen3VL/runs --port 6006
```

### 5.4 验证方法

```bash
# 检查 checkpoint 目录
ls output_folder/Hawkeye-Qwen3VL/
# 期望: checkpoint-200/, checkpoint-400/, ..., checkpoint-final/

# 用训练好的 adapter 跑 smoke inference 验证
python scripts/qwen3vl/smoke_infer.py \
  --model-path output_folder/Hawkeye-Qwen3VL/checkpoint-200 \
  --model-base $MODEL_PATH \
  --video-path dataset/vid_split/test_new/<category>/<video>.mp4 \
  --print-shapes --max-new-tokens 16
```

---

## Step 6 — 使用训练好的模型评估

### 6.1 命令

```bash
export HAWKEYE_MODEL_PATH=$WORK_ROOT/output_folder/Hawkeye-Qwen3VL/checkpoint-XXX
export HAWKEYE_MODEL_BASE=$MODEL_PATH
bash scripts/qwen3vl/eval_full.sh
```

### 6.2 验证方法

```bash
# 检查结果
ls dataset/saved_result/test_res/
python -c "
import pandas as pd, os, glob
csvs = glob.glob('dataset/saved_result/test_res/*.csv')
total = 0
for f in csvs:
    df = pd.read_csv(f)
    total += len(df)
    print(f'{os.path.basename(f)}: {len(df)} rows, empty={df[\"output\"].eq(\"\").sum()}')
print(f'Total samples evaluated: {total}')
"
```

---

## 常见问题速查 / Quick Troubleshooting

### 环境问题

```bash
# Q: ImportError for transformers classes
pip install -U "transformers>=4.57.0"

# Q: torch_geometric missing
pip install torch-geometric

# Q: Cannot decode video
pip install "qwen-vl-utils[decord]==0.0.14"
```

### 显存优化

```bash
# Q: CUDA OOM during training
# A1: 降低序列长度
--model_max_length 2048

# A2: 使用 QLoRA (4-bit)
--bits 4

# A3: 增大梯度累积步数（不增加显存）
--gradient_accumulation_steps 16
```

### 数据问题

```bash
# Q: scene feature not found
# A: 检查 graph_feat / rel_feat 两个路径
find dataset/ -name "frame_1.npy" | head -5

# Q: training JSON parse error
python -c "import json; data=json.load(open('dataset/new_train.json')); print(len(data), 'samples'); print(data[0].keys())"
```

### 推理输出问题

```bash
# Q: 生成文本乱码 / 包含多余特殊字符
# A: 检查 last_prefix_lens
python scripts/qwen3vl/smoke_infer.py ... --print-shapes
# 观察 "true_prefix_lens" 是否合理（通常为几百到几千）
```

---

## 架构对比说明 / Architecture Notes

### 替换前 vs 替换后

| 组件 | 原 (Vicuna/LLaVA) | 新 (Qwen3-VL) |
|------|------------------|--------------|
| LLM backbone | Vicuna-7B | Qwen3-VL-8B |
| 视觉编码器 | LanguageBind Video | Qwen3-VL ViT (内置) |
| Multimodal projector | mm_projector (MLP) | Qwen3-VL 内置 visual router |
| Pose Tower | Linear(85→H) | Linear(85→H) ✅ 保留 |
| Scene Tower | GTN 图网络 | GTN 图网络 ✅ 保留 |
| MoE 融合模块 | HawkeyeMoE | HawkeyeMoE ✅ 保留 |
| 特征注入方式 | prepare_inputs_labels | _splice_hawkeye_tokens ✅ 等价 |

### 未完全等价的部分

1. **MoE 路由公式**：原版用 `sigmoid → normalize`，新版用 `softmax`
2. **标签掩码策略**：新版更准确但依赖 assistant sentinel token 布局
3. **LoRA 覆盖范围**：新版仅覆盖 attention 层（`q/k/v/o_proj`），未覆盖 MLP 层

这些差异不影响运行，但如需严格对齐原论文实现可作为后续优化方向。

---

## 详细架构文档

见 [`PROJECT_LOGIC_FLOW.md`](PROJECT_LOGIC_FLOW.md)。

---
