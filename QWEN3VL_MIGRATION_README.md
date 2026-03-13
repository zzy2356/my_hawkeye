# Hawkeye × Qwen3-VL-8B-Instruct 迁移指南

本文档描述如何将 Hawkeye 项目的多模态基模替换为 Qwen3-VL-8B-Instruct，并提供从环境搭建到完整评估的全流程命令，以及每一步的检查方法和常见问题排查。

---

## 工作环境

| 项目 | 路径 |
|------|------|
| 工作区 | `/home/djingwang/zyzhu/` |
| 项目代码 | `/home/djingwang/zyzhu/my_hawkeye/` |
| Qwen3-VL 模型 | `/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct` |
| GPU | 49 GB 显卡 |

---

## 第零步：环境准备

### 0.1 创建并激活 Conda 环境

```bash
conda create -n hawkeye_qwen3 python=3.10 -y
conda activate hawkeye_qwen3
```

### 0.2 安装依赖

```bash
cd /home/djingwang/zyzhu/my_hawkeye

# PyTorch（CUDA 12.x 对应版本，根据实际 CUDA 版本调整）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Transformers >= 4.57.0（Qwen3-VL 必须）
pip install "transformers>=4.57.0" "tokenizers>=0.21.0" "accelerate>=0.34.0"

# 训练相关
pip install deepspeed==0.17.1
pip install peft==0.17.1

# Flash Attention（加速注意力计算）
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Qwen-VL 工具
pip install "qwen-vl-utils[decord]==0.0.14"

# Hawkeye 额外依赖
pip install torch-geometric fairscale
pip install packaging pandas numpy tqdm
```

### 0.3 检查环境

```bash
# 检查 Python 版本
python --version
# 期望：Python 3.10.x

# 检查 transformers 版本（必须 >= 4.57.0）
python -c "import transformers; print('transformers:', transformers.__version__)"

# 检查 PyTorch + CUDA
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"

# 检查 GPU 显存
nvidia-smi
```

**✅ 成功标志**：
- transformers 版本 >= 4.57.0
- CUDA available: True
- GPU 显存 >= 49 GB

**❌ 失败排查**：
- `transformers < 4.57.0`：运行 `pip install "transformers>=4.57.0"` 升级
- CUDA not available：检查 `nvcc --version` 和 torch 安装的 CUDA 版本是否匹配
- flash-attn 编译失败：见末尾常见问题章节

---

## 第一步：模型加载测试

### 1.1 单独测试 Qwen3-VL 基础模型加载

```bash
cd /home/djingwang/zyzhu/my_hawkeye

python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

model_path = '/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct'

print('Loading processor...')
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print('Processor loaded:', type(processor))

print('Loading model...')
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
print('Model loaded:', type(model))
print('Hidden size:', model.config.text_config.hidden_size)
print('GPU memory:', round(torch.cuda.memory_allocated() / 1024**3, 2), 'GB')
"
```

### 1.2 测试 Hawkeye Adapter 包装

```bash
python -c "
import sys
sys.path.insert(0, '.')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch

model_path = '/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct'
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(
    model_path, None, model_name, device='cuda'
)
print('Adapter type:', type(model).__name__)
print('Hidden size:', model._hidden_size)
print('pose_projector:', model.pose_projector)
print('scene_projector:', model.scene_projector)
print('context_len:', context_len)
print('GPU memory:', round(torch.cuda.memory_allocated() / 1024**3, 2), 'GB')
"
```

**✅ 成功标志**：
- Adapter type: `Qwen3VLHawkeyeAdapter`
- Hidden size: 3584（Qwen3-VL-8B）
- pose_projector: `Linear(in_features=85, out_features=3584, bias=True)`
- GPU 显存占用约 16–18 GB

**❌ 失败排查**：
- `ModuleNotFoundError: No module named 'transformers'`：`pip install transformers>=4.57.0`
- `Unable to infer hidden size`：确认 model_path 下有 `config.json`
- OOM：见末尾常见问题章节

---

## 第二步：Smoke 推理测试

使用 `scripts/qwen3vl/smoke_infer.py` 进行单视频推理，验证 Adapter 的完整前向路径（包含 pose/scene 辅助特征注入）。

### 2.1 准备测试视频

可以使用仓库中的演示视频，或自备任意 mp4 文件：

```bash
ls /home/djingwang/zyzhu/my_hawkeye/Qwen3-VL/qwen-vl-finetune/demo/videos/
# 或者使用任意短视频
VIDEO_PATH="/home/djingwang/zyzhu/my_hawkeye/Qwen3-VL/qwen-vl-finetune/demo/videos/v_7bUu05RIksU.mp4"
```

### 2.2 运行 Smoke 推理

```bash
cd /home/djingwang/zyzhu/my_hawkeye

export HAWKEYE_MODEL_PATH="/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct"

python scripts/qwen3vl/smoke_infer.py \
  --model-path "${HAWKEYE_MODEL_PATH}" \
  --video-path "Qwen3-VL/qwen-vl-finetune/demo/videos/v_7bUu05RIksU.mp4" \
  --prompt "Please determine whether this video shows abnormal or negative events." \
  --max-new-tokens 64
```

**✅ 成功标志**：
- 输出 `=== Smoke Inference Output ===` 后跟非空文本
- 无 `RuntimeError`、无 `shape mismatch` 错误
- 输出示例：`"This video appears to show normal activity without any obvious signs of abnormal behavior."`

**❌ 失败排查**：
- `FileNotFoundError: Video path not found`：替换为实际存在的 mp4 文件路径
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied`：检查 pose/scene projector 的维度，确认 Bug 2 修复已生效
- `RuntimeError: CUDA out of memory`：添加 `--load-4bit` 参数

---

## 第三步：eval.py 评估测试（小样本验证）

### 3.1 检查数据目录

```bash
# 检查必需目录是否存在
ls -la /home/djingwang/zyzhu/my_hawkeye/dataset/vid_split/test_new/ | head -5
ls -la /home/djingwang/zyzhu/my_hawkeye/dataset/pose_feat/test/ | head -5
ls -la /home/djingwang/zyzhu/my_hawkeye/dataset/graph_feat/test/ | head -5

# 检查输出目录
mkdir -p /home/djingwang/zyzhu/my_hawkeye/dataset/saved_result/test_res/
```

### 3.2 检查数据格式（以第一个样本为例）

```bash
python -c "
import numpy as np, os

test_dir = 'dataset/vid_split/test_new'
folder = os.listdir(test_dir)[0]
print('Sample folder:', folder)

pose_dir = f'dataset/pose_feat/test/{folder}'
pose_files = sorted(os.listdir(pose_dir))[:1]
for f in pose_files:
    data = np.load(os.path.join(pose_dir, f))
    print(f'pose {f}: shape={data.shape}')  # 期望: (N, 17, 5)

graph_dir = f'dataset/graph_feat/test/{folder}'
graph_files = sorted(os.listdir(graph_dir))[:1]
for f in graph_files:
    data = np.load(os.path.join(graph_dir, f))
    print(f'graph {f}: shape={data.shape}')  # 期望: (N, 353)
"
```

### 3.3 运行评估（先用环境变量限制样本数，或直接运行）

```bash
cd /home/djingwang/zyzhu/my_hawkeye

export HAWKEYE_MODEL_PATH="/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct"

python eval.py
```

### 3.4 检查输出

```bash
# 查看生成的 CSV 文件
ls -la dataset/saved_result/test_res/
head -3 dataset/saved_result/test_res/*.csv
```

**✅ 成功标志**：
- `dataset/saved_result/test_res/` 中出现 `.csv` 文件
- CSV 包含 `file` 和 `output` 两列
- `output` 列非空，是合理的文本回答

**❌ 失败排查**：
- `FileNotFoundError: dataset/vid_split/test_new`：确认数据目录路径正确
- `ValueError: Qwen3-VL inference requires processor["qwen"]`：检查模型是否正确加载为 Qwen3VLHawkeyeAdapter
- CSV 的 `output` 列全为空：检查 `_run_qwen3_vl_inference` 返回值

---

## 第四步：训练 Debug 测试（2步）

验证训练数据管道、DataCollator、模型前向传播端到端正确。

### 4.1 检查训练数据

```bash
# 检查训练 JSON 文件
python -c "
import json
with open('dataset/new_train.json') as f:
    data = json.load(f)
print(f'Total samples: {len(data)}')
print('First sample keys:', list(data[0].keys()))
"

# 检查 DeepSpeed 配置
cat scripts/zero2.json
```

### 4.2 运行 Debug 训练（2 steps）

```bash
cd /home/djingwang/zyzhu/my_hawkeye

export HAWKEYE_WORK_ROOT="/home/djingwang/zyzhu"
export HAWKEYE_MODEL_PATH="/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct"

bash scripts/qwen3vl/train_debug.sh
```

### 4.3 检查训练输出

```bash
# 检查 checkpoint 是否保存
ls -la /home/djingwang/zyzhu/output_folder/Hawkeye-Qwen3VL-debug/

# 检查训练日志
cat /home/djingwang/zyzhu/output_folder/Hawkeye-Qwen3VL-debug/trainer_state.json | python -m json.tool | grep "global_step"
```

**✅ 成功标志**：
- 日志中出现 `global_step: 2`
- `output_folder/Hawkeye-Qwen3VL-debug/checkpoint-2/` 目录存在
- 无 `CUDA error`、无 `shape mismatch`

**❌ 失败排查**：
- `replace_llama_attn_with_flash_attn` 报错：确认 `train_mem.py` 中 Qwen3-VL 条件判断已生效
- `KeyError: 'input_ids'`：检查 DataCollator 是否正确返回 Qwen3-VL 格式
- OOM：在 `train_debug.sh` 中添加 `--gradient_checkpointing True`（已默认开启）

---

## 第五步：完整 LoRA 训练

### 5.1 运行完整训练

```bash
cd /home/djingwang/zyzhu/my_hawkeye

export HAWKEYE_WORK_ROOT="/home/djingwang/zyzhu"
export HAWKEYE_MODEL_PATH="/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct"

bash scripts/qwen3vl/train_lora.sh
```

### 5.2 监控训练进度

```bash
# 实时查看 GPU 利用率
watch -n 5 nvidia-smi

# 查看训练日志（另一个终端）
tail -f /home/djingwang/zyzhu/output_folder/Hawkeye-Qwen3VL-lora/trainer_state.json
```

### 5.3 显存估算（Qwen3-VL-8B + LoRA，bfloat16）

| 组件 | 显存估算 |
|------|---------|
| 模型权重（bfloat16） | ~16 GB |
| ViT 视觉编码器 | ~2 GB |
| LoRA 参数 + 优化器状态 | ~8 GB |
| 激活值 + KV Cache | ~8 GB |
| **合计** | **~34–38 GB** ✅（49 GB 可用） |

**✅ 成功标志**：
- 训练 loss 持续下降
- checkpoint 定期保存
- GPU 利用率 > 80%

---

## 第六步：完整评估（使用 LoRA checkpoint）

```bash
cd /home/djingwang/zyzhu/my_hawkeye

# 将训练好的 LoRA checkpoint 路径设为 model_path
export HAWKEYE_MODEL_PATH="/home/djingwang/zyzhu/output_folder/Hawkeye-Qwen3VL-lora/checkpoint-final"

python eval.py
```

### 统计评估结果

```bash
python -c "
import pandas as pd, os, glob

result_dir = 'dataset/saved_result/test_res'
csv_files = glob.glob(f'{result_dir}/*.csv')
print(f'Total result files: {len(csv_files)}')

all_outputs = []
for f in csv_files:
    df = pd.read_csv(f)
    all_outputs.extend(df['output'].tolist())

non_empty = [x for x in all_outputs if str(x).strip()]
print(f'Total samples: {len(all_outputs)}')
print(f'Non-empty outputs: {len(non_empty)} ({100*len(non_empty)/max(len(all_outputs),1):.1f}%)')
"
```

---

## 常见问题排查

### OOM（显存不足）

```bash
# 方案 1：使用 4-bit 量化（推理）
python scripts/qwen3vl/smoke_infer.py --load-4bit ...

# 方案 2：减小 batch size
# 在训练脚本中设置 --per_device_train_batch_size 1

# 方案 3：增加梯度累积步数
# --gradient_accumulation_steps 4

# 方案 4：减小 model_max_length
# --model_max_length 2048
```

### transformers 版本不兼容

```bash
# 检查当前版本
python -c "import transformers; print(transformers.__version__)"

# 升级
pip install "transformers>=4.57.0" "tokenizers>=0.21.0" "accelerate>=0.34.0"

# 或者使用 upgrade_env.sh
bash scripts/qwen3vl/upgrade_env.sh
```

### flash-attn 编译失败

```bash
# 安装编译依赖
pip install ninja packaging

# 使用预编译版本（根据 CUDA 版本选择）
# 到 https://github.com/Dao-AILab/flash-attention/releases 下载对应版本
# 例如：
pip install flash_attn-2.7.4.post1+cu124torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 如果 flash-attn 无法安装，可以不安装（性能下降，但可以运行）
# 在训练脚本中去掉相关选项
```

### 数据路径找不到

```bash
# 检查数据目录结构
find dataset/ -name "*.npy" | head -10
find dataset/ -name "*.mp4" | head -10

# 检查具体路径
ls dataset/vid_split/test_new/
ls dataset/pose_feat/test/
ls dataset/graph_feat/test/
```

### Qwen3-VL 模型文件不完整

```bash
# 检查模型文件
ls -la /home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct/
# 期望看到：config.json, *.safetensors, tokenizer.json, processor_config.json 等

# 检查 config
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct', trust_remote_code=True)
print(cfg)
"
```

### pose/scene 维度不匹配

如果遇到 `mat1 and mat2 shapes cannot be multiplied`，检查：

```bash
python -c "
import torch
from llava.model.language_model.qwen3_vl_hawkeye import Qwen3VLHawkeyeAdapter

# 模拟 adapter 的 encode_aux_features
class MockConfig:
    hidden_size = 3584
    pad_token_id = 0

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.embed = torch.nn.Embedding(1000, 3584)
    def get_input_embeddings(self):
        return self.embed
    def parameters(self):
        return self.embed.parameters()

adapter = Qwen3VLHawkeyeAdapter(MockModel())
pose = torch.zeros(5, 17, 5)
scene = torch.zeros(5, 353)
feat = adapter.encode_aux_features(pose, scene)
print('aux feature shape:', feat.shape)  # 期望: (3584,)
"
```

---

## 架构说明

```
视频输入 → Qwen3-VL ViT → Qwen3-VL LLM
                                ↑
pose_feat (5,17,5) → flatten(85) → pose_projector(85→3584) ─┐
                                                              ├─ moe_projector → aux_token(3584)
scene_feat (N,353) → mean(353) → scene_projector(353→3584) ─┘
                                                              ↓
                                          [aux_token, <video_tokens>, text_tokens]
```

辅助特征（pose/scene）被投影为单个 token，并拼接到输入序列的最前面，随后与视频和文本 tokens 一起送入 Qwen3-VL 的 transformer layers。

---

## 快速参考命令

```bash
# 激活环境
conda activate hawkeye_qwen3
cd /home/djingwang/zyzhu/my_hawkeye

# 设置环境变量
export HAWKEYE_WORK_ROOT="/home/djingwang/zyzhu"
export HAWKEYE_MODEL_PATH="/home/djingwang/zyzhu/models/Qwen3-VL-8B-Instruct"

# Smoke 测试
python scripts/qwen3vl/smoke_infer.py --model-path "${HAWKEYE_MODEL_PATH}" --video-path <video.mp4>

# 评估
python eval.py

# Debug 训练
bash scripts/qwen3vl/train_debug.sh

# 完整 LoRA 训练
bash scripts/qwen3vl/train_lora.sh
```
