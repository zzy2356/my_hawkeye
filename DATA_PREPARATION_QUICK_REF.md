# Hawkeye 数据准备快速参考

## 数据流概览

```
TSL-300 / UCF-Crime (原始视频)
    ↓
[Step 1] 视频组织 (prepare_dataset.py --extract-frames)
    ↓
dataset/vid_split/folder/1.mp4, 2.mp4, ...
    ↓
[Step 2] 姿态提取 (extract_poses.py) ← HigherHRNet
    ↓
dataset/pose_feat/train/folder/frame_1.npy (5, 17, 5)
    ↓
[Step 3] 场景提取 (extract_scenes.py) ← RelTR
    ↓
dataset/rel_feat/folder/frame_1.npy (5, 353)
    ↓
[Step 4] 生成 JSON (prepare_dataset.py --generate-json)
    ↓
dataset/new_train.json
    ↓
[Step 5] 训练 (train_debug.sh / train_lora.sh)
```

---

## 关键文件结构

```
dataset/
├── split_train.txt              ← 训练集文件夹列表
├── split_test.txt               ← 测试集文件夹列表
├── new_train.json               ← 训练注解（自动生成）
│
├── vid_split/                   ← 组织后的视频
│   ├── 1_Ekman6_disgust_3/
│   │   ├── 1.mp4
│   │   ├── 2.mp4
│   │   └── ...
│   └── ...
│
├── pose_feat/train/             ← 姿态特征
│   ├── 1_Ekman6_disgust_3/
│   │   ├── frame_1.npy (5, 17, 5)
│   │   ├── frame_2.npy
│   │   └── ...
│   └── ...
│
└── rel_feat/                    ← 场景特征
    ├── 1_Ekman6_disgust_3/
    │   ├── frame_1.npy (5, 353)
    │   ├── frame_2.npy
    │   └── ...
    └── ...
```

---

## 一键执行命令

### 前置条件

```bash
# 安装 HigherHRNet
git clone https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git
cd HigherHRNet-Human-Pose-Estimation
pip install -r requirements.txt
# 下载预训练模型到 models/

# 安装 RelTR
git clone https://github.com/yrcong/RelTR.git
cd RelTR
pip install -r requirements.txt
# 下载预训练模型到 models/
```

### 执行流程

```bash
cd /home/djingwang/zyzhu/my_hawkeye

# Step 1: 组织视频
python scripts/prepare_dataset.py \
  --tsl-root /path/to/TSL-300 \
  --ucf-root /path/to/UCF-Crime \
  --output-dir dataset \
  --extract-frames

# Step 2: 提取姿态特征
python scripts/extract_poses.py

# Step 3: 提取场景特征
python scripts/extract_scenes.py

# Step 4: 生成训练 JSON
python scripts/prepare_dataset.py \
  --output-dir dataset \
  --generate-json

# Step 5: 验证数据
python scripts/verify_dataset.py

# Step 6: 开始训练
bash scripts/qwen3vl/train_debug.sh
```

---

## 特征维度速查表

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| 视频帧 | (H, W, 3) | 原始视频帧 |
| 姿态特征 | (5, 17, 5) | 5帧 × 17个关键点 × (x,y,conf,sx,sy) |
| 场景特征 | (5, 353) | 5帧 × 353维场景图特征 |
| 输入 input_ids | (1, ~12700) | 文本 token 序列 |
| 输出 label | 0 或 1 | 二分类标签 |

---

## 数据验证检查清单

- [ ] `dataset/vid_split/` 包含所有文件夹和视频
- [ ] `dataset/pose_feat/train/` 包含所有 frame_*.npy，形状 (5, 17, 5)
- [ ] `dataset/rel_feat/` 包含所有 frame_*.npy，形状 (5, 353)
- [ ] `dataset/new_train.json` 包含正确数量的注解
- [ ] 所有视频文件夹在 split_train.txt 中有对应条目
- [ ] 特征文件数量与视频文件数量一致

---

## 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `FileNotFoundError: split_train.txt` | 分割文件不存在 | 确保在项目根目录运行 |
| `shape mismatch (5, 17, 5)` | HigherHRNet 输出维度不同 | 检查模型版本，调整脚本 |
| `shape mismatch (5, 353)` | RelTR 输出维度不同 | 检查模型版本，调整脚本 |
| `KeyError: 'conversations'` | JSON 格式错误 | 检查 new_train.json 结构 |
| 特征提取很慢 | 单进程处理 | 使用 multiprocessing 并行化 |

---

## 数据集统计

### TSL-300 数据集

- **总视频数**：~300
- **情感类别**：disgust, joy, fear, anger, sadness, surprise
- **负面情感**：disgust, anger, fear, sadness → label=1
- **正面情感**：joy, surprise → label=0

### UCF-Crime 数据集

- **总视频数**：~1900
- **异常类别**：Abuse, Arrest, Arson, Assault, Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, Vandalism
- **异常视频**：→ label=1
- **正常视频**：Normal_Videos_* → label=0

---

## 下一步

数据准备完成后：

1. **运行 smoke test**：
   ```bash
   python scripts/qwen3vl/smoke_infer.py \
     --model-path models/Qwen3-VL-8B-Instruct \
     --video-path dataset/vid_split/1_Ekman6_disgust_3/1.mp4 \
     --pose-npy dataset/pose_feat/train/1_Ekman6_disgust_3/frame_1.npy \
     --scene-npy dataset/rel_feat/1_Ekman6_disgust_3/frame_1.npy \
     --print-shapes
   ```

2. **运行 debug 训练**：
   ```bash
   bash scripts/qwen3vl/train_debug.sh
   ```

3. **运行完整训练**：
   ```bash
   bash scripts/qwen3vl/train_lora.sh
   ```

4. **评估模型**：
   ```bash
   python eval.py
   ```

