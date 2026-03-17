# Hawkeye 数据准备完整指南

## 概述

Hawkeye 项目使用两个数据集：
- **TSL-300**：情感识别数据集（IasDig 任务）
- **UCF-Crime**：异常检测数据集

本指南详细说明如何从原始数据集准备出项目所需的格式。

---

## 第一步：获取原始数据集

### TSL-300 数据集

1. 访问 [TSL-300 GitHub](https://github.com/nku-zhichengzhang/TSL300)
2. 下载数据集（包含视频和标签）
3. 解压到本地，例如：`/path/to/TSL-300/`

**TSL-300 原始结构示例：**
```
TSL-300/
├── 1_Ekman6_disgust_3/
│   ├── video.mp4
│   └── ...
├── 2_Ekman6_joy_1308/
│   ├── video.mp4
│   └── ...
└── ...
```

### UCF-Crime 数据集

1. 访问 [UCF-Crime GitHub](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)
2. 下载数据集
3. 解压到本地，例如：`/path/to/UCF-Crime/`

**UCF-Crime 原始结构示例：**
```
UCF-Crime/
├── Abuse/
│   ├── Abuse001_x264.mp4
│   ├── Abuse002_x264.mp4
│   └── ...
├── Arrest/
│   ├── Arrest001_x264.mp4
│   └── ...
└── Normal_Videos_Part_1/
    ├── Normal_Videos_Part_1_x264_001.mp4
    └── ...
```

---

## 第二步：视频组织和分割

### 2.1 理解数据划分

项目已有 `dataset/split_train.txt` 和 `dataset/split_test.txt`，定义了训练/测试集的文件夹划分：

**split_train.txt** 包含：
```
1_Ekman6_disgust_3
2_Ekman6_joy_1308
3_Ekman6_fear_699
...
```

**split_test.txt** 包含：
```
9_CMU_MOSEI_lzVA--tIse0
17_CMU_MOSEI_CbRexsp1HKw
...
```

### 2.2 组织视频文件

运行脚本组织视频：

```bash
cd /home/djingwang/zyzhu/my_hawkeye

python scripts/prepare_dataset.py \
  --tsl-root /path/to/TSL-300 \
  --ucf-root /path/to/UCF-Crime \
  --output-dir dataset \
  --extract-frames
```

**输出结构：**
```
dataset/
├── vid_split/
│   ├── 1_Ekman6_disgust_3/
│   │   ├── 1.mp4
│   │   ├── 2.mp4
│   │   └── ...
│   ├── 2_Ekman6_joy_1308/
│   │   ├── 1.mp4
│   │   └── ...
│   └── ...
└── Ucf/
    └── Ucfcrime_split/
        ├── Abuse028_x264/
        │   ├── 1.mp4
        │   └── ...
        └── ...
```

**关键点：**
- 每个文件夹内的视频按顺序重命名为 `1.mp4`, `2.mp4`, ...
- 这个编号对应后续特征提取的 `frame_1.npy`, `frame_2.npy`, ...

---

## 第三步：特征提取

### 3.1 姿态特征提取（Pose Features）

使用 **HigherHRNet** 提取人体姿态关键点。

#### 安装 HigherHRNet

```bash
# 克隆仓库
git clone https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git
cd HigherHRNet-Human-Pose-Estimation

# 安装依赖
pip install -r requirements.txt

# 下载预训练模型
# 按照官方指南下载 pose_higher_hrnet_w32_512.pth
```

#### 提取姿态特征

创建脚本 `scripts/extract_poses.py`：

```python
import cv2
import numpy as np
import os
from pathlib import Path
import torch
# 导入 HigherHRNet 模型（根据实际安装路径调整）

def extract_pose_from_video(video_path, model, num_frames=5):
    """
    从视频中提取 num_frames 帧的姿态特征
    
    Returns:
        pose_features: (num_frames, 17, 5)
        - 17: 人体关键点数量
        - 5: (x, y, confidence, scale_x, scale_y)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 均匀采样 num_frames 帧
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    pose_features = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # 如果读取失败，用零填充
            pose_features.append(np.zeros((17, 5)))
            continue
        
        # 使用 HigherHRNet 推理
        # 这里需要根据实际模型接口调整
        pose = model.inference(frame)  # 返回 (17, 5)
        pose_features.append(pose)
    
    cap.release()
    return np.array(pose_features)  # (num_frames, 17, 5)

# 主程序
video_dir = "dataset/vid_split"
output_dir = "dataset/pose_feat/train"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
model = load_higher_hrnet_model("path/to/pose_higher_hrnet_w32_512.pth")

for folder in os.listdir(video_dir):
    folder_path = os.path.join(video_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)
    
    for video_file in sorted(os.listdir(folder_path)):
        if not video_file.endswith('.mp4'):
            continue
        
        video_path = os.path.join(folder_path, video_file)
        frame_idx = int(video_file.split('.')[0])
        
        pose_features = extract_pose_from_video(video_path, model, num_frames=5)
        output_path = os.path.join(output_folder, f"frame_{frame_idx}.npy")
        
        np.save(output_path, pose_features)
        print(f"Saved: {output_path} (shape: {pose_features.shape})")
```

运行提取：

```bash
python scripts/extract_poses.py
```

**输出结构：**
```
dataset/pose_feat/train/
├── 1_Ekman6_disgust_3/
│   ├── frame_1.npy  (shape: 5, 17, 5)
│   ├── frame_2.npy
│   └── ...
├── 2_Ekman6_joy_1308/
│   ├── frame_1.npy
│   └── ...
└── ...
```

### 3.2 场景图特征提取（Scene Features）

使用 **RelTR** 提取对象关系特征。

#### 安装 RelTR

```bash
# 克隆仓库
git clone https://github.com/yrcong/RelTR.git
cd RelTR

# 安装依赖
pip install -r requirements.txt

# 下载预训练模型
# 按照官方指南下载模型
```

#### 提取场景特征

创建脚本 `scripts/extract_scenes.py`：

```python
import cv2
import numpy as np
import os
import torch
# 导入 RelTR 模型

def extract_scene_from_video(video_path, model, num_frames=5):
    """
    从视频中提取 num_frames 帧的场景图特征
    
    Returns:
        scene_features: (num_frames, 353)
        - 353: 场景图特征维度（根据 RelTR 输出调整）
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    scene_features = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            scene_features.append(np.zeros(353))
            continue
        
        # 使用 RelTR 推理
        scene = model.inference(frame)  # 返回 (353,)
        scene_features.append(scene)
    
    cap.release()
    return np.array(scene_features)  # (num_frames, 353)

# 主程序
video_dir = "dataset/vid_split"
output_dir = "dataset/rel_feat"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
model = load_reltr_model("path/to/reltr_model.pth")

for folder in os.listdir(video_dir):
    folder_path = os.path.join(video_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)
    
    for video_file in sorted(os.listdir(folder_path)):
        if not video_file.endswith('.mp4'):
            continue
        
        video_path = os.path.join(folder_path, video_file)
        frame_idx = int(video_file.split('.')[0])
        
        scene_features = extract_scene_from_video(video_path, model, num_frames=5)
        output_path = os.path.join(output_folder, f"frame_{frame_idx}.npy")
        
        np.save(output_path, scene_features)
        print(f"Saved: {output_path} (shape: {scene_features.shape})")
```

运行提取：

```bash
python scripts/extract_scenes.py
```

**输出结构：**
```
dataset/rel_feat/
├── 1_Ekman6_disgust_3/
│   ├── frame_1.npy  (shape: 5, 353)
│   ├── frame_2.npy
│   └── ...
├── 2_Ekman6_joy_1308/
│   ├── frame_1.npy
│   └── ...
└── ...
```

---

## 第四步：生成训练 JSON

运行脚本生成训练注解文件：

```bash
python scripts/prepare_dataset.py \
  --output-dir dataset \
  --generate-json
```

**输出：** `dataset/new_train.json`

```json
[
  {
    "path": "1_Ekman6_disgust_3/1.mp4",
    "label": "1",
    "mode": "train",
    "conversations": [
      {
        "from": "human",
        "value": "Please determine whether the emotional attributes of the video are negative or not. If negative, answer 1, else answer 0. The answer should just contain 0 or 1 without other contents.\n<video>"
      },
      {
        "from": "gpt",
        "value": "1"
      }
    ]
  },
  ...
]
```

---

## 第五步：验证数据完整性

创建验证脚本 `scripts/verify_dataset.py`：

```python
import os
import numpy as np
from pathlib import Path

def verify_dataset(dataset_root="dataset"):
    """验证数据集完整性"""
    
    issues = []
    
    # 检查视频文件
    vid_split = os.path.join(dataset_root, "vid_split")
    if not os.path.exists(vid_split):
        issues.append(f"Missing: {vid_split}")
    else:
        video_count = sum(len([f for f in os.listdir(os.path.join(vid_split, d)) 
                              if f.endswith('.mp4')])
                         for d in os.listdir(vid_split) 
                         if os.path.isdir(os.path.join(vid_split, d)))
        print(f"✓ Found {video_count} video files")
    
    # 检查姿态特征
    pose_feat = os.path.join(dataset_root, "pose_feat", "train")
    if not os.path.exists(pose_feat):
        issues.append(f"Missing: {pose_feat}")
    else:
        pose_count = sum(len([f for f in os.listdir(os.path.join(pose_feat, d)) 
                             if f.endswith('.npy')])
                        for d in os.listdir(pose_feat) 
                        if os.path.isdir(os.path.join(pose_feat, d)))
        print(f"✓ Found {pose_count} pose feature files")
        
        # 检查形状
        sample_pose = np.load(os.path.join(pose_feat, os.listdir(pose_feat)[0], 
                                          os.listdir(os.path.join(pose_feat, os.listdir(pose_feat)[0]))[0]))
        if sample_pose.shape != (5, 17, 5):
            issues.append(f"Wrong pose shape: {sample_pose.shape}, expected (5, 17, 5)")
        else:
            print(f"✓ Pose features shape correct: {sample_pose.shape}")
    
    # 检查场景特征
    scene_feat = os.path.join(dataset_root, "rel_feat")
    if not os.path.exists(scene_feat):
        issues.append(f"Missing: {scene_feat}")
    else:
        scene_count = sum(len([f for f in os.listdir(os.path.join(scene_feat, d)) 
                              if f.endswith('.npy')])
                         for d in os.listdir(scene_feat) 
                         if os.path.isdir(os.path.join(scene_feat, d)))
        print(f"✓ Found {scene_count} scene feature files")
        
        # 检查形状
        sample_scene = np.load(os.path.join(scene_feat, os.listdir(scene_feat)[0], 
                                           os.listdir(os.path.join(scene_feat, os.listdir(scene_feat)[0]))[0]))
        if sample_scene.shape != (5, 353):
            issues.append(f"Wrong scene shape: {sample_scene.shape}, expected (5, 353)")
        else:
            print(f"✓ Scene features shape correct: {sample_scene.shape}")
    
    # 检查 JSON
    json_file = os.path.join(dataset_root, "new_train.json")
    if not os.path.exists(json_file):
        issues.append(f"Missing: {json_file}")
    else:
        import json
        with open(json_file) as f:
            data = json.load(f)
        print(f"✓ Found {len(data)} training annotations")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ Dataset verification passed!")
        return True

if __name__ == "__main__":
    verify_dataset()
```

运行验证：

```bash
python scripts/verify_dataset.py
```

---

## 完整流程总结

```bash
# 1. 组织视频
python scripts/prepare_dataset.py \
  --tsl-root /path/to/TSL-300 \
  --ucf-root /path/to/UCF-Crime \
  --output-dir dataset \
  --extract-frames

# 2. 提取姿态特征（需要 HigherHRNet）
python scripts/extract_poses.py

# 3. 提取场景特征（需要 RelTR）
python scripts/extract_scenes.py

# 4. 生成训练 JSON
python scripts/prepare_dataset.py \
  --output-dir dataset \
  --generate-json

# 5. 验证数据完整性
python scripts/verify_dataset.py

# 6. 开始训练
bash scripts/qwen3vl/train_debug.sh
```

---

## 常见问题

### Q: 特征提取很慢怎么办？
A: 可以使用多进程并行提取。修改 `extract_poses.py` 和 `extract_scenes.py` 使用 `multiprocessing` 或 `joblib`。

### Q: 特征维度不对怎么办？
A: 检查 HigherHRNet 和 RelTR 的输出维度，根据实际情况调整脚本中的 `(17, 5)` 和 `353`。

### Q: 如何只用 TSL-300 或只用 UCF-Crime？
A: 修改 `split_train.txt` 和 `split_test.txt`，只保留对应数据集的文件夹名称。

### Q: 如何自定义 train/test 划分比例？
A: 修改 `split_train.txt` 和 `split_test.txt` 文件，重新运行数据准备脚本。

