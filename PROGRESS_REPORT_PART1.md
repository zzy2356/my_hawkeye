# Hawkeye 项目进度报告（第一部分）

## 一、论文（设计）目标

### 原始论文目标
- **任务**：隐式异常情绪发现与定位（IasDig, Implicit anomalous sentiment Discovering and grounding）
- **应用场景**：监控视频、无人机侦察视频等重建视频中的异常情绪识别
- **核心创新**：
  1. 设计图结构化场景建模模块（Graph-structured Scene Modeling）
  2. 设计平衡异构 MoE 模块（Balanced Heterogeneous MoE）
  3. 融合视频、姿态、场景关系信息进行多模态理解

### 当前工作目标（Qwen3-VL 迁移）
- **主要目标**：将原 Hawkeye（基于 LLaVA + Vicuna + LanguageBind）迁移到 Qwen3-VL 多模态大模型
- **设计目标**：
  1. 保留原论文的"场景增强 MoE"核心机制
  2. 实现"视频特征 + 场景增强 MoE"在占位符处的同构拼接
  3. 支持 Qwen3-VL 的多模态输入处理流程
  4. 保持与原 LLaVA 链路的并行可用性

---

## 二、论文（设计）内容

### 原论文核心内容
1. **场景建模**：使用 GTN（Graph Transformer Network）对视频中的人物姿态和物体关系进行图建模
2. **MoE 融合**：通过异构专家混合网络融合姿态特征和场景特征
3. **多模态融合**：将视频、姿态、场景特征在 LLM 的占位符处进行拼接
4. **IasDig 任务**：交互式发现和定位异常情绪

### 当前迁移设计内容
1. **多模态编码层**：
   - 视频编码：Qwen3-VL 自带视觉编码器（替代 LanguageBind）
   - 姿态编码：PoseTower（Linear 85→4096）
   - 场景编码：SceneGraphTower（GTN 结构）

2. **场景增强 MoE**：
   - 输入：编码后的姿态 token + 场景 token
   - 处理：MoE 路由和专家重采样
   - 输出：场景增强 token 序列

3. **融合机制**：
   - 在 `<video>` 占位符处实现 **[Qwen 视频 embedding] + [Hawkeye MoE token]** 的拼接
   - 通过 `_materialize_qwen_multimodal_embeds` 和 `_splice_hawkeye_tokens` 实现

4. **训练与推理**：
   - 训练：LazySupervisedDataset → collate_qwen3vl_batch → Qwen3VLHawkeyeAdapter.forward
   - 推理：eval.py 或 smoke_infer.py → 生成异常情绪标签和定位

---

## 三、开题报告

### 3.1 选题的目的和意义

#### 研究背景
- 现有视频异常检测方法主要依赖显式信息（表情、语音、文字），在监控视频中这些信息往往缺失
- 隐式异常情绪（如犯罪倾向）需要通过人物姿态、物体关系等场景信息来识别
- 现有 Video-LLM（如 LLaVA-Video）缺乏对场景结构信息的显式建模

#### 研究目的
1. **原论文目的**：提出 Hawkeye 模型，通过场景增强 MoE 机制，在 IasDig 任务上实现更好的异常情绪识别
2. **当前迁移目的**：
   - 将 Hawkeye 从 LLaVA 框架迁移到更强大的 Qwen3-VL 多模态大模型
   - 验证场景增强 MoE 机制在新框架下的有效性
   - 为后续模型升级和优化奠定基础

#### 研究意义
- **学术意义**：验证场景信息对隐式异常情绪识别的重要性
- **应用意义**：为监控视频、无人机侦察等实际应用提供更准确的异常检测方案
- **工程意义**：建立可扩展的多模态融合框架，支持不同 LLM 骨干的快速迁移

### 3.2 国内外研究现状

#### 国外研究现状
1. **Video-LLM 领域**：
   - LLaVA-Video：基于 LLaVA 的视频理解模型
   - Qwen3-VL：阿里开源的多模态大模型，支持视频、图像、文本的统一处理
   - GPT-4V：闭源的多模态模型，性能强但不可微调

2. **异常检测领域**：
   - 传统方法：基于光流、轨迹的异常检测
   - 深度学习方法：基于 CNN/RNN 的视频异常检测
   - LLM 方法：利用 LLM 的语义理解能力进行异常识别

3. **场景理解领域**：
   - 场景图生成（Scene Graph Generation）：RelTR、SGG 等
   - 图神经网络（GNN）：GCN、GAT、GTN 等
   - 多模态融合：Cross-modal attention、MoE 等

#### 国内研究现状
- 清华、北大等高校在视频理解、异常检测领域有相关研究
- 工业界（阿里、腾讯、字节）在多模态大模型方面投入较大
- 原 Hawkeye 论文发表于 ACM MM 2024，代表国内在该领域的最新进展

#### 现有方案的不足
- LLaVA 框架相对较旧，视觉编码能力有限
- 场景信息的显式建模仍不充分
- 缺乏对"场景稀疏"和"场景密集"两类数据的平衡处理

### 3.3 主要研究内容

#### 内容 1：多模态编码层的适配
- **目标**：将 Qwen3-VL 的视觉编码器与 Hawkeye 的姿态、场景编码器进行融合
- **内容**：
  - 保留 PoseTower 和 SceneGraphTower 的实现
  - 使用 Qwen3-VL 的视觉编码替代 LanguageBind
  - 实现统一的 embedding 维度映射

#### 内容 2：场景增强 MoE 的集成
- **目标**：确保 MoE 路由和专家混合在新框架下正常工作
- **内容**：
  - 复用原 llava_arch.py 中的 MoE 实现
  - 在 Qwen3VLHawkeyeAdapter 中集成 MoE 逻辑
  - 验证 MoE 的路由权重和专家输出

#### 内容 3：占位符融合机制的实现
- **目标**：实现"视频特征 + 场景增强 MoE"的同构拼接
- **内容**：
  - `_materialize_qwen_multimodal_embeds`：显式物化 Qwen 视频 embedding
  - `_splice_hawkeye_tokens`：在视频 token 后插入 MoE token
  - 验证最终 embedding 序列的正确性

#### 内容 4：训练与推理流程的适配
- **目标**：建立完整的 Qwen3-VL + Hawkeye 训练和推理流程
- **内容**：
  - 数据预处理：qwen3vl_data.py 中的 Qwen chat template 处理
  - 训练脚本：train.py 中的 Qwen 分支逻辑
  - 推理脚本：eval.py 和 smoke_infer.py 的 Qwen 路径

#### 内容 5：与原 LLaVA 链路的并行维护
- **目标**：保持原 LLaVA 链路的可用性，支持对比实验
- **内容**：
  - 在 train.py 中通过 `_is_qwen3_vl_model_name` 判断模型类型
  - DataCollator 中的 qwen_multimodal 分支
  - eval.py 中的 _run_legacy_hawkeye_inference 和 _run_qwen3_vl_inference

---

## 四、实施方案、进度安排及预期效果

### 4.1 实施方案

#### 阶段 1：架构设计与代码框架搭建（已完成）
- ✅ 设计 Qwen3VLHawkeyeAdapter 类结构
- ✅ 实现 PoseTower、SceneGraphTower、HawkeyeMoE 的集成
- ✅ 完成 _materialize_qwen_multimodal_embeds 和 _splice_hawkeye_tokens 的实现
- ✅ 建立 qwen3vl_data.py 的数据预处理流程

#### 阶段 2：训练流程验证（进行中）
- 🔄 在 GPU 服务器上进行 smoke test（2-5 步训练）
- 🔄 验证数据加载、模型 forward、loss 计算的正确性
- 🔄 调试 pose/scene 特征的维度和拼接逻辑

#### 阶段 3：完整训练与评估（待进行）
- ⏳ 在 TSL-300 和 UCF-Crime 数据集上进行完整训练
- ⏳ 对比 Qwen3-VL Hawkeye 与原 LLaVA Hawkeye 的性能
- ⏳ 评估 FNR、F2、mAP 等指标

#### 阶段 4：优化与发布（待进行）
- ⏳ 根据实验结果进行模型优化
- ⏳ 整理代码、文档和模型检查点
- ⏳ 发布最终版本

### 4.2 进度安排

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|---------|------|
| 1 | 架构设计与代码框架 | 2 周 | ✅ 完成 |
| 2 | GPU 服务器 smoke test | 1 周 | 🔄 进行中 |
| 2 | 数据加载与 forward 调试 | 1 周 | 🔄 进行中 |
| 3 | TSL-300 完整训练 | 2 周 | ⏳ 待进行 |
| 3 | UCF-Crime 完整训练 | 2 周 | ⏳ 待进行 |
| 3 | 性能对比与分析 | 1 周 | ⏳ 待进行 |
| 4 | 优化与发布 | 1 周 | ⏳ 待进行 |

### 4.3 预期效果

#### 功能预期
- ✅ Qwen3-VL + Hawkeye 完整训练流程可用
- ✅ 占位符处实现"视频特征 + 场景增强 MoE"的同构拼接
- ✅ 支持 TSL-300 和 UCF-Crime 数据集的训练与评估
- ✅ 原 LLaVA 链路保持可用

#### 性能预期
- 🎯 Qwen3-VL Hawkeye 在 TSL-300 上的 FNR ≤ 40%（原论文 35.82%）
- 🎯 Qwen3-VL Hawkeye 在 UCF-Crime 上的 FNR ≤ 50%（原论文 45.66%）
- 🎯 相比原 LLaVA 版本，性能持平或更优

#### 工程预期
- 📦 完整的代码框架和文档
- 📦 可复现的训练脚本和配置
- 📦 预训练模型检查点
- 📦 清晰的架构对比文档

