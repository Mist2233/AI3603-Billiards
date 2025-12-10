# Training Guide

本目录包含训练智能体的相关代码和说明。

## 算力需求与申请建议

为了训练高性能的智能体（特别是基于强化学习的方法），建议申请学校的算力资源。

### 推荐配置
- **CPU**: 16+ 核心 (仿真计算密集型)
- **RAM**: 64GB+
- **GPU**: 1x NVIDIA GPU (8GB+ VRAM) - 如果使用深度强化学习 (PPO/SAC)
- **Disk**: 50GB+ (用于存储 checkpoints 和 logs)

### 申请清单示例
> 项目名称：AI3603 台球智能体训练
> 环境需求：Linux, Conda, Python 3.10
> 软件包：numpy, pooltool-billiards, bayesian-optimization, (可选 torch/tensorflow)
> 预计时长：24-72 小时 (取决于训练算法)

## 训练流程

### 1. 环境准备
同 `eval/README.md`，确保安装了所有依赖。

### 2. 训练脚本 (示例)
目前提供了一个基础的训练骨架 `train_script.py`（需自行实现具体训练逻辑）。

```bash
cd train
python train_script.py --episodes 1000 --save-dir ./checkpoints
```

### 3. 训练策略选择
- **参数搜索 (CPU)**: 使用贝叶斯优化或进化算法，调整 `NewAgent` 中的启发式参数（如力度、角度偏移等）。
- **强化学习 (GPU)**: 使用 PPO/SAC 等算法。需引入 `torch` 或 `tensorflow`，将环境封装为 Gym 接口，进行大规模交互训练。

## 目录结构
- `checkpoints/`: 存放训练好的模型权重
- `logs/`: 存放训练日志 (TensorBoard 等)
- `train_script.py`: 训练主入口
