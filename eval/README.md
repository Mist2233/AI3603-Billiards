# Evaluation Guide

本目录包含用于评估 Agent 性能的脚本和说明。

## 环境配置

1. **创建 Conda 环境** (Python 3.10)
   ```bash
   conda create -n pooltool python=3.10 -y
   conda activate pooltool
   ```

2. **安装依赖**
   ```bash
   # 在项目根目录下运行
   pip install -r requirements.txt
   # 如果遇到 pooltool-billiards 依赖冲突，可尝试：
   # pip install pooltool-billiards numpy bayesian-optimization packaging
   ```

## 运行评估

评估脚本 `evaluate.py` 位于项目根目录。它将运行 40 局比赛（四局循环制），统计 Agent A (BasicAgent) 和 Agent B (NewAgent) 的胜率。

```bash
# 确保在项目根目录
cd /path/to/AI3603-Billiards
conda activate pooltool

# 运行评估
python evaluate.py
```

## 评估规则

- **局数**: 40 局
- **轮换**: 每 4 局为一个循环，涵盖先后手和球型（实心/条纹）的所有组合。
- **计分**:
  - 胜: 1 分
  - 平: 0.5 分
  - 负: 0 分
- **胜率**: `总得分 / 40`

## 结果示例

运行结束后，终端将输出类似如下的统计信息：

```
最终结果： {'AGENT_A_WIN': 15, 'AGENT_B_WIN': 23, 'SAME': 2, 
           'AGENT_A_SCORE': 16.0, 'AGENT_B_SCORE': 24.0}
```

## 提交说明

最终提交时，请确保 `agent.py` 中包含你训练好的 `NewAgent` 代码。如果使用了额外的模型文件（如权重），请确保代码中正确加载，并将模型文件一同打包。
