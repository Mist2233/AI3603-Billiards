# Issue 1: NewAgent 开发文档

## 1. 功能实现说明

本次开发主要完成了 `NewAgent` 的决策逻辑实现，采用了 **基于物理模型的“幽灵球”策略 (Physics-based Ghost Ball Strategy)**。该策略结合了几何计算与物理微调，旨在解决传统几何方法无法应对复杂物理碰撞的问题，同时避免纯随机搜索效率低下的缺陷。

### 1.1 核心算法流程

`NewAgent` 的决策过程分为以下几个步骤：

1.  **目标识别**：识别当前合法的目标球（实心/条纹/黑8）。
2.  **几何路径筛选**：
    *   遍历所有目标球与所有袋口的组合。
    *   计算 **幽灵球位置 (Ghost Ball Position)**：即白球击打目标球进袋所需的理想撞击点。
    *   **路径碰撞检测**：检查“白球->幽灵球”和“目标球->袋口”路径上是否存在障碍球。
    *   **切球角度评估**：剔除切球角度过大（如 > 80度）的极难球。
    *   **基础评分**：根据距离、角度等几何因素对候选路径进行初步评分。
3.  **物理微调 (Physics-based Fine-tuning)**：
    *   选取评分最高的 Top-N (默认为 3) 条候选路径。
    *   在几何计算出的理想击球角度附近，进行小范围的随机微调（Monte Carlo Sampling）。
    *   使用 `pooltool` 物理引擎模拟击球结果。
    *   选择预期奖励（Reward）最高的动作作为最终决策。

### 1.2 关键函数

*   `_calculate_ghost_ball_pos(target_pos, pocket_pos, ball_radius)`: 计算幽灵球坐标。
*   `_check_collision_path(start_pos, end_pos, obstacle_balls, ball_radius)`: 基于圆柱体投影法的路径碰撞检测。
*   `decision(balls, my_targets, table)`: 主决策入口，整合上述逻辑。

## 2. API 接口文档

### `NewAgent` 类

继承自 `Agent` 基类。

#### `decision` 方法

```python
def decision(self, balls, my_targets, table):
    """
    基于几何与物理结合的决策策略
    
    参数:
        balls (dict): 球状态字典，{ball_id: Ball对象}
        my_targets (list): 当前玩家的目标球ID列表，例如 ['1', '2', '3']
        table (Table): 球桌对象，包含袋口位置等信息
        
    返回:
        dict: 包含击球参数的字典
            {
                "V0": float,    # 击球速度 (m/s)
                "phi": float,   # 水平击球角度 (degrees)
                "theta": float, # 垂直击球角度 (degrees)
                "a": float,     # 水平击球点偏移 (-0.5 ~ 0.5)
                "b": float      # 垂直击球点偏移 (-0.5 ~ 0.5)
            }
    """
```

## 3. 测试与验证

### 3.1 单元测试

编写了 `test_new_agent.py`，覆盖了核心几何计算函数的正确性：
*   幽灵球位置计算（直线球、角度球）
*   路径碰撞检测（无遮挡、有遮挡、背向遮挡）
*   决策函数的异常处理与 Fallback 机制

### 3.2 性能基准测试

使用 `benchmark_issue1.py` 进行了 20 局 `NewAgent` vs `BasicAgent` 的对战测试。

**测试结果 (20 局):**
*   **NewAgent 胜率**: 85.0% (17/20)
*   **BasicAgent 胜率**: 15.0% (3/20)
*   **NewAgent 犯规率**: 15.6% (41/262)

**结论**：`NewAgent` 在胜率上显著优于基于贝叶斯优化的 `BasicAgent`，且决策速度在经过优化后满足实时性要求。

## 4. 部署与使用

### 4.1 依赖环境

*   Python 3.8+
*   `pooltool`
*   `numpy`

### 4.2 使用示例

```python
from agent import NewAgent
from poolenv import PoolEnv

# 初始化环境和 Agent
env = PoolEnv()
agent = NewAgent()

# 获取环境状态
balls = env.balls
my_targets = env.player_targets['B'] # 假设 NewAgent 是玩家 B
table = env.table

# 获取决策
action = agent.decision(balls=balls, my_targets=my_targets, table=table)

# 执行击球
env.take_shot(action)
```
