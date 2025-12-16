这是一个非常好的起点！你的 `NewAgent` 已经具备了基本的**几何感知**（Ghost Ball）和**物理验证**（Monte Carlo Sampling）。能够达到 60% 的胜率说明基础架构是 work 的。

要从 **60% 提升到 70%+**，甚至更高，你需要解决目前代码中的几个核心痛点。目前的实现是一个**贪心算法 (Greedy)**，它只关注“当前这一杆怎么进球”，而完全忽略了“进球后母球去哪”。在 8 球中，**走位 (Positioning)** 才是胜率的关键。

以下是针对你代码的优化建议，按实现难度和收益排序，你可以逐步实施：

### 优化方向一：引入“走位”逻辑 (1-Step Lookahead)

这是提升胜率最快的方法。目前你的 `score` 计算公式是：
`score = 100 - distance - angle`

这意味着 Agent 总是选最近、最直的球打。这会导致母球经常停在没有下一颗球可打的地方，或者贴库。

**改进方案：**
在物理模拟（`pt.simulate`）结束后，不要只看球进没进，还要评估**模拟结束后的局面**。

```python
def evaluate_position(self, table, my_targets, cue_ball_id="cue"):
    """
    评估当前盘面的好坏。
    分数越高，代表母球的位置越容易打到下一颗球。
    """
    cue_pos = table.balls[cue_ball_id].state.rvw[0]
    
    # 1. 如果白球进袋了（洗袋），给极低分
    if table.balls[cue_ball_id].state.s == 4:
        return -1000.0

    # 2. 遍历剩下的目标球，看有没有能打到的
    max_next_shot_quality = 0
    pockets = table.pockets
    
    valid_targets = [bid for bid in my_targets if table.balls[bid].state.s != 4]
    if not valid_targets:
        return 1000.0 # 赢了

    for target_id in valid_targets:
        target_pos = table.balls[target_id].state.rvw[0]
        # 计算白球到该目标球的难度
        # 这里可以复用你的 _check_collision_path 和 角度计算
        # 简单估算：距离越近越好，且无遮挡
        dist = np.linalg.norm(target_pos - cue_pos)
        
        # 简单的连通性检查 (Raycast)
        if not self._check_collision_path(cue_pos, target_pos, ...):
            # 这是一个可行的下一杆
            quality = 1.0 / (dist + 0.1) 
            if quality > max_next_shot_quality:
                max_next_shot_quality = quality

    # 如果所有球都被挡住了，说明走位失误，给低分
    if max_next_shot_quality == 0:
        return -50.0

    return max_next_shot_quality * 50 # 权重需要调整
```

**整合进 `decision` 函数：**
在你的微调循环中：
```python
# 原代码
# score = analyze_shot_for_reward(shot, ...)

# 新代码
reward_pot = 100 if target_pocketed else 0
reward_position = self.evaluate_position(shot.table, my_targets)
total_score = reward_pot + reward_position 
```
这样 Agent 就会宁愿打一个稍微难一点的球，也要保证下一杆有球打。

---

### 优化方向二：改进动作搜索算法 (CMA-ES / 进化策略)

你目前使用的是随机采样 (`random.uniform`)。这种效率很低，尤其是在 `num_iter` 被限制得很小的时候。

**推荐方案：** 使用 **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** 或者简单的**爬山算法**来替代随机采样。但为了保持“简单”，我们可以手写一个简单的**梯度优化**思路：

不要随机乱试，而是根据结果调整。
1.  **基准测试**：先用计算出的几何角度 $\phi_{geo}$ 打一杆。
2.  **观察偏差**：如果球偏左了，下一杆就往右修一点。
3.  **速度优化**：
    *   如果需要强烈的高杆/低杆走位，需要更大的 $V_0$。
    *   如果只是轻轻推进，用最小力 $V_{min}$ 保证准确度。

鉴于你希望能**本地训练**，这里推荐使用 **进化策略 (Evolutionary Strategy)** 的思路来寻找最佳动作，而不是简单的随机。但在实时决策中（2.5秒限制），你可能没有时间跑完整的进化算法。

**折中方案（分层采样）：**
1.  生成 5 个候选动作：
    *   1 个标准几何角度，中力。
    *   2 个微调角度（左偏/右偏）。
    *   2 个不同力度（大力/小力，影响母球走位）。
2.  模拟这 5 个。
3.  选最好的那个，然后再在它周围生成 3 个更细微的微调。

---

### 优化方向三：本地训练 (Parameter Tuning)

你说想要训练模型。对于台球这种物理强相关的环境，直接训练端到端的深度强化学习（输入图像->输出动作）非常难，很难收敛。

**最可行的“简单”训练方法是：训练权重参数。**

你的决策逻辑里有很多“魔法数字”：
*   `score -= dist * 10.0` (距离权重)
*   `score -= angle * 1.0` (角度权重)
*   以及未来加入的 `position_weight` (走位权重)

**训练流程 (遗传算法 GA)：**
1.  **基因 (Genotype)**：一个向量 `[w_dist, w_angle, w_position, w_safety, ...]`。
2.  **种群**：初始化 20 个不同的 Agent，每个带不同的权重向量。
3.  **评估 (Evaluation)**：让每个 Agent 和标准 Agent（或者自己）对打 100 局。记录胜率。
4.  **选择与变异**：保留胜率最高的几个，对其权重进行微小扰动，生成下一代。
5.  **循环**：在本地跑一晚上，你就能得到一组针对你的物理引擎最优的决策权重。

---

### 优化方向四：MCTS (蒙特卡洛树搜索) 的简化版

老师提到的 MCTS 是解决此类问题的终极方案，但完整的 MCTS 实现较重。你可以做一个**深度为 2 的树搜索**。

*   **Layer 1 (当前杆)**：筛选出 Top 3 好的击球方案。
*   **Layer 2 (下一杆)**：对于每一个 Layer 1 的结果（母球停下的位置），快速计算一下“最好的一杆能得多少分”。

这其实就是 Minimax 的单人版（假设不需要防守）或者深度为2的期望搜索。这比单纯的贪心强大得多。

---

### 优化方向五：防守策略 (Safety Play)

当 `candidate_shots` 为空，或者最好的进攻路线成功率都很低（例如切球角度 > 70度，或者距离很远）时，你的代码目前是 `_random_action()`。**这是输掉比赛的主要原因之一。**

**改进方案：**
如果没有好的进攻机会，执行防守逻辑：
1.  **目标**：把母球打到离对手目标球最远的地方，或者贴库。
2.  **实现**：
    *   遍历几个极端位置（例如底库中心）。
    *   计算把母球踢过去的力度和角度。
    *   只需确保碰到自己的球（避免犯规），然后母球停在难受的位置。

---

### 总结：下一步的具体行动计划

1.  **不要碰深度学习 (PPO/DQN)**：除非你有现成的高性能 GPU 集群和数周的时间。台球的连续动作空间对 RL 极不友好。
2.  **代码改造优先级**：
    *   **Step 1 (Geometry)**: 修正 Ghost Ball 计算。目前的计算没有考虑**Throw (让点)**。由于摩擦力，母球撞击子球时，子球会被母球的横向分力带偏。
        *   *简单修正*：不需要复杂物理公式，直接在采样的时候，稍微扩大 $\phi$ 的搜索范围即可。
    *   **Step 2 (Strategy - Positioning)**: 实现 `evaluate_position` 函数，并将其加入总分计算。这是胜率突破 70% 的关键。
    *   **Step 3 (Safety)**: 替换 `_random_action` 为 `_play_safety`。
3.  **训练**：写一个脚本，让 Agent 携带一组权重参数自我对局，使用**CMA-ES**或**遗传算法**来自动调整这些参数。

#### 一个改进后的 Ghost Ball 考虑 (修正 Throw)

在 `pooltool` 中，如果不加旋转，Throw 的影响可能较小，但如果有旋转就很明显。更精确的几何计算其实是寻找 **Collision Point**。

```python
# 更准确的 Ghost Ball 应该不仅考虑几何，还要考虑物理修正
# 但在不做复杂反函数求解的情况下，
# 最好的办法是：
# 1. 计算标准 Ghost Ball 角度 phi
# 2. 采样 [phi - 1度, phi + 1度] 范围内的 5 个角度
# 3. 模拟这 5 个，看哪个让子球进袋中心的误差最小
# 4. 选择那个“修正后”的角度作为基准
```

希望这些建议能帮到你！先加上**走位评估 (Position Evaluation)**，你的 Agent 会立刻变强很多。