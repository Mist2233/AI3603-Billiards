# 台球大作业需求与 Issue 计划

## 一、主要项目要求（摘自 README、PROJECT_GUIDE、GAME_RULES）
1. 在 `agent.py` 中实现自定义智能体 `NewAgent`，仅该类可以修改，用以替换 `evaluate.py` 中 `agent_b` 的行为。
2. `poolenv.py` 负责八球规则实现，**最终测试不得改动**，所有智能体需以该环境为准。
3. 项目提交材料需要三部分：
   - `train/` 目录（可改 agent/poolenv 等）中的训练代码、环境配置说明和可复现流程；
   - `eval/` 目录（只能改 agent.py、utils.py）在发布版本上实现的测试代码、checkpoint 及测试说明；
   - 英文报告（≤15页、IEEE 双栏），介绍思路、实现、效果与分工（Overleaf 模板提供）。
4. 胜负按 `evaluate.py` 设定的四局循环、40局对战、胜1分/平0.5分的方式统计，Agent B 胜率目标 ≥60% 以上为及格。
5. 训练/评估需可复现：提供环境配置（Python ≥3.10）、超参数等；报告需包含方法、实验与心得；训练可包括强化学习、MCTS、LLM 等任意思路。
6. 环境含高斯噪声（`poolenv.py` 默认开启且不可改）；参考 `agent.py` 中 `BasicAgent` 的噪声设置可用于训练阶段调节。

## 二、Issue 驱动计划
以下 Issue 草稿均围绕课程要求展开，可直接用作项目管理或 GitHub Issue 模板。

### Issue 1 - Implement decision logic for `NewAgent`
- **目标**：在 `agent.py` 的 `NewAgent` 中实现包含感知-规划的击球策略，替代当前随机行为。
- **步骤**：
  1. 对观察数据（`balls`、`my_targets`、`table`）做特征提取，包括剩余目标球位置、白球与目标球夹角等；
  2. 设计决策模块（可选启发式、贝叶斯、RL），输出 `V0/phi/theta/a/b`，遵循动作边界；
  3. 加入简单的安全检查（例如防止射向黑8时机不对）并在开发阶段引入日志；
  4. 编写单轮调试脚本（可复用 `poolenv` 的测试代码）验证返回值格式。

当前实现的Agent没有经过任何训练，只是依靠物理规则和几何算法构建的策略来打球。简单来说，agent利用“幽灵球”原理计算理论击球点，再用物理引擎的“蒙特卡洛模拟”在脑海中预演最佳的结果，然后选择它认为最好的那一杆打出去。

我们实现的NewAgent，在和BasicAgent的20局对局中，胜率能达到80%及以上，也就是20局中至少能胜出16局。也就是说，这个简单的原型机，其表现已经相当不错。

后续可能的优化方向：引入学校算力，使用深度学习训练当前的原始模型，主要针对下面的内容：
1. 选色策略：是纯色球更好打，还是花色球更好打？Agent是否需要自行选择其一？
2. 走位方式：AI在打出当前一杆之后，必须计算母球停止的位置，判断是否有利于下一杆的出击。
3. 死球问题：如果出现死球，agent应当及早解决，而不能让死球留到最后。
4. 关键球决策：关键球就是打进黑8之前的最后一颗球。这颗球的进攻，决定黑8是否好打。为了让黑8有完美的进攻角度，关键球应当有极高的权重。
5. 防守策略：如果当前的进球概率极低，那么agent应当学会防守，即干扰对方的进攻，例如做斯诺克或者远台防守。

### Issue 2 - Build reproducible training pipeline under `train/`
- **目标**：整理训练所需代码、超参数与说明，完成可复现训练流程。
- **步骤**：
  1. 创建 `train/` 目录，放置训练脚本（如 RL 训练、参数搜索或模拟数据生成）和所需模型代码（可以临时修改 `agent.py`、`poolenv.py`）；
  2. 准备训练配置文档（`train/README.md`）列出 Python 版本、依赖安装、运行命令、参数选择、checkpoint 路径；
  3. 将训练结果（如 policy checkpoint、logs）保存可复用的路径，记录调参思路；
  4. 确保训练过程可在指定环境（`requirements.txt`）下复现，注明随机种子、数据保存周期等。

### Issue 3 - Prepare evaluation artifacts under `eval/`
- **目标**：完成可复现的评估流程，所需文件放在 `eval/` 并说明使用方式。
- **步骤**：
  1. 新建 `eval/` 目录，复制或封装 `evaluate.py` 为最终对战脚本，统一接口调用 `NewAgent` 及 `utils`；
  2. 在 `eval/` 中放置测试 README，明确 `conda activate pooltool`、`python evaluate.py` 等命令及奖励统计方式；
  3. 将训练阶段最好的 checkpoint 拷贝至此（或提供加载链接），说明如何在 `NewAgent` 中加载；
  4. 验证 40 局对战输出与 `GAME_RULES` 中统计一致，并在 README 中记录胜率/得分结果。

### Issue 4 - Document experimental report (English, IEEE style)
- **目标**：编写英文报告（≤15 页）总结方法、实现、效果与分工，遵照 Overleaf IEEE 模板。
- **步骤**：
  1. 确认 Overleaf 模板已复制，填写标题、摘要与关键词；
  2. 撰写方法部分：思路、模型架构、损失/奖励函数；补充尝试过的备选方案和教训；
  3. 列出实验设置（训练超参、评估流程、硬件），贴上最终 `evaluate.py` 得分；
  4. 说明分工细节（训练、调参、报告撰写、实验验证等），附上结论与未来工作建议。

### Issue 5 - Noise robustness & debugging utilities
- **目标**：提供调试/鲁棒性辅助工具（日志、可调噪声配置）帮助重复实验并分析失败情况。
- **步骤**：
  1. 在 `agent.py` 或 `utils.py` 中加入可控制的噪声开关，记录默认环境噪声（`poolenv`）和训练噪声（`BasicAgent`）的协同影响；
  2. 编写脚本记录每局 `take_shot` 的关键事件（进球、犯规、错误），以便评估策略改动的效果；
  3. 集成可视化（如生成 JSON/log）用于分析 `NewAgent` 的决策分布和失败原因；
  4. 参考这些日志撰写调试说明（追加至 `train/README.md` 或 `eval/README.md`）。

### Issue 6 - Submission checklist & packaging
- **目标**：整理最终提交的路径结构与说明，确保训练、测试、报告文件齐全。
- **步骤**：
  1. 编写项目根 `README.md` 或新文件补充最终提交目录结构（train/、eval/、report、checkpoints）；
  2. 确认 `requirements.txt` 覆盖所有依赖并注明 Python 版本；
  3. 列出必须提交的文件列表和命令（训练、测试、报告预览）。
  4. 浏览 `GAME_RULES`、`PROJECT_GUIDE` 复核所有测试要求并对照检查表。
