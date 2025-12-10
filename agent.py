"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random

# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数

    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]

    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """

    # 1. 基本分析
    new_pocketed = [
        bid
        for bid, b in shot.balls.items()
        if b.state.s == 4 and last_state[bid].state.s != 4
    ]

    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [
        bid
        for bid in new_pocketed
        if bid not in player_targets and bid not in ["cue", "8"]
    ]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
            other_ids = [i for i in ids if i != "cue"]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
            foul_first_hit = True
    else:
        remaining_own_before = [
            bid for bid in player_targets if last_state[bid].state.s != 4
        ]
        opponent_plus_eight = [
            bid
            for bid in last_state.keys()
            if bid not in player_targets and bid not in ["cue"]
        ]
        if "8" not in opponent_plus_eight:
            opponent_plus_eight.append("8")

        if (
            len(remaining_own_before) > 0
            and first_contact_ball_id in opponent_plus_eight
        ):
            foul_first_hit = True

    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if "cushion" in et:
            if "cue" in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if (
        len(new_pocketed) == 0
        and first_contact_ball_id is not None
        and (not cue_hit_cushion)
        and (not target_hit_cushion)
    ):
        foul_no_rail = True

    # 计算奖励分数
    score = 0

    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (
            len(player_targets) == 1 and player_targets[0] == "8"
        )
        score += 100 if is_targeting_eight_ball_legally else -150

    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30

    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20

    if (
        score == 0
        and not cue_pocketed
        and not eight_pocketed
        and not foul_first_hit
        and not foul_no_rail
    ):
        score = 10

    return score


class Agent:
    """Agent 基类"""

    def __init__(self):
        pass

    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）

        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass

    def _random_action(
        self,
    ):
        """生成随机击球动作

        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            "V0": round(random.uniform(0.5, 8.0), 2),  # 初速度 0.5~8.0 m/s
            "phi": round(random.uniform(0, 360), 2),  # 水平角度 (0°~360°)
            "theta": round(random.uniform(0, 90), 2),  # 垂直角度
            "a": round(
                random.uniform(-0.5, 0.5), 3
            ),  # 杆头横向偏移（单位：球半径比例）
            "b": round(random.uniform(-0.5, 0.5), 3),  # 杆头纵向偏移
        }
        return action


class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""

    def __init__(self, target_balls=None):
        """初始化 Agent

        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()

        # 搜索空间
        self.pbounds = {
            "V0": (0.5, 8.0),
            "phi": (0, 360),
            "theta": (0, 90),
            "a": (-0.5, 0.5),
            "b": (-0.5, 0.5),
        }

        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2

        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {"V0": 0.1, "phi": 0.1, "theta": 0.1, "a": 0.003, "b": 0.003}
        self.enable_noise = False

        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器

        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子

        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed,
        )

        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8, gamma_pan=1.0
        )

        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer,
        )
        optimizer._gp = gpr

        return optimizer

    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数 (Optimized for Benchmark)
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象

        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            return self._random_action()
        try:
            # 简化版 BasicAgent 逻辑：大幅减少搜索次数以加快测试速度
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {
                bid: copy.deepcopy(ball) for bid, ball in balls.items()
            }

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]

            # 1.动态创建“奖励函数” (Wrapper)
            def reward_fn_wrapper(V0, phi, theta, a, b):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

                try:
                    shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    pt.simulate(shot, inplace=True)
                except Exception:
                    return -500

                score = analyze_shot_for_reward(
                    shot=shot, last_state=last_state_snapshot, player_targets=my_targets
                )
                return score

            # 减少搜索次数
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            
            # 使用更少的迭代次数 (例如 5 init + 2 iter)
            optimizer.maximize(init_points=5, n_iter=2)

            best_result = optimizer.max
            best_params = best_result["params"]
            best_score = best_result["target"]

            if best_score < 10:
                return self._random_action()
                
            action = {
                "V0": float(best_params["V0"]),
                "phi": float(best_params["phi"]),
                "theta": float(best_params["theta"]),
                "a": float(best_params["a"]),
                "b": float(best_params["b"]),
            }
            return action

        except Exception as e:
            print(f"[BasicAgent] 异常: {e}")
            return self._random_action()


class NewAgent(Agent):
    """自定义 Agent 实现：基于目标球导向的蒙特卡洛搜索"""

    def __init__(self):
        super().__init__()
        self.num_iter = 5  # 减少采样次数以加快速度

    def _calc_angle(self, pos1, pos2):
        """计算从pos1指向pos2的角度（度）"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

    def _calc_distance(self, pos1, pos2):
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _calculate_ghost_ball_pos(self, target_pos, pocket_pos, ball_radius=0.028575):
        """
        计算幽灵球位置
        
        参数:
            target_pos: 目标球位置 (x, y, z)
            pocket_pos: 袋口位置 (x, y, z)
            ball_radius: 球半径 (默认 57.15mm / 2)
            
        返回:
            ghost_pos: 幽灵球中心坐标 (x, y, z)
        """
        # 目标球到袋口的向量
        vec_to_pocket = np.array(pocket_pos) - np.array(target_pos)
        # 忽略高度分量 (z)
        vec_to_pocket[2] = 0
        
        dist = np.linalg.norm(vec_to_pocket)
        if dist < 1e-6:
            return target_pos # Should not happen if target not in pocket
            
        # 单位向量
        unit_vec = vec_to_pocket / dist
        
        # 幽灵球位置 = 目标球位置 - 2*R * unit_vec
        # 注意：这里方向是 target -> pocket，所以幽灵球在 target 的反方向
        # 实际上幽灵球应该是白球击打时占据的位置，使得目标球受力沿 unit_vec 运动
        # 所以白球中心应该在 target_pos - 2*R * unit_vec (如果 unit_vec 是 target->pocket)
        # 等等，如果是 target->pocket，那么碰撞点在 target + R * unit_vec ? 不对
        # 碰撞点是 target 和 pocket 连线上的点吗？
        # 动量传递沿着连心线。
        # 要让 target 沿 unit_vec 飞出，连心线必须是 unit_vec 方向。
        # 即 (target_pos - cue_pos_at_impact) // unit_vec
        # 所以 cue_pos_at_impact = target_pos - 2*R * unit_vec
        
        ghost_pos = np.array(target_pos) - 2 * ball_radius * unit_vec
        # 保持 z 坐标一致 (通常是球半径)
        ghost_pos[2] = target_pos[2]
        
        return ghost_pos

    def _check_collision_path(self, start_pos, end_pos, obstacle_balls, ball_radius=0.028575):
        """
        简单的路径碰撞检测 (圆柱体检测)
        
        参数:
            start_pos: 起始点 (白球位置)
            end_pos: 终点 (幽灵球位置)
            obstacle_balls: 障碍球列表 [Ball objects]
            ball_radius: 球半径
            
        返回:
            bool: 是否有碰撞 (True=有阻挡, False=无阻挡)
        """
        # 路径向量
        path_vec = np.array(end_pos) - np.array(start_pos)
        path_vec[2] = 0 # 2D projection
        path_len = np.linalg.norm(path_vec)
        
        if path_len < 1e-6:
            return False
            
        path_dir = path_vec / path_len
        
        # 检测每个障碍球
        for ball in obstacle_balls:
            obs_pos = np.array(ball.state.rvw[0])
            obs_pos[2] = 0
            
            # 障碍球到起始点的向量
            to_obs = obs_pos - np.array(start_pos)
            to_obs[2] = 0
            
            # 投影长度
            proj_len = np.dot(to_obs, path_dir)
            
            # 只有在路径段内的障碍物才算 (稍稍放宽一点范围)
            if proj_len < -ball_radius or proj_len > path_len + ball_radius:
                continue
                
            # 垂直距离
            closest_point = np.array(start_pos) + proj_len * path_dir
            closest_point[2] = 0
            dist_sq = np.sum((obs_pos - closest_point)**2)
            
            # 判定阈值：2*R (球心距小于2R即碰撞)
            # 稍微加一点 margin 避免擦边太极限
            if dist_sq < (2 * ball_radius * 1.05) ** 2:
                return True
                
        return False

    def decision(self, balls=None, my_targets=None, table=None):
        """
        基于几何与物理结合的决策策略 (NewAgent Issue 1)
        
        策略流程：
        1. 识别有效目标球 (my_targets)
        2. 对每个目标球，遍历所有6个袋口，计算“幽灵球”位置
        3. 剔除无效路径：
           - 目标球到袋口被阻挡
           - 白球到幽灵球被阻挡
           - 击球角度过大（切球太薄）
        4. 对候选路径进行评分：
           - 距离越近越好
           - 角度越正越好
           - 优先选择容易进的袋口 (中袋通常较难)
        5. 对 Top N 候选路径进行物理微调 (Monte Carlo Sampling)
           - 在理论击球角附近微调 V0, phi
           - 模拟并选择最终 Reward 最高的动作
        """
        if balls is None:
            return self._random_action()

        try:
            cue_ball = balls["cue"]
            cue_pos = cue_ball.state.rvw[0]
            ball_radius = cue_ball.params.R

            # 1. 确定有效目标球
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if not remaining_own:
                targets = ["8"]  # 必须打黑8
            else:
                targets = remaining_own

            candidate_shots = [] # List of (score, target_id, pocket_id, ghost_pos, aim_angle, dist)
            
            # 获取所有袋口位置
            pockets = table.pockets
            # pockets is a dict or list? In pooltool, table.pockets is usually a dict by pocket ID
            # Let's inspect available pockets keys if possible, or iterate values.
            # Assuming table.pockets.values() gives pocket objects with center property
            
            # 2. 遍历 目标球 x 袋口
            for target_id in targets:
                if target_id not in balls:
                    continue
                target_ball = balls[target_id]
                target_pos = target_ball.state.rvw[0]
                
                # 排除已经进袋的球 (double check)
                if target_ball.state.s == 4:
                    continue
                    
                for pocket_id, pocket in pockets.items():
                    # pocket.center gives [x, y, z]
                    pocket_pos = pocket.center
                    
                    # A. 计算幽灵球位置
                    ghost_pos = self._calculate_ghost_ball_pos(target_pos, pocket_pos, ball_radius)
                    
                    # B. 计算击球角度 (Cut Angle)
                    # 向量: 白球 -> 幽灵球
                    vec_cue_to_ghost = ghost_pos - cue_pos
                    dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
                    
                    # 向量: 幽灵球 -> 目标球 (即击球方向)
                    vec_ghost_to_target = target_pos - ghost_pos
                    
                    # 计算夹角: (白球->幽灵球) vs (幽灵球->目标球)
                    # 如果共线 (0度)，则是正对直球。
                    # 如果接近 90度，则是极薄切球。
                    
                    # 归一化
                    if dist_cue_to_ghost < 1e-6:
                        # 白球就在幽灵球位置（贴球），需要特殊处理，暂时跳过
                        continue
                        
                    dir_cue_to_ghost = vec_cue_to_ghost / dist_cue_to_ghost
                    dir_ghost_to_target = vec_ghost_to_target / np.linalg.norm(vec_ghost_to_target)
                    
                    # cos_theta = dot(u, v)
                    cos_cut = np.dot(dir_cue_to_ghost, dir_ghost_to_target)
                    cut_angle_rad = np.arccos(np.clip(cos_cut, -1.0, 1.0))
                    cut_angle_deg = np.degrees(cut_angle_rad)
                    
                    # 剔除切球角度过大的 (例如 > 80度)
                    if abs(cut_angle_deg) > 80:
                        continue
                        
                    # C. 碰撞检测
                    # Path 1: Target -> Pocket
                    # 障碍球: 除了 Target 和 Cue 之外的所有球 (包括黑8，除非它是目标)
                    obstacles_1 = [b for bid, b in balls.items() if bid != target_id and bid != 'cue' and b.state.s != 4]
                    if self._check_collision_path(target_pos, pocket_pos, obstacles_1, ball_radius):
                        continue # 目标球进袋路线被挡
                        
                    # Path 2: Cue -> Ghost Ball
                    # 障碍球: 除了 Cue 和 Target 之外的所有球
                    # 注意：目标球本身不能算作“白球到幽灵球”路径上的障碍，因为幽灵球就是紧贴目标球的
                    obstacles_2 = [b for bid, b in balls.items() if bid != target_id and bid != 'cue' and b.state.s != 4]
                    if self._check_collision_path(cue_pos, ghost_pos, obstacles_2, ball_radius):
                        continue # 白球击打路线被挡
                        
                    # D. 基础评分 (Heuristic Score)
                    # 距离越近分越高
                    # 角度越小分越高
                    score = 100.0
                    score -= dist_cue_to_ghost * 10.0 # 距离惩罚
                    score -= abs(cut_angle_deg) * 1.0 # 角度惩罚 (1度扣1分)
                    
                    # 计算物理瞄准角度 phi
                    aim_angle = self._calc_angle(cue_pos, ghost_pos)
                    
                    candidate_shots.append({
                        'score': score,
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'aim_angle': aim_angle,
                        'dist': dist_cue_to_ghost
                    })

            # 3. 如果没有候选路径，使用随机或保守策略
            if not candidate_shots:
                #print("[NewAgent] 未找到清晰的进攻路线，尝试解球或随机击球。")
                # 可以尝试找一个最近的球碰一下，避免犯规
                # 暂时Fallback到随机
                return self._random_action()
                
            # 4. 对 Top N 候选进行物理模拟微调
            # Sort by score descending
            candidate_shots.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = candidate_shots[:3] # 取前3名
            
            #print(f"[NewAgent] 筛选出 {len(candidate_shots)} 条路线，对前 {len(top_candidates)} 条进行物理微调...")
            
            best_action = None
            best_final_score = -float("inf")
            
            # 保存状态快照
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # 优化：进一步减少微调次数
            # 如果候选不多，且分数差异大，可以直接选第一名
            # 为了 Benchmark 速度，我们减少 num_iter
            SIM_ITER = 2 # 从 5 降到 2
            
            for shot_plan in top_candidates:
                base_phi = shot_plan['aim_angle']
                target_id = shot_plan['target_id']
                
                # 在 base_phi 附近微调
                # 距离越远，对角度越敏感。
                # 简单起见，采样 5-10 次
                
                # 速度估算: 距离越远力越大
                # 基础速度: 1.5 m/s + dist * 2.0
                base_v = 1.5 + shot_plan['dist'] * 2.5
                base_v = np.clip(base_v, 1.0, 7.0)
                
                for i in range(SIM_ITER): 
                    # 采样
                    phi = base_phi + random.uniform(-1.5, 1.5) # 小范围微调
                    V0 = base_v + random.uniform(-0.5, 1.0)
                    V0 = np.clip(V0, 0.5, 8.0)
                    
                    # 简单的杆法: 中杆或轻微低杆(拉杆)
                    theta = 0
                    a = 0
                    b = random.uniform(-0.1, 0.1)
                    
                    # 构建模拟
                    sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                    sim_table = copy.deepcopy(table)
                    cue = pt.Cue(cue_ball_id="cue")
                    shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                    shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    try:
                        pt.simulate(shot, inplace=True)
                        score = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                        
                        # 额外奖励：如果打进了计划中的球
                        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state_snapshot[bid].state.s != 4]
                        if target_id in new_pocketed:
                            score += 20 # 鼓励按计划执行
                            
                    except Exception:
                        score = -1000
                        
                    if score > best_final_score:
                        best_final_score = score
                        best_action = {
                            "V0": V0, "phi": phi, "theta": theta, "a": a, "b": b
                        }
            
            if best_action is None:
                return self._random_action()
                
            #print(f"[NewAgent] 最终决策 (预期得分 {best_final_score:.1f}): V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}")
            return best_action

        except Exception as e:
            print(f"[NewAgent] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
