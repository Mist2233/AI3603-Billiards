import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent

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

    def evaluate_position(self, table, my_targets, cue_ball_id="cue"):
        """
        评估当前盘面的好坏 (1-Step Lookahead)。
        分数越高，代表母球的位置越容易打到下一颗球。
        """
        try:
            # 1. 如果白球进袋了（洗袋），给极低分
            # 注意：在 pooltool 模拟后，球的状态如果 s==4 代表落袋
            # 但我们需要检查 cue ball 在模拟结束时的状态
            cue_ball = table.balls.get(cue_ball_id)
            if cue_ball is None or cue_ball.state.s == 4:
                return -1000.0

            cue_pos = cue_ball.state.rvw[0]
            
            # 增加：白球贴库惩罚
            # 如果白球距离库边太近 (比如 < 1.5 * R)，击球会非常困难
            # pooltool 桌面坐标范围通常是 x: [0, table.w], y: [0, table.l]
            # 球半径 R ~ 0.028575
            ball_radius = 0.028575
            margin = 1.5 * ball_radius
            
            if (cue_pos[0] < margin or cue_pos[0] > table.w - margin or
                cue_pos[1] < margin or cue_pos[1] > table.l - margin):
                # 贴库惩罚：扣除一定分数，避免这种位置
                # 但不至于像洗袋那样致命 (-1000)
                # 扣 30 分，相当于一次犯规的代价
                return -30.0

            # 2. 过滤已进袋的目标球
            valid_targets = []
            for bid in my_targets:
                b = table.balls.get(bid)
                if b and b.state.s != 4:
                    valid_targets.append(bid)
            
            if not valid_targets:
                return 1000.0 # 赢了（或者只剩黑8了，视作好局面）

            max_next_shot_quality = 0.0
            
            # 转换 balls 格式以供 _check_collision_path 使用
            
            for target_id in valid_targets:
                target_ball = table.balls[target_id]
                target_pos = target_ball.state.rvw[0]
                
                # 简单估算：距离
                dist = np.linalg.norm(target_pos - cue_pos)
                
                # 连通性检查 (是否被阻挡)
                obstacles = [b for bid, b in table.balls.items() if bid != target_id]
                is_blocked = self._check_collision_path(cue_pos, target_pos, obstacles)
                
                if not is_blocked:
                    # 这是一个可行的下一杆
                    # 距离越近越好，但也别太近不好运杆
                    if dist < 0.1: # 太近了
                        quality = 5.0
                    else:
                        quality = 10.0 / (dist + 0.5)
                    
                    if quality > max_next_shot_quality:
                        max_next_shot_quality = quality

            # 如果所有球都被挡住了，说明走位失误（被斯诺克），给罚分
            if max_next_shot_quality == 0:
                return -50.0

            return max_next_shot_quality * 5.0 # 权重调整
            
        except Exception as e:
            # 防御性编程
            return 0.0

    def _safety_action(self, balls, my_targets):
        """
        防守策略：当没有好的进攻机会时，尝试轻推最近的己方球，避免犯规。
        """
        try:
            cue_ball = balls["cue"]
            cue_pos = cue_ball.state.rvw[0]
            
            best_target = None
            min_dist = float("inf")
            
            # 找最近的有效目标球
            for tid in my_targets:
                if tid not in balls: continue
                t_pos = balls[tid].state.rvw[0]
                dist = np.linalg.norm(t_pos - cue_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_target = tid
            
            if best_target:
                t_pos = balls[best_target].state.rvw[0]
                # 计算瞄准角度 (直接瞄准球心)
                direction = t_pos - cue_pos
                phi = math.atan2(direction[1], direction[0])
                
                # 轻推：速度刚够碰到球
                # V0 估算: 0.5 + dist * 1.5
                V0 = 0.5 + min_dist * 1.5
                V0 = np.clip(V0, 0.5, 3.0)
                
                return {
                    "V0": V0,
                    "phi": math.degrees(phi),
                    "theta": 0, "a": 0, "b": 0
                }
        except Exception:
            pass
            
        return self._random_action()

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
        
        start_time = time.time()
        TIME_LIMIT = 2.5 # Leave 0.5s margin for safety (Total ~3s)

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
                # Time check
                if time.time() - start_time > TIME_LIMIT * 0.6: # If 60% time used in geometric search, stop early
                    if candidate_shots: # If we have candidates, stop looking for more
                        break

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
                # 尝试防守策略：轻推最近的球
                return self._safety_action(balls, my_targets)
                
            # 4. 对 Top N 候选进行物理模拟微调
            # Sort by score descending
            candidate_shots.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = candidate_shots[:3] # 取前3名
            
            best_action = None
            best_final_score = -float("inf")
            
            # 保存状态快照
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            for shot_plan in top_candidates:
                # Time Check
                if time.time() - start_time > TIME_LIMIT:
                    break

                base_phi = shot_plan['aim_angle']
                target_id = shot_plan['target_id']
                dist = shot_plan['dist']
                
                # 速度估算: 距离越远力越大
                base_v = 1.5 + dist * 2.5
                base_v = np.clip(base_v, 1.0, 7.0)
                
                # 启发式微调策略 (Smart Search)
                # 不再随机乱猜，而是有针对性地测试
                # 1. 标准角度
                # 2. 微调左
                # 3. 微调右
                if dist < 0.5:
                    angle_offsets = [0, -1.0, 1.0]
                else:
                    angle_offsets = [0, -0.4, 0.4] # 远距离更敏感
                
                # 速度微调：也可以尝试稍微大一点力，看走位是否更好
                # 但为了节省时间，固定用计算出的最佳力度
                
                for offset_phi in angle_offsets:
                    # 再次检查时间
                    if time.time() - start_time > TIME_LIMIT:
                        break
                        
                    phi = base_phi + offset_phi
                    V0 = base_v
                    
                    # 简单的杆法: 中杆
                    theta = 0
                    a = 0
                    b = 0
                    
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
                        
                        # 走位评估 (Positioning)
                        position_score = self.evaluate_position(shot.table, my_targets)
                        score += position_score
                            
                    except Exception:
                        score = -1000
                        
                    if score > best_final_score:
                        best_final_score = score
                        best_action = {
                            "V0": V0, "phi": phi, "theta": theta, "a": a, "b": b
                        }
                
                # Early Exit: 如果找到了很好的结果（比如进球且走位不错），就不浪费时间看后面的候选了
                if best_final_score > 120: 
                    # 基础分~100 + 进球奖励20 + 走位分(0~50)
                    # >120 说明肯定进球了，且走位不是极差
                    break
            
            if best_action is None:
                return self._random_action()
                
            #print(f"[NewAgent] 最终决策 (预期得分 {best_final_score:.1f}): V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}")
            return best_action

        except Exception as e:
            print(f"[NewAgent] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()