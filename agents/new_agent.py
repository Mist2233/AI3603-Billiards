import math
import time
import copy
import random
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
import signal


class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action

class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class NewAgent(Agent):
    """自定义 Agent 实现：基于目标球导向的蒙特卡洛搜索"""

    def __init__(self):
        super().__init__()
        self.num_iter = 40
        self.c_puct = 1.25
        self.ball_radius = 0.028575
        self.sim_noise = {
            "V0": 0.10,
            "phi": 0.10,
            "theta": 0.10,
            "a": 0.003,
            "b": 0.003,
        }
        self._c_puct = 1.35
        self._sims_midgame = 90
        self._sims_endgame = 110
        self._sims_break = 110

    def _calc_angle(self, pos1, pos2):
        """计算从pos1指向pos2的角度（度）"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

    def _calc_distance(self, pos1, pos2):
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _clip_action(self, action):
        return {
            "V0": float(np.clip(action["V0"], 0.5, 8.0)),
            "phi": float(action["phi"] % 360),
            "theta": float(np.clip(action.get("theta", 0.0), 0.0, 90.0)),
            "a": float(np.clip(action.get("a", 0.0), -0.5, 0.5)),
            "b": float(np.clip(action.get("b", 0.0), -0.5, 0.5)),
        }

    def _add_noise(self, action):
        return self._clip_action(
            {
                "V0": action["V0"] + np.random.normal(0, self.sim_noise["V0"]),
                "phi": action["phi"] + np.random.normal(0, self.sim_noise["phi"]),
                "theta": action.get("theta", 0.0) + np.random.normal(0, self.sim_noise["theta"]),
                "a": action.get("a", 0.0) + np.random.normal(0, self.sim_noise["a"]),
                "b": action.get("b", 0.0) + np.random.normal(0, self.sim_noise["b"]),
            }
        )

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
            if dist_sq < (2 * ball_radius * 1.10) ** 2:
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
                obstacles = [
                    b
                    for bid, b in table.balls.items()
                    if bid not in (target_id, cue_ball_id) and b.state.s != 4
                ]
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
            remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
            targets = remaining if remaining else (["8"] if "8" in balls and balls["8"].state.s != 4 else [])
            for tid in targets:
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
                V0 = 1.2 + min_dist * 1.8
                V0 = np.clip(V0, 1.5, 4.0)
                
                return self._clip_action({
                    "V0": V0,
                    "phi": math.degrees(phi),
                    "theta": 0, "a": 0, "b": 0
                })
        except Exception:
            pass

        return {"V0": 1.0, "phi": 0.0, "theta": 0.0, "a": 0.0, "b": 0.0}

    def _is_break_state(self, balls, table):
        obj_positions = []
        for bid, b in balls.items():
            if bid in ("cue",) or b.state.s == 4:
                continue
            try:
                p = np.array(b.state.rvw[0], dtype=float)
                obj_positions.append(p[:2])
            except Exception:
                continue
        if len(obj_positions) < 12:
            return False
        pts = np.array(obj_positions)
        centroid = pts.mean(axis=0)
        spread = np.max(np.linalg.norm(pts - centroid, axis=1))
        cue = balls.get("cue")
        if cue is None:
            return False
        cue_pos = np.array(cue.state.rvw[0], dtype=float)[:2]
        cue_to_cluster = float(np.linalg.norm(cue_pos - centroid))
        return spread < 0.22 and cue_to_cluster > 0.45

    def _break_action(self, balls, table):
        obj_positions = []
        for bid, b in balls.items():
            if bid in ("cue",) or b.state.s == 4:
                continue
            p = np.array(b.state.rvw[0], dtype=float)
            obj_positions.append(p[:2])
        if not obj_positions:
            return self._clip_action({"V0": 6.5, "phi": 0.0, "theta": 0.0, "a": 0.0, "b": 0.0})
        centroid = np.array(obj_positions).mean(axis=0)
        cue_pos = np.array(balls["cue"].state.rvw[0], dtype=float)[:2]
        phi = math.degrees(math.atan2(centroid[1] - cue_pos[1], centroid[0] - cue_pos[0])) % 360
        return self._clip_action({"V0": 7.2, "phi": phi, "theta": 0.0, "a": 0.0, "b": 0.0})

    def _estimate_pot_prob(self, cut_angle_deg, dist_cue_to_ghost, dist_obj_to_pocket):
        angle_factor = (abs(cut_angle_deg) / 60.0) ** 2
        cue_dist_factor = dist_cue_to_ghost / 1.5
        obj_dist_factor = dist_obj_to_pocket / 1.5
        difficulty = angle_factor + 0.6 * cue_dist_factor + 0.6 * obj_dist_factor
        prob = math.exp(-2.2 * difficulty)
        return float(np.clip(prob, 0.0, 1.0))

    def _extra_penalty(self, shot, last_state, player_targets):
        try:
            new_pocketed = [
                bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4
            ]
            cue_pocketed = "cue" in new_pocketed or (shot.balls.get("cue") is not None and shot.balls["cue"].state.s == 4)
            eight_pocketed = "8" in new_pocketed or (shot.balls.get("8") is not None and shot.balls["8"].state.s == 4)

            valid_ball_ids = {
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
            }
            first_contact_ball_id = None
            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, "ids") else []
                if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
                    other_ids = [i for i in ids if i != "cue" and i in valid_ball_ids]
                    if other_ids:
                        first_contact_ball_id = other_ids[0]
                        break

            foul_first_hit = False
            if first_contact_ball_id is None:
                if len(last_state) > 2 or player_targets != ["8"]:
                    foul_first_hit = True
            else:
                if first_contact_ball_id not in player_targets:
                    foul_first_hit = True

            cue_hit_cushion = False
            target_hit_cushion = False
            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, "ids") else []
                if "cushion" in et:
                    if "cue" in ids:
                        cue_hit_cushion = True
                    if first_contact_ball_id is not None and first_contact_ball_id in ids:
                        target_hit_cushion = True

            foul_no_rail = False
            if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
                foul_no_rail = True

            extra = 0.0
            if cue_pocketed:
                extra -= 220.0
            if eight_pocketed and player_targets != ["8"]:
                extra -= 520.0
            if foul_first_hit:
                extra -= 90.0
            if foul_no_rail:
                extra -= 70.0
            return extra
        except Exception:
            return 0.0

    def _generate_candidate_actions(self, balls, my_targets, table):
        cue_ball = balls.get("cue")
        if cue_ball is None:
            return []
        cue_pos = cue_ball.state.rvw[0]
        ball_radius = getattr(cue_ball.params, "R", self.ball_radius)

        remaining_own = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        targets = remaining_own if remaining_own else ["8"]

        candidates = []
        for target_id in targets:
            if target_id not in balls or balls[target_id].state.s == 4:
                continue
            target_pos = balls[target_id].state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                ghost_pos = self._calculate_ghost_ball_pos(target_pos, pocket_pos, ball_radius)
                vec_cue_to_ghost = ghost_pos - cue_pos
                dist_cue_to_ghost = float(np.linalg.norm(vec_cue_to_ghost))
                if dist_cue_to_ghost < 1e-6:
                    continue

                vec_ghost_to_target = target_pos - ghost_pos
                dist_ghost_to_target = float(np.linalg.norm(vec_ghost_to_target))
                if dist_ghost_to_target < 1e-6:
                    continue

                dir_cue_to_ghost = vec_cue_to_ghost / dist_cue_to_ghost
                dir_ghost_to_target = vec_ghost_to_target / dist_ghost_to_target
                cos_cut = float(np.dot(dir_cue_to_ghost, dir_ghost_to_target))
                cut_angle_deg = float(np.degrees(np.arccos(np.clip(cos_cut, -1.0, 1.0))))
                if abs(cut_angle_deg) > 72:
                    continue

                obstacles_obj = [
                    b
                    for bid, b in balls.items()
                    if bid not in (target_id, "cue") and b.state.s != 4
                ]
                if self._check_collision_path(target_pos, pocket_pos, obstacles_obj, ball_radius):
                    continue
                if self._check_collision_path(cue_pos, ghost_pos, obstacles_obj, ball_radius):
                    continue

                dist_obj_to_pocket = float(np.linalg.norm(np.array(target_pos) - np.array(pocket_pos)))
                prob = self._estimate_pot_prob(cut_angle_deg, dist_cue_to_ghost, dist_obj_to_pocket)
                if prob < 0.22:
                    continue

                aim_phi = self._calc_angle(cue_pos, ghost_pos)
                base_v = 1.6 + 2.1 * dist_cue_to_ghost + 0.35 * dist_obj_to_pocket
                base_v = float(np.clip(base_v, 1.4, 6.8))

                pocket_weight = 1.0
                pid = str(pocket_id).lower()
                if "c" in pid:
                    pocket_weight = 0.88

                base_score = prob * 100.0 * pocket_weight
                if len(targets) == 1 and targets[0] == "8":
                    base_score *= 1.25

                candidates.append(
                    {
                        "base_score": base_score,
                        "action": {"V0": base_v, "phi": aim_phi, "theta": 0.0, "a": 0.0, "b": 0.0},
                        "target_id": target_id,
                    }
                )

        candidates.sort(key=lambda x: x["base_score"], reverse=True)
        best = candidates[:10]
        actions = []
        for c in best:
            a0 = c["action"]
            dist_scale = 0.5 if a0["V0"] < 2.0 else 0.35
            phi_offsets = [0.0, -0.6, 0.6] if dist_scale > 0.45 else [0.0, -0.35, 0.35]
            v_scales = [0.95, 1.0, 1.05]
            for dp in phi_offsets:
                for sv in v_scales:
                    actions.append(self._clip_action({"V0": a0["V0"] * sv, "phi": a0["phi"] + dp, "theta": 0.0, "a": 0.0, "b": 0.0}))
        if not actions:
            actions.append(self._safety_action(balls, my_targets))
        random.shuffle(actions)
        return actions[:36]

    def _generate_broad_actions(self, balls, my_targets, table):
        cue_ball = balls.get("cue")
        if cue_ball is None:
            return []
        cue_pos = np.array(cue_ball.state.rvw[0], dtype=float)

        remaining_own = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        targets = remaining_own if remaining_own else ["8"]

        actions = []
        for target_id in targets:
            if target_id not in balls or balls[target_id].state.s == 4:
                continue
            target_pos = np.array(balls[target_id].state.rvw[0], dtype=float)
            vec = target_pos - cue_pos
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                continue
            phi = float(np.degrees(np.arctan2(vec[1], vec[0])) % 360)
            base_v = float(np.clip(1.2 + dist * 2.4, 1.0, 7.6))

            phi_offsets = [-0.7, 0.0, 0.7] if dist < 0.9 else [-0.45, 0.0, 0.45]
            v_scales = [0.85, 1.0, 1.15]
            for dp in phi_offsets:
                for sv in v_scales:
                    actions.append(
                        self._clip_action(
                            {"V0": base_v * sv, "phi": phi + dp, "theta": 0.0, "a": 0.0, "b": 0.0}
                        )
                    )

        random.shuffle(actions)
        return actions[:36]

    def _simulate_action(self, balls, table, action):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        noisy_action = self._add_noise(action)
        shot.cue.set_state(
            V0=noisy_action["V0"],
            phi=noisy_action["phi"],
            theta=noisy_action["theta"],
            a=noisy_action["a"],
            b=noisy_action["b"],
        )
        ok = simulate_with_timeout(shot, timeout=3)
        return shot if ok else None

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            return self._random_action()

        try:
            targets = my_targets or []
            remaining_targets = [bid for bid in targets if bid in balls and balls[bid].state.s != 4]
            if not remaining_targets:
                if "8" in balls and balls["8"].state.s != 4:
                    remaining_targets = ["8"]
                else:
                    return self._safety_action(balls, targets)

            is_break = self._is_break_state(balls, table)
            is_endgame = (remaining_targets == ["8"]) or (len([b for b in remaining_targets if b != "8"]) <= 2)

            sims = self._sims_break if is_break else (self._sims_endgame if is_endgame else self._sims_midgame)
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            candidate_actions = []
            if is_break:
                base = self._break_action(balls, table)
                candidate_actions.extend(
                    [
                        base,
                        self._clip_action({**base, "V0": base["V0"] * 0.95}),
                        self._clip_action({**base, "V0": base["V0"] * 1.03}),
                        self._clip_action({**base, "phi": base["phi"] + 0.6}),
                        self._clip_action({**base, "phi": base["phi"] - 0.6}),
                    ]
                )
                for _ in range(6):
                    candidate_actions.append(self._random_action())
            else:
                candidate_actions.extend(self._generate_broad_actions(balls, remaining_targets, table))
                candidate_actions.extend(self._generate_candidate_actions(balls, remaining_targets, table))
                candidate_actions.append(self._safety_action(balls, remaining_targets))
                for _ in range(5):
                    candidate_actions.append(self._random_action())

            if not candidate_actions:
                return self._safety_action(balls, remaining_targets)

            n_candidates = len(candidate_actions)
            N = np.zeros(n_candidates, dtype=float)
            Q = np.zeros(n_candidates, dtype=float)

            for i in range(sims):
                if i < n_candidates:
                    idx = i
                else:
                    total_n = float(np.sum(N))
                    ucb_values = (Q / (N + 1e-6)) + self._c_puct * np.sqrt(np.log(total_n + 1.0) / (N + 1e-6))
                    idx = int(np.argmax(ucb_values))

                shot = self._simulate_action(balls, table, candidate_actions[idx])
                if shot is None:
                    raw_reward = -800.0
                else:
                    raw_reward = float(analyze_shot_for_reward(shot, last_state_snapshot, remaining_targets))
                    raw_reward += float(self._extra_penalty(shot, last_state_snapshot, remaining_targets))
                    raw_reward += 0.30 * float(self.evaluate_position(shot.table, remaining_targets))

                normalized_reward = (raw_reward - (-800.0)) / (300.0 - (-800.0))
                normalized_reward = float(np.clip(normalized_reward, 0.0, 1.0))

                N[idx] += 1.0
                Q[idx] += normalized_reward

            avg_rewards = Q / (N + 1e-6)
            best_idx = int(np.argmax(avg_rewards))
            best_action = candidate_actions[best_idx]

            if float(avg_rewards[best_idx]) < 0.52:
                return self._safety_action(balls, remaining_targets)

            return self._clip_action(best_action)

        except Exception as e:
            print(f"[NewAgent] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._safety_action(balls, my_targets)
