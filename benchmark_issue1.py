
import os
import sys
import numpy as np
from poolenv import PoolEnv
from agent import NewAgent, BasicAgent

def benchmark_agent(num_episodes=20):
    """
    运行简短的测试来评估 NewAgent 的表现
    """
    env = PoolEnv()
    
    # 我们主要测试 NewAgent
    # 为了简化，我们让 BasicAgent 作为对手（虽然它目前也是比较弱的）
    agent_a = BasicAgent()
    agent_b = NewAgent()
    
    results = {
        'NewAgent_Win': 0,
        'BasicAgent_Win': 0,
        'Draw': 0,
        'Total_Shots': 0,
        'Fouls': 0
    }
    
    print(f"Starting benchmark for {num_episodes} episodes...")
    
    # 模拟 evaluate.py 的先后手轮换机制
    players = [agent_a, agent_b]
    target_ball_choice = ["solid", "solid", "stripe", "stripe"]
    
    for i in range(num_episodes):
        print()
        print(f"------- 第 {i} 局比赛开始 -------")
        # 决定谁打什么球 (与 evaluate.py 一致)
        current_target_ball = target_ball_choice[i % 4]
        try:
            obs = env.reset(target_ball=current_target_ball)
        except TypeError:
            obs = env.reset(target_ball=current_target_ball)
            
        player_class = players[i % 2].__class__.__name__
        print(f"本局 Player A: {player_class}, 目标球型: {current_target_ball}")
            
        done = False
        step_count = 0
        
        while not done:
            player_id = env.get_curr_player() # 'A' or 'B'
            print(f"[第{env.hit_count}次击球] player: {player_id}")
            
            # 关键修复：根据 evaluate.py 的逻辑选择 Agent
            # i % 2 == 0: Agent A (Basic) is Player A, Agent B (New) is Player B
            if player_id == "A":
                current_agent = players[i % 2]
            else:
                current_agent = players[(i + 1) % 2]
            
            # 获取观测
            obs = env.get_observation(player_id)
            # 决策
            action = current_agent.decision(*obs)
            
            # 打印决策信息 (Optional)
            if isinstance(current_agent, NewAgent):
                # print(f"NewAgent Action: V0={action['V0']:.2f}, phi={action['phi']:.2f}")
                pass
                
            # 执行
            step_info = env.take_shot(action)
            
            # 检查结束
            done, info = env.get_done()
            
            if done:
                winner = info['winner']
                print(f"Game Over. Winner: {winner}")
                
                if winner == 'SAME':
                    results['Draw'] += 1
                elif winner == 'A':
                    # 如果 Player A 赢了，看谁是 Player A
                    if i % 2 == 0:
                        results['BasicAgent_Win'] += 1 # Basic was A
                    else:
                        results['NewAgent_Win'] += 1   # New was A
                elif winner == 'B':
                    # 如果 Player B 赢了，看谁是 Player B
                    if i % 2 == 0:
                        results['NewAgent_Win'] += 1   # New was B
                    else:
                        results['BasicAgent_Win'] += 1 # Basic was B

    # 计算胜率 (NewAgent)
    win_rate = (results['NewAgent_Win'] + 0.5 * results['Draw']) / num_episodes
    print("\nBenchmark Results:")
    print(f"NewAgent Win Rate: {win_rate:.2%}")
    print(f"Details: {results}")

if __name__ == "__main__":
    benchmark_agent()
