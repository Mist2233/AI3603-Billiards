
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
    
    for i in range(num_episodes):
        # 决定谁打什么球
        target_ball_type = 'solid' if i % 2 == 0 else 'stripe'
        try:
            obs = env.reset(target_ball=target_ball_type)
        except TypeError:
            # Fallback for old poolenv version if needed, but we saw code requires target_ball
            obs = env.reset(target_ball=target_ball_type)
            
        done = False
        step_count = 0
        
        while not done:
            current_player_idx = env.curr_player # 0 or 1
            # 假设 idx 0 是 Agent A (Basic), idx 1 是 Agent B (New)
            # 根据 evaluate.py 的逻辑，通常偶数局 Agent A 先手，奇数局 Agent B 先手
            
            # 实际上 PoolEnv 内部维护了 turn
            
            # obs['balls'] might be None in some env implementations if not returned by reset/step
            # But looking at poolenv.py, reset returns nothing? Wait.
            # poolenv.py:224 reset ends without return?
            # Let's check poolenv.py reset return value.
            
            balls_data = env.balls
            player_targets = env.player_targets
            
            if current_player_idx == 0:
                action = agent_a.decision(
                    balls=balls_data,
                    my_targets=player_targets['A'],
                    table=env.table
                )
            else:
                action = agent_b.decision(
                    balls=balls_data,
                    my_targets=player_targets['B'],
                    table=env.table
                )
                
            step_result = env.take_shot(action)
            # take_shot returns dict
            
            done, info = env.get_done()
            
            step_count += 1
            
            if current_player_idx == 1:
                results['Total_Shots'] += 1
                # Check for foul in current step result
                # step_result is dict with keys: 'ME_INTO_POCKET', 'ENEMY_INTO_POCKET', 'WHITE_BALL_INTO_POCKET', 'BLACK_BALL_INTO_POCKET', 'FOUL_FIRST_HIT', 'NO_POCKET_NO_RAIL', 'NO_HIT', 'BALLS'
                
                is_foul = False
                if step_result.get('WHITE_BALL_INTO_POCKET', False):
                    is_foul = True
                elif step_result.get('FOUL_FIRST_HIT', False):
                    is_foul = True
                elif step_result.get('NO_POCKET_NO_RAIL', False):
                    # In some rules this is a foul, but poolenv.py logic seems to just swap turns?
                    # Let's check poolenv.py logic.
                    # line 395: returns 'NO_HIT': True, which implies foul/swap turn usually.
                    # line 380: self.curr_player = 1 - self.curr_player (swaps turn)
                    pass
                elif step_result.get('NO_HIT', False):
                     # poolenv line 379: "本杆白球未接触任何球，交换球权" -> usually a foul in pool
                     is_foul = True
                
                if is_foul:
                    results['Fouls'] += 1
            
        winner = env.winner
        # env.winner returns 'A' or 'B' or 'SAME' or None
        
        # We need to map 'A'/'B' to 0/1 based on who was who.
        # But wait, we fixed A=Basic, B=NewAgent in our loop logic above:
        # if current_player_idx == 0: agent_a (Basic)
        # else: agent_b (New)
        # And poolenv usually maps idx 0 -> 'A', idx 1 -> 'B' (see poolenv.py:232)
        
        if winner == 'B': # NewAgent won
            results['NewAgent_Win'] += 1
        elif winner == 'A': # BasicAgent won
            results['BasicAgent_Win'] += 1
        else:
            results['Draw'] += 1
            
        print(f"Episode {i+1}/{num_episodes} finished. Winner: {'NewAgent' if winner==1 else 'BasicAgent' if winner==0 else 'Draw'}")

    print("\nBenchmark Results:")
    print(f"NewAgent Wins: {results['NewAgent_Win']} ({results['NewAgent_Win']/num_episodes*100:.1f}%)")
    print(f"BasicAgent Wins: {results['BasicAgent_Win']} ({results['BasicAgent_Win']/num_episodes*100:.1f}%)")
    print(f"Foul Rate (NewAgent): {results['Fouls']}/{results['Total_Shots']} ({results['Fouls']/max(1, results['Total_Shots'])*100:.1f}%)")

if __name__ == "__main__":
    benchmark_agent()
