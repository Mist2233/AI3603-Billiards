import pooltool as pt
from poolenv import PoolEnv
from agent import NewAgent


def test_new_agent():
    print("Testing NewAgent...")
    env = PoolEnv()
    env.reset(target_ball="solid")

    agent = NewAgent()

    # Get observation for Player A (who is current player after reset)
    player = env.get_curr_player()
    obs = env.get_observation(player)

    print(f"Current player: {player}")
    print(f"Targets: {obs[1]}")

    # Make decision
    print("Calling agent.decision()...")
    action = agent.decision(*obs)

    print("Action received:")
    print(action)

    # Verify keys
    required_keys = ["V0", "phi", "theta", "a", "b"]
    missing_keys = [k for k in required_keys if k not in action]

    if missing_keys:
        print(f"FAILED: Missing keys in action: {missing_keys}")
    else:
        print("SUCCESS: Action format is correct.")

    # Try to take shot
    print("Taking shot in environment...")
    res = env.take_shot(action)
    print("Shot result keys:", res.keys())
    print("Test Complete.")


if __name__ == "__main__":
    test_new_agent()
