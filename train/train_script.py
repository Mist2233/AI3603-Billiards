import argparse
import os
import sys

# Add project root to path to import poolenv and agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poolenv import PoolEnv
from agent import NewAgent, BasicAgent


def train(args):
    print(f"Starting training for {args.episodes} episodes...")
    print(f"Checkpoints will be saved to {args.save_dir}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    env = PoolEnv()
    agent = NewAgent()

    # Placeholder for training loop
    for i in range(args.episodes):
        # 1. Reset environment
        # 2. Loop until done
        # 3. Collect data / Update policy

        if (i + 1) % 100 == 0:
            print(f"Episode {i+1}/{args.episodes} completed.")
            # Save checkpoint
            # agent.save(os.path.join(args.save_dir, f"checkpoint_{i+1}.pt"))

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Billiards Agent")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training episodes"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()
    train(args)
