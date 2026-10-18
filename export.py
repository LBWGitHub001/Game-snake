import argparse
import os
import random
import time
import numpy as np
import torch

from configure import device
from client import GameAPI
from model import ActorCritic


def get_args():
    parser = argparse.ArgumentParser("Evaluate & Visualize Trained Snake AI")
    parser.add_argument("--model-path", type=str, default="./Models/AC.pth", help="Path to model weights")
    parser.add_argument("--episodes", type=int, default=5, help="Number of games to watch")
    parser.add_argument("--fps", type=int, default=12, help="Visualization speed (frames per second)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.model_path):
        # Fallback check for best_model.pth in same folder
        alt_path = os.path.join(os.path.dirname(args.model_path), "best_model.pth")
        if os.path.exists(alt_path):
            args.model_path = alt_path
        else:
            print(f"Error: Model checkpoint not found at '{args.model_path}'.")
            print("Please run 'python train.py' first to train a model.")
            return

    print(f"Loading model checkpoint from {args.model_path}...")
    model = ActorCritic().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    env = GameAPI()
    env.render()

    scores = []
    print("\nStarting evaluation games (close window or press Ctrl+C to stop)...")

    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            total_reward = 0.0

            while True:
                # Deterministic greedy action selection with Action Masking
                mask = env.get_action_mask()
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(state_tensor)
                    masked_logits = torch.where(mask_tensor, logits, -1e8)
                    action = torch.argmax(masked_logits, dim=-1).item()

                state, reward, done = env.stepOver(action)
                total_reward += reward

                # Render current frame
                running = env.show(fps=args.fps)
                if not running:
                    print("Window closed by user.")
                    return

                if done:
                    break

            scores.append(env.score)
            print(
                f"Game {ep}/{args.episodes} finished | "
                f"Score: {env.score} | "
                f"Steps: {env.steps} | "
                f"Reward: {total_reward:.1f}"
            )
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        env.close()

    if scores:
        print(f"\nEvaluation Summary:")
        print(f"  Average Score: {sum(scores) / len(scores):.2f}")
        print(f"  Max Score:     {max(scores)}")


if __name__ == "__main__":
    args = get_args()
    main(args)
