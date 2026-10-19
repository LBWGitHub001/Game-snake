import argparse
import os
import random
import time
import numpy as np
import torch

from configure import device
from client import ParallelGameEnv, GameAPI
from memory import OfflineRolloutBuffer
from PPO import PPO


def get_args():
    parser = argparse.ArgumentParser("Multi-Threaded Vectorized PPO Snake Training")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of concurrent game environments/threads")
    parser.add_argument("--steps-per-env", type=int, default=256, help="Rollout steps per env before offline PPO update")
    parser.add_argument("--total-timesteps", type=int, default=300000, help="Total environment interaction steps")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate (Adam)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--eps", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy bonus coefficient to prevent circular policy collapse")
    parser.add_argument("--epochs", type=int, default=4, help="Offline training epochs per rollout buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for offline update")
    parser.add_argument("--save-path", type=str, default="./Models/AC.pth", help="Checkpoint save path")
    parser.add_argument("--save-freq", type=int, default=20, help="Save model checkpoint every N rollout iterations")
    parser.add_argument("--save-dataset", type=str, default=None, help="Optional path to export offline dataset (.npz)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    batch_samples = args.num_envs * args.steps_per_env
    total_iterations = max(1, args.total_timesteps // batch_samples)

    print("======================================================================")
    print(" 🚀 Starting Multi-Threaded Offline PPO Snake Training")
    print(f" - Compute Device:         {device}")
    print(f" - Concurrent Games/Threads:{args.num_envs}")
    print(f" - Steps Per Game:         {args.steps_per_env}")
    print(f" - Offline Batch Samples:   {batch_samples} transitions / iteration")
    print(f" - Total Iterations:       {total_iterations} (~{args.total_timesteps} steps)")
    print(f" - Offline Update Epochs:  {args.epochs} (batch size {args.batch_size})")
    print(f" - LR / Gamma / Ent-Coef:  {args.lr} / {args.gamma} / {args.ent_coef}")
    print(f" - Checkpoint Save Path:   {args.save_path}")
    print("======================================================================")

    # Initialize parallel multi-threaded environments and buffer
    vec_env = ParallelGameEnv(num_envs=args.num_envs)
    buffer = OfflineRolloutBuffer(
        steps_per_env=args.steps_per_env,
        num_envs=args.num_envs,
        state_shape=(3, 12, 12),
        device=device,
    )

    ppo = PPO(
        lr=args.lr,
        betas=(0.9, 0.999),
        gamma=args.gamma,
        epochs=args.epochs,
        eps=args.eps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        device=device,
    )

    states, masks = vec_env.reset()
    last_dones = np.zeros(args.num_envs, dtype=np.bool_)

    total_steps = 0
    recent_scores = []
    best_score = 0
    start_time = time.time()

    try:
        for iteration in range(1, total_iterations + 1):
            iter_start_time = time.time()
            buffer.reset()

            # ----------------------------------------------------
            # Phase 1: Multi-threaded parallel rollout collection
            # ----------------------------------------------------
            for step in range(args.steps_per_env):
                total_steps += args.num_envs

                # Batched policy forward pass with Action Masking
                actions, logprobs, values = ppo.policy_old.act_batch(states, masks=masks)

                # Parallel environment step across threads
                next_states, rewards, dones, next_masks, infos = vec_env.step(actions)

                # Store transitions into offline buffer (including masks)
                buffer.insert(states, actions, rewards, dones, logprobs, values, masks)

                # Collect completed game scores
                for info in infos:
                    if info["is_dead"]:
                        score = info["score"]
                        recent_scores.append(score)
                        if score > best_score:
                            best_score = score
                            best_path = os.path.join(os.path.dirname(args.save_path), "best_model.pth")
                            ppo.save(best_path)

                states = next_states
                masks = next_masks
                last_dones = dones

            # Compute bootstrap value for GAE
            _, _, last_values = ppo.policy_old.act_batch(states, masks=masks)
            buffer.compute_returns_and_advantages(
                last_values=last_values,
                last_dones=last_dones,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

            # Optional dataset export
            if args.save_dataset is not None and iteration == total_iterations:
                buffer.save_dataset(args.save_dataset)

            # ----------------------------------------------------
            # Phase 2: Offline / Batch PPO policy optimization
            # ----------------------------------------------------
            avg_loss = ppo.update_offline(
                buffer=buffer,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            # Compute training statistics
            iter_time = time.time() - iter_start_time
            sps = int(batch_samples / iter_time) if iter_time > 0 else 0

            if len(recent_scores) > 100:
                recent_scores = recent_scores[-100:]
            avg_score = float(np.mean(recent_scores)) if recent_scores else 0.0

            print(
                f"Iter: {iteration:4d}/{total_iterations} | "
                f"Steps: {total_steps:7d} | "
                f"SPS: {sps:5d} | "
                f"Avg100 Score: {avg_score:4.2f} | "
                f"Best: {best_score:2d} | "
                f"Loss: {avg_loss:6.3f}"
            )

            # Periodic saving
            if iteration % args.save_freq == 0:
                ppo.save(args.save_path)
                print(f"--> [Checkpoint] Saved model to {args.save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")
    finally:
        ppo.save(args.save_path)
        vec_env.close()
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s! Final model saved to {args.save_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)
