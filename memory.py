import os
import numpy as np
import torch


class Memory:
    """Standard transition buffer for single-environment PPO."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminal = []
        self.values = []
        self.masks = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.is_terminal[:]
        del self.values[:]
        del self.masks[:]

    def __len__(self):
        return len(self.states)


class OfflineRolloutBuffer:
    """Buffer to store rollouts from multiple parallel environments with Action Masks."""

    def __init__(self, steps_per_env, num_envs, state_shape=(3, 12, 12), device="cpu"):
        self.steps_per_env = steps_per_env
        self.num_envs = num_envs
        self.state_shape = state_shape
        self.device = device
        self.total_samples = steps_per_env * num_envs

        self.states = np.zeros((steps_per_env, num_envs, *state_shape), dtype=np.float32)
        self.actions = np.zeros((steps_per_env, num_envs), dtype=np.int64)
        self.rewards = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.dones = np.zeros((steps_per_env, num_envs), dtype=np.bool_)
        self.logprobs = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.values = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.masks = np.zeros((steps_per_env, num_envs, 3), dtype=np.bool_)

        self.step_idx = 0
        self.returns = None
        self.advantages = None

    def insert(self, state, action, reward, done, logprob, value, mask):
        """Insert a parallel step across all environments."""
        self.states[self.step_idx] = state
        self.actions[self.step_idx] = action
        self.rewards[self.step_idx] = reward
        self.dones[self.step_idx] = done
        self.logprobs[self.step_idx] = logprob
        self.values[self.step_idx] = value
        self.masks[self.step_idx] = mask
        self.step_idx += 1

    def compute_returns_and_advantages(self, last_values, last_dones, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        self.advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0.0

        for t in reversed(range(self.steps_per_env)):
            next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
            if t == self.steps_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size=128):
        """Flatten across steps and envs, yielding shuffled mini-batches with masks for offline PPO training."""
        b_states = self.states.reshape(-1, *self.state_shape)
        b_actions = self.actions.reshape(-1)
        b_logprobs = self.logprobs.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_masks = self.masks.reshape(-1, 3)

        # Normalize advantages
        adv_mean = b_advantages.mean()
        adv_std = b_advantages.std()
        if adv_std > 1e-7:
            b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-7)

        # Convert to tensors directly on device
        t_states = torch.tensor(b_states, dtype=torch.float32, device=self.device)
        t_actions = torch.tensor(b_actions, dtype=torch.long, device=self.device)
        t_logprobs = torch.tensor(b_logprobs, dtype=torch.float32, device=self.device)
        t_returns = torch.tensor(b_returns, dtype=torch.float32, device=self.device)
        t_advantages = torch.tensor(b_advantages, dtype=torch.float32, device=self.device)
        t_masks = torch.tensor(b_masks, dtype=torch.bool, device=self.device)

        total_size = len(b_states)
        indices = torch.randperm(total_size, device=self.device)

        for start_idx in range(0, total_size, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield (
                t_states[batch_idx],
                t_actions[batch_idx],
                t_logprobs[batch_idx],
                t_returns[batch_idx],
                t_advantages[batch_idx],
                t_masks[batch_idx],
            )

    def reset(self):
        """Reset step pointer."""
        self.step_idx = 0
        self.returns = None
        self.advantages = None

    def save_dataset(self, filepath):
        """Export collected offline transitions to an .npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            logprobs=self.logprobs,
            values=self.values,
            masks=self.masks,
            returns=self.returns,
            advantages=self.advantages,
        )
        print(f"Exported offline dataset ({self.total_samples} samples) to {filepath}")
