import os
import torch
import torch.nn as nn
import torch.optim as optim
from configure import device
from model import ActorCritic


class PPO:
    """Proximal Policy Optimization (PPO-Clip) Agent."""

    def __init__(
        self,
        lr=0.0003,
        betas=(0.9, 0.999),
        gamma=0.99,
        epochs=4,
        eps=0.2,
        timestep=2000,
        batch_size=64,
        state_size=(3, 12, 12),
        action_size=3,
        ent_coef=0.02,
        device=device
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        self.timestep = timestep
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.ent_coef = ent_coef
        self.device = device

        # Networks: current policy and rollout policy
        self.policy = ActorCritic(in_channels=state_size[0], action_dim=action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic(in_channels=state_size[0], action_dim=action_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

    def update(self, memory):
        """Update policy and value network using collected trajectories."""
        if len(memory) == 0:
            return

        # 1. Calculate Monte Carlo discounted returns
        returns = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminal)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 2. Prepare tensors
        old_states = torch.stack(memory.states).to(self.device)
        old_actions = torch.tensor(memory.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32, device=self.device)
        old_masks = (
            torch.tensor(np.array(memory.masks), dtype=torch.bool, device=self.device)
            if hasattr(memory, "masks") and len(memory.masks) == len(memory.actions)
            else None
        )

        # 3. Compute baseline values and advantages
        if len(memory.values) == len(returns):
            old_values = torch.tensor(memory.values, dtype=torch.float32, device=self.device)
            advantages = returns - old_values
        else:
            with torch.no_grad():
                _, _, old_values = self.policy_old.evaluate(old_states, old_actions, masks=old_masks)
                advantages = returns - old_values

        # Normalize advantages for training stability
        if advantages.numel() > 1 and advantages.std() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 4. Mini-batch PPO Updates
        dataset_size = old_states.size(0)
        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]

                b_states = old_states[batch_idx]
                b_actions = old_actions[batch_idx]
                b_old_logprobs = old_logprobs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]
                b_masks = old_masks[batch_idx] if old_masks is not None else None

                logprobs, dist_entropy, state_values = self.policy.evaluate(
                    b_states, b_actions, masks=b_masks
                )

                # Ratio: r(theta) = exp(log pi_new - log pi_old)
                ratios = torch.exp(logprobs - b_old_logprobs)

                # Surrogate Loss with PPO clipping
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps, 1.0 + self.eps) * b_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.loss_fn(state_values, b_returns)
                entropy_loss = -dist_entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        # 5. Synchronize old policy with updated policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update_offline(self, buffer, epochs=4, batch_size=128):
        """Perform offline / batch PPO training on the collected multi-environment rollout buffer with Action Masking."""
        loss_history = []
        for _ in range(epochs):
            for (
                b_states,
                b_actions,
                b_old_logprobs,
                b_returns,
                b_advantages,
                b_masks,
            ) in buffer.get_batches(batch_size):
                logprobs, dist_entropy, state_values = self.policy.evaluate(
                    b_states, b_actions, masks=b_masks
                )

                # Ratio: r(theta) = exp(log pi_new - log pi_old)
                ratios = torch.exp(logprobs - b_old_logprobs)

                # Surrogate Loss with PPO clipping
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps, 1.0 + self.eps) * b_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.loss_fn(state_values, b_returns)
                entropy_loss = -dist_entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                loss_history.append(loss.item())

        # Synchronize rollout policy with updated policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        return float(torch.tensor(loss_history).mean().item()) if loss_history else 0.0

    def save(self, pth):
        os.makedirs(os.path.dirname(os.path.abspath(pth)), exist_ok=True)
        torch.save(self.policy.state_dict(), pth)

    def load(self, pth):
        state_dict = torch.load(pth, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy_old.load_state_dict(state_dict)