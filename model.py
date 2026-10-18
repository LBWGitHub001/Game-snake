import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from configure import device


class ActorCritic(nn.Module):
    """Actor-Critic network for Snake game using a shared CNN backbone.

    - Input: State grid tensor of shape (batch_size, 3, 12, 12) or (3, 12, 12)
    - Channels:
        Channel 0: Snake Head (binary/normalized)
        Channel 1: Snake Body (decaying gradient from neck to tail)
        Channel 2: Food (binary)
    - Outputs:
        Actor: 3 action logits (0: Turn Left, 1: Straight, 2: Turn Right)
        Critic: State value V(s)
    """

    def __init__(self, in_channels=3, action_dim=3):
        super().__init__()
        # Shared CNN feature extractor (preserves 12x12 spatial layout)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12, 256),
            nn.ReLU()
        )

        # Heads
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        feat = self.fc(self.conv(x))
        action_logits = self.actor(feat)
        state_values = self.critic(feat)
        return action_logits, state_values

    def action_layer(self, x):
        """Returns softmax action probabilities (for backward compatibility)."""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)

    def value_layer(self, x):
        """Returns state values (for backward compatibility)."""
        _, values = self.forward(x)
        return values

    def act(self, state, mask=None, memory=None):
        """Select action during single-environment interaction."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device=device, dtype=torch.float32)

        if state.dim() == 3:
            state_input = state.unsqueeze(0)
        else:
            state_input = state

        with torch.no_grad():
            logits, value = self.forward(state_input)
            if mask is not None:
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
                if mask_t.dim() == 1:
                    mask_t = mask_t.unsqueeze(0)
                logits = torch.where(mask_t, logits, -1e8)

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        act_idx = action.item()
        act_log_prob = log_prob.item()
        val = value.squeeze(-1).item()

        if memory is not None:
            store_state = state.squeeze(0) if state.dim() == 4 else state
            memory.states.append(store_state.detach().cpu())
            memory.actions.append(act_idx)
            memory.logprobs.append(act_log_prob)
            memory.values.append(val)
            if hasattr(memory, "masks"):
                memory.masks.append(mask)

        return act_idx

    def act_batch(self, states, masks=None):
        """Batched action selection for parallel multi-environment collection.

        Args:
            states: np.ndarray or torch.Tensor of shape (N, 3, 12, 12)
            masks: np.ndarray or torch.Tensor of shape (N, 3)

        Returns:
            actions (np.ndarray): (N,)
            logprobs (np.ndarray): (N,)
            values (np.ndarray): (N,)
        """
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32, device=device)
        else:
            states = states.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            logits, values = self.forward(states)
            if masks is not None:
                masks_t = torch.as_tensor(masks, dtype=torch.bool, device=logits.device)
                logits = torch.where(masks_t, logits, -1e8)

            dist = Categorical(logits=logits)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            logprobs.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
        )

    def evaluate(self, states, actions, masks=None):
        """Evaluate batch of states and actions during PPO update."""
        logits, values = self.forward(states)
        if masks is not None:
            masks_t = torch.as_tensor(masks, dtype=torch.bool, device=logits.device)
            logits = torch.where(masks_t, logits, -1e8)

        dist = Categorical(logits=logits)

        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = values.squeeze(-1)

        return action_log_probs, dist_entropy, state_values

