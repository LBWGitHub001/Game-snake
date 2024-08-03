import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from configure import device

'''
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # Actor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=3)
    def action_layer(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=-1)
        return x
'''
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # critic
        self.vconv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.vconv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.vfc1 = nn.Linear(in_features=256, out_features=64)
        self.vfc2 = nn.Linear(in_features=64, out_features=16)
        self.vfc3 = nn.Linear(in_features=16, out_features=1)

    def value_layer(self, x):
        x = F.relu(self.vconv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.vconv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 256)
        x = F.relu(self.vfc1(x))
        x = F.relu(self.vfc2(x))
        x = F.relu(self.vfc3(x))
        return x


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # Actor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=3)

        # critic
        self.vconv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.vconv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.vfc1 = nn.Linear(in_features=256, out_features=64)
        self.vfc2 = nn.Linear(in_features=64, out_features=16)
        self.vfc3 = nn.Linear(in_features=16, out_features=1)

    def action_layer(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=-1)
        return x

    def value_layer(self, x):
        x = F.relu(self.vconv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.vconv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 256)
        x = F.relu(self.vfc1(x))
        x = F.relu(self.vfc2(x))
        x = F.relu(self.vfc3(x))
        return x

    def forward(self, x):
        x = self.action_layer(x)
        x = self.value_layer(x)
        return x


    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)  # 按照概率进行采样
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_values = self.value_layer(state)

        return action_log_probs, torch.squeeze(dist_entropy), state_values
