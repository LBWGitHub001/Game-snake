import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.config = config
        actor=config['actor']
        critic=config['critic']
        # Actor
        self.conv1 = nn.Conv2d(in_channels=actor['cnn1']['in_channels'],
                               out_channels=actor['cnn1']['out_channels'],
                               kernel_size=actor['cnn1']['kernel_size'], padding=1)
        self.conv2 = nn.Conv2d(in_channels=actor['cnn2']['in_channels'],
                               out_channels=actor['cnn2']['out_channels'],
                               kernel_size=actor['cnn2']['kernel_size'], padding=1)
        self.fc1 = nn.Linear(in_features=actor['liner1']['in_features'],
                             out_features=actor['liner1']['out_features'])
        self.fc2 = nn.Linear(in_features=actor['liner2']['in_features'],
                             out_features=actor['liner2']['out_features'])

        # critic
        self.vconv1 = nn.Conv2d(in_channels=critic['cnn1']['in_channels'],
                               out_channels=critic['cnn1']['out_channels'],
                               kernel_size=critic['cnn1']['kernel_size'], padding=1)
        self.vconv2 = nn.Conv2d(in_channels=critic['cnn2']['in_channels'],
                                out_channels=critic['cnn2']['out_channels'],
                                kernel_size=critic['cnn2']['kernel_size'], padding=1)
        self.vfc1 = nn.Linear(in_features=critic['liner1']['in_features'],
                             out_features=critic['liner1']['out_features'])
        self.vfc2 = nn.Linear(in_features=critic['liner2']['in_features'],
                              out_features=critic['liner2']['out_features'])
        self.vfc3 = nn.Linear(in_features=critic['liner3']['in_features'],
                              out_features=critic['liner3']['out_features'])

    def action_layer(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=-1)
        return x

    def value_layer(self, x):
        x = self.vconv1(x)
        x = F.relu(x)
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
        state = torch.from_numpy(state).float()#.to(device)
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