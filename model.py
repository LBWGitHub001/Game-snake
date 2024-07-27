import torch
import torch.nn as nn
import torch.nn.functional as F
from configure import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Actor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=3)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

    def save(self,path):
        torch.save(self.state_dict(), path)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        #Actor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64, out_features=32)

        #critic
        self.vfc1 = nn.Linear(in_features=32, out_features=16)
        self.vfc2 = nn.Linear(in_features=16, out_features=16)
        self.vfc3 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = self.action_layer(x)
        x = self.value_layer(x)
        return x

    def action_layer(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return x

    def value_layer(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.vfc1(x))
        x = F.relu(self.vfc2(x))
        x = self.vfc3(x)
        return x

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_network(state)
