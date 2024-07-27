from configure import *
import torch.optim as optim
from torch import nn
from model import ActorCritic
class PPO:
    def __init__(self, lr, betas, gamma, epochs, eps, state_size=(12, 12), action_size=3):
        #参数获取
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        self.lr = lr
        self.betas = betas
        #定义网络
        self.policy = ActorCritic().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        #定义损失函数
        self.loss = nn.MSELoss()

    def update(self,memory):
        rewards = []

