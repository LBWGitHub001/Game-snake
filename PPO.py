import torch

from configure import *
import torch.optim as optim
from model import ActorCritic
from torch import nn


class PPO:
    def __init__(self, lr, betas, gamma, epochs, eps, timestep, state_size=(12, 12), action_size=3):
        # 参数获取
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        self.lr = lr
        self.betas = betas
        self.timestep = timestep
        # 定义网络
        self.policy = ActorCritic().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        #self.optimizerC = optim.Adam(self.policy.Critic.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 定义损失函数
        self.loss = nn.MSELoss()

    def update(self, memory):
        # 使用蒙特卡洛截断估计奖励
        rewards = []
        discounted_rewards = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminal)):
            if is_terminal: #如果gameover了，那么没有奖励
                discounted_rewards = 0
            discounted_rewards = reward + discounted_rewards * self.gamma
            rewards.insert(0, discounted_rewards)
        # 标准化
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        #list转tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        #更新模型
        for _ in range(self.epochs):
            # 评价久数据
            logprobs, state_value, dist_entropy = self.policy.evaluate(old_states, old_actions)

            #计算比率PPO2
            ratios = torch.exp(old_logprobs - old_logprobs.detach())

            #计算Loss
            advantages = rewards - state_value.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            #TD差分
            DR_T = (torch.ones((self.timestep))*discounted_rewards).to(device).detach()
            loss = -torch.min(surr1, surr2) + self.loss(state_value, rewards)*0.5 - 0.01*dist_entropy
            loss = loss.mean()
            #开始学习，更新参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        #复制旧数据
        self.policy_old.load_state_dict(self.policy.state_dict())

