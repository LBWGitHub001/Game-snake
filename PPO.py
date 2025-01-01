import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from models import Model
from utils import *
from threading import Lock
from game.game import *
from game.gameEngine import GameEngine

from Logger import getLogger

class PPO:
    def __init__(self, config, device, loss):
        self.config = config
        # 参数获取
        # self.state_size = state_size
        # self.action_size = action_size
        self.gamma = config['gamma']
        self.epochs = config['epoch']
        self.eps = config['eps']
        # self.lr = lr
        # self.betas = betas
        self.timestep = config['timestep']
        # 定义网络
        self.device = device
        self.policy = Model(config).to(device=device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))
        # 定义损失函数
        self.loss = loss
        # 定义数据收集器
        self.DataSelector = DataSelector()  # 数据收集器
        self.dataModelList = list()  # 模型线程
        for i in range(config['num_games']):  # 循环创建游戏线程
            dataModel = Model(config).to(device=device)
            dataThread = DataBlockThread(id=i,
                                         config=config,
                                         dataModel=dataModel,
                                         dataSelector=[self.DataSelector],
                                         device=device
                                         )
            self.dataModelList.append(dataThread)

        self.DataSelector.setDataModelList(self.dataModelList)  # 将数据收集器和模型列表链接

        for dataModel in self.dataModelList:
            dataModel.setTimestep(config['timestep'])
            thread = dataModel.createThread()
            thread.start()

    def start(self):
        # 使用蒙特卡洛截断估计奖励
        cvParse.acquire()
        cvParse.wait()
        block = self.DataSelector.read()
        data=block.data
        threadId = block.id

        old_states = data[0]
        old_actions = data[1]
        old_rewards = data[2]
        dist = Categorical(old_actions)  # 按照概率进行采样
        old_logprobs = dist.log_prob(dist.sample())
        # 标准化
        rewards = (old_rewards - old_rewards.mean()) / (old_rewards.std() + 1e-5)

        # 更新模型
        for _ in range(self.epochs):
            # 评价久数据
            logprobs, state_value, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算比率PPO2
            ratios = torch.exp(old_logprobs - old_logprobs.detach())

            # 计算Loss
            advantages = rewards - state_value.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            # TD差分
            DR_T = (torch.ones((self.timestep)) * 1).to(self.device).detach()
            loss = -torch.min(surr1, surr2) + self.loss(state_value, rewards) * 0.5 - 0.01 * dist_entropy
            loss = loss.mean()
            # 开始学习，更新参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            getLogger().setEpoch()
            getLogger().printInfo()
        getLogger().setEposide()

        # 复制旧数据
        cvParse.notify()
        cvParse.release()
        self.DataSelector.updateModel(threadId,self.policy.state_dict())