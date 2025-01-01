import threading

import models
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from PPO import *
from utils.utils import *
from Logger import getLogger
import game.game  as game

## 读取数据集
def getData():
    pass


## 读取运行参数
def getParse():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--eposide", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--config-path", type=str, default='config.yaml')
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--model", type=str, default='LeNet')
    parser.add_argument("--save-path", type=str, default='./models')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print('Cuda:',torch.cuda.is_available())
    paser = getParse()
    # 读取参数到变量
    eposide = paser.eposide
    epochs = paser.epochs
    configPath = paser.config_path
    num_games = paser.num_games

    # 定义配置对象
    config = readConfig(configPath)  # 配置
    dataSelector = game.DataSelector()

    # 设置训练设备信息
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置训练信息
    loss = nn.MSELoss()  # 损失函数，二范数损失

    getLogger().setConfig(config)
    ppo = PPO(config=config, device=device, loss=loss)
    for _ in range(config['eposide']*config['num_games']):
        ppo.start()
