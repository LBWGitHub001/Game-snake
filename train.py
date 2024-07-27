import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import CNN
from configure import *
from PPO import PPO


def main(args):
    ppo = PPO(args.lr, (args.bate1, args.bate2), args.gamma, args.epochs, args.eps)
    get_GameManager(args.num_game)
    #定义变量
    running_reward = 0
    avg_length = 0
    timestep = 0

    for eposide in range(1,args.eposides+1):
        #启动游戏节点
        state = GMM[0].reset()
        for step in range(args.timestep):
            timestep += 1

            #运行policy_old
            action = ppo.policy_old.act(state,memory)

def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--timestep", type=int, default=2000)  # 收集多少数据之后开始学习
    parser.add_argument("--episodes", type=int, default=50000)  # 玩多少遍游戏
    parser.add_argument("--lr", type=float, default=0.03)  # 学习率
    parser.add_argument("--gamma", type=int, default=0.99)  # 折扣因子
    parser.add_argument("--save-path", type=str, default='./models')
    parser.add_argument("--num-game", type=int, default=1)  # 游戏线程
    parser.add_argument("--epochs", type=int, default=4)  # 一次数据训练次数
    parser.add_argument("--seed", type=int, default=None)  # 随机种子
    parser.add_argument("--eps", type=float, default=0.2)  # PPO2算法中的界
    parser.add_argument("--bate1", type=float, default=0.2)  # Adam动量的位置惯性系数
    parser.add_argument("--bate2", type=float, default=0.2)  # Adam动量的速度惯性系数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
