import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from configure import *
from PPO import PPO


def main(args):
    ppo = PPO(args.lr, (args.bate1, args.bate2), args.gamma, args.epochs, args.eps, args.timestep)
    get_GameManager(args.num_game)
    # 定义变量
    running_reward = 0
    avg_length = 0
    timestep = 0
    if args.render:
        GMM[0].render()
    for eposide in range(1, args.episodes + 1):
        # 启动游戏节点
        running_reward = 0
        state = GMM[0].reset()
        # timestep = 0
        for step in range(1, args.timestep + 1):
            timestep += 1

            # 运行policy_old
            action = ppo.policy_old.act(state, memory)
            state = GMM[0].set_action(action)
            reward = GMM[0].get_reward()
            done = GMM[0].get_done()

            # 保存reward,is_terminal
            memory.rewards.append(reward)
            memory.is_terminal.append(done)

            # 更新模型
            if step == args.timestep:
                ppo.update(memory)
                memory.clear()
                timestep = 0

            # 累加奖励
            running_reward += reward

            if GMM[0].get_done():
                GMM[0].reset()

        print("Eposide: ", eposide, "  Reward: ", running_reward / args.timestep)
    ppo.save("./Models/AC.pth")

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
    parser.add_argument("--render", type=bool, default=False)  # 是否要显示过程
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
