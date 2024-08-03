import argparse
import os
import torch
from configure import *
from PPO import PPO, ActorCritic


def main(args):
    savePath = args.save_path
    state = GMM[0].reset()
    GMM[0].render(args.render)
    ppo = ActorCritic()
    ppo.load_state_dict(torch.load(savePath))

    while True:
        action = ppo.policy.action_layer(state)
        opt = torch.argmax(action).item()
        GMM[0].set_action(opt)



def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--save-path", type=str, default='./Models/AC.pth')
    parser.add_argument("--seed", type=int, default=None)  # 随机种子
    parser.add_argument("--render", type=bool, default=True)  # 是否要显示过程
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
