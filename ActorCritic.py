from configure import *
import torch
from torch import nn
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        #actor
        self.action_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        )