import torch
from client import GameManager, GameManager
from memory import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory = Memory()

flag = False

