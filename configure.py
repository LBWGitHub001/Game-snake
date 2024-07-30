import torch
from GameAPI import GameManager
from memory import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory = Memory()


GMM = GameManager()
flag = False
def get_GameManager(num_game=1):
    global flag
    global GMM
    if flag is False:
        GMM.set_game_num(num_game)
        flag = True
    return GMM
