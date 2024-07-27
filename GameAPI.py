import numpy as np


class GameAPI:
    def __init__(self):
        self.Graph = np.zeros((12, 12))
        self.next_Graph = np.zeros((12, 12))
        self.award = 0

    def reset(self):               #初始化,重新开始游戏,返回值:12*12图像
        pass
    def set_operation(self, opt):  #从模型获取操作opt,并进行处理,返回值12*12图像(相当于最后调用一下get_next_graph)
        pass

    def get_now_graph(self):  #获取操作前的图像,返回值:12*12图像
        pass

    def get_next_graph(self):  #获取操作后的图像,返回值:12*12图像
        pass

    def get_award(self):  #获取奖励,不需要覆写
        pass

    def get_head(self):  #获取蛇的头坐标,返回值:(x,y)元组
        pass

    def get_food(self):  #获取食物坐标,返回值:(x,y)元组
        pass

    def get_len(self):  #获取蛇的长度,返回值:int
        pass


class GameManager:
    def __init__(self):
        self.games = []
        self.num_game = 0

    def set_game_num(self, num_game):
        self.games = [GameAPI() for i in range(num_game)]
        self.num_game = num_game

    def get_game(self, index):
        return self.games[index]

    def __getitem__(self, index):
        return self.get_game(index)

    def __iter__(self):
        return iter(self.games)



