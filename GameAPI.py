import numpy as np
class GameAPI:
    def __init__(self):
        self.Graph = np.zeros((12, 12))
        self.next_Graph = np.zeros((12, 12))
        self.award = 0

    def set_operation(self, opt): #从模型获取操作opt,并进行处理
        pass

    def get_now_graph(self):  #获取操作前的图像
        pass

    def get_next_graph(self):  #获取操作后的图像
        pass

    def get_award(self):  #获取奖励,不需要覆写
        pass

    def get_head(self):  #获取蛇的头坐标
        pass

    def get_food(self):  #获取食物坐标
        pass

    def get_len(self):  #获取蛇的长度
        pass
