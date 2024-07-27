
direction = {'left': 0, 'right': 1, 'up': 2, 'down': 3}


class Snake():
    def __init__(self,X_max, Y_max):
        self.body = []
        self.direction = 'left'
        self.X_max = X_max
        self.Y_max = Y_max
        self.total = X_max * Y_max
    def AddBody(self, new_block):  # 为蛇增加一节长度
        self.body.append(new_block)

    @property
    def layer(self):    #是一个元组列表
        return self.body
