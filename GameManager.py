import pygame
from snake import Snake
from map import Map

color_set = {'black': (0, 0, 0), 'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
             'yellow': (255, 255, 0), 'purple': (255, 0, 255), 'cyan': (0, 255, 255), 'white': (255, 255, 255)}


class GameManager:
    def __init__(self, width, height, block_size=None):
        pygame.init()
        # 初始化地图信息
        self.width = width
        self.height = height
        if block_size is None:
            self.block_size = 20
        else:
            self.block_size = block_size
        self.X_max = self.width // self.block_size
        self.Y_max = self.height // self.block_size
        self.total = self.X_max * self.Y_max
        # 存储地图的情况
        self.graph = [[0 for i in range(self.X_max)] for j in range(self.Y_max)]

        # 创建窗口
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('贪吃蛇')

        self.snack = Snake(self.X_max, self.Y_max)
        self.map = Map(self.X_max, self.Y_max)

    def change_position(self, x, y):  # 将x,y坐标转化为对应block的中心坐标
        return x * self.block_size - self.block_size / 2, y * self.block_size - self.block_size / 2

    def layer_overlay(self):
        for block in self.snack.layer:
            self.graph[block[0]][block[1]] = 1
        index = self.map.layer // self.block_size
        block = (index % self.X_max, index // self.X_max)
        while self.graph[block[0]][block[1]] != 0:
            index += 1
            index %= self.total
            block = (index % self.X_max, index // self.X_max)
        self.map.add_food((block[0], block[1]))
        self.graph[block[0]][block[1]] = 2

    def show_Game(self):
        for x in range(1, len(self.graph) + 1):
            for y in range(1, len(self.graph[x]) + 1):
                if self.graph[x][y] == 0:
                    self.draw_rectangle(x, y, 'black')
                    self.graph[x][y] = 0
                elif self.graph[x][y] == 1:
                    self.draw_rectangle(x, y, 'blue')
                elif self.graph[x][y] == 2:
                    self.draw_rectangle(x, y, 'yellow')

    def draw_rectangle_free(self, x, y, color=(0, 0, 0)):
        x, y = self.change_position(x, y)
        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, self.block_size, self.block_size))

    def draw_rectangle(self, x, y, color='black'):
        x, y = self.change_position(x, y)
        color = color_set[color]
        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, self.block_size, self.block_size))

    def update(self):
        pygame.display.flip()
