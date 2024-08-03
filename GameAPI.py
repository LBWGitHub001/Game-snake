import warnings

import numpy as np
import pygame
from game import Frame, Direction, screen_width, screen_height, block_size
from game import show, position
import math
import time

operation = [{Direction.UP: Direction.LEFT, Direction.DOWN: Direction.RIGHT, Direction.LEFT: Direction.DOWN,
              Direction.RIGHT: Direction.UP},
             {Direction.UP: Direction.UP, Direction.DOWN: Direction.DOWN, Direction.LEFT: Direction.LEFT,
              Direction.RIGHT: Direction.RIGHT},
             {Direction.UP: Direction.RIGHT, Direction.DOWN: Direction.LEFT, Direction.LEFT: Direction.UP,
              Direction.RIGHT: Direction.DOWN}]


class GameAPI:
    def __init__(self):
        self.Graph = np.zeros((3, 12, 12))
        self.pre_Graph = np.zeros((3, 12, 12))
        self.award = 0
        self.frame = None
        self.isDead = False
        self.isEat = False
        self.isRender = False
        self.food = []
        self.snake_pre = [(6, 6), (5, 6)]
        self.snake_post = [(6, 6), (5, 6)]

    def reset(self):  # 初始化,重新开始游戏,返回值:12*12 图像
        if self.frame is None:
            self.frame = Frame()
        else:
            self.frame.clear()  # 清空游戏前端数据
            self.clear()  # 清空缓存数据
        return self.toNumpy(self.frame.getGraph())

    def set_action(self, opt):  # 从模型获取操作opt,并进行处理,返回值12*12图像
        opts = operation[opt]
        self.pre_Graph = self.Graph.copy()
        dir = opts[self.frame.getDirection()]
        self.frame.setDirection(dir)
        self.frame.forward()
        self.get_food()
        self.snake_pre = []
        for it in self.snake_post:
            self.snake_pre.append(it)

        if self.isRender:
            show(self.screen, self.frame.getGraph(), self.font)
            time.sleep(1)

        graph = self.frame.getGraph()
        self.snake_post = graph[0]
        return self.toNumpy(graph)

    def get_now_graph(self):  # 获取操作前的图像,返回值:12*12图像
        warnings.warn("此方法已废弃，不推荐使用", DeprecationWarning)
        self.food = self.get_food()
        self.get_len('pre')
        return self.toNumpy(self.frame.getGraph())

    def get_next_graph(self):  # 获取操作后的图像,返回值:12*12图像
        warnings.warn("此方法已废弃，不推荐使用", DeprecationWarning)
        self.isDead = self.frame.forward()
        self.food = self.get_food()
        self.get_len('post')
        return self.toNumpy(self.frame.getGraph())

    def get_reward(self):  # 获取奖励,不需要覆写
        award = 0
        d_food_pre = (self.snake_pre[0][0] - self.food[0][0]) ** 2 + (self.snake_pre[0][1] - self.food[0][1]) ** 2
        d_food_post = (self.snake_post[0][0] - self.food[0][0]) ** 2 + (self.snake_post[0][1] - self.food[0][1]) ** 2
        if d_food_pre > d_food_post:
            award += 5
        else:
            award -= 5
        d_center_pre = (self.snake_pre[0][0] - 6.5) ** 2 + (self.snake_pre[0][1] - 6.5) ** 2
        d_center_post = (self.snake_post[0][0] - 6.5) ** 2 + (self.snake_post[0][1] - 6.5) ** 2
        if d_center_pre > d_center_post:
            award += 5
        else:
            award -= 5
        if self.isDead:
            award -= 100
        if self.isEat:
            award += 20
        if self.isRender:
            print("This step,the Reward is " + str(award))
        return award

    def get_head(self):  # 获取蛇的头坐标,返回值:(x,y)元组
        return self.frame.snake[0]

    def get_food(self):  # 获取食物坐标,返回值:(x,y)元组
        _, food, _ = self.frame.getGraph()
        self.food = food
        return food

    def get_len(self, str):  # 获取蛇的长度,返回值:int
        snake, _, _ = self.frame.getGraph()
        if str == 'pre':
            self.snake_pre = snake
        elif str == 'post':
            self.snake_post = snake
        return len(snake)

    def get_done(self):  # 返回当前游戏是否结束(Game over)
        return self.isDead

    def toNumpy(self, graph):
        self.Graph = np.zeros((3, 12, 12))
        snack = graph[0]
        food = graph[1][0]
        head = snack[0]
        body_count = 0
        for body in snack[1:]:
            self.color_block((body[0], body[1]), (0, 255 * math.exp(-body_count), 0))
            body_count += 0.02
        self.color_block((head[0], head[1]), (0, 0, 255))
        self.color_block((food[0], food[1]), (255, 255, 0))
        return self.Graph

    def color_block(self, block, color):
        self.Graph[0][block[0]][block[1]] = color[0]
        self.Graph[1][block[0]][block[1]] = color[1]
        self.Graph[2][block[0]][block[1]] = color[2]

    def render(self):
        self.isRender = True
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 50)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('贪吃蛇')
        self.screen.fill((0, 0, 0))

    def clear(self):
        self.Graph = np.zeros((3, 12, 12))
        self.pre_Graph = np.zeros((3, 12, 12))
        self.award = 0
        self.isDead = False
        self.isEat = False
        self.food.clear()
        self.snake_pre = [(6, 6), (5, 6)]
        self.snake_post = [(6, 6), (5, 6)]


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
