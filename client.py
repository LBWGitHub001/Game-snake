import numpy as np
import pygame
import math
import random
import time
from threading import Lock
# import threading
from enum import Enum, auto

from pandas import options

screen_width = 1200
screen_height = 1200
block_size = 100
block = (block_size, block_size)
X_max = screen_width // block_size
Y_max = screen_height // block_size
total = X_max * Y_max
fps = 300
# id_prop = {0: 'null', 1: 'body', 2: 'head', 3: 'food'}
color = {'black': (0, 0, 0), 'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
         'yellow': (255, 255, 0), 'purple': (255, 0, 255), 'cyan': (0, 255, 255), 'white': (255, 255, 255)}
isDead = False


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


operation = [{Direction.UP: Direction.LEFT, Direction.DOWN: Direction.RIGHT, Direction.LEFT: Direction.DOWN,
              Direction.RIGHT: Direction.UP},
             {Direction.UP: Direction.UP, Direction.DOWN: Direction.DOWN, Direction.LEFT: Direction.LEFT,
              Direction.RIGHT: Direction.RIGHT},
             {Direction.UP: Direction.RIGHT, Direction.DOWN: Direction.LEFT, Direction.LEFT: Direction.UP,
              Direction.RIGHT: Direction.DOWN}]


class GameAPI:
    def __init__(self):
        #游戏数据
        self.direction = Direction.RIGHT
        #蛇的数据
        self.food=[]
        self.snake = [(6, 6), (5, 6)]  # save tuple for the position of snake head&body
        self.food=self.frameGenFood()

        self.award = 0
        self.isDead = False
        self.isEat = False
        self.isRender = False




    def reset(self):  # 初始化,重新开始游戏,返回值:12*12图像
        #游戏数据
        #游戏数据
        self.direction = Direction.RIGHT
        #蛇的数据
        self.food=[]
        self.snake = [(6, 6), (5, 6)]  # save tuple for the position of snake head&body
        self.food=self.frameGenFood()

        self.award = 0
        self.isDead = False
        self.isEat = False
        self.isRender = False
        return self.get_state()

    def frameGenFood(self):
        if not self.food:  # if food Empty then Add one
            food = random.randint(0, total - 1)
            x = food % X_max
            y = food // X_max
            i = 1
            m = 1
            while (x,y) not in self.snake:
                food = food + 1
                food %= total
                x = food % X_max
                y = food // X_max
            return x, y
        else:
            return self.food

    def get_state(self):
        return self.toNumpy()
    def stepOver(self,action):
        opt=operation[action][self.direction]

        award = 0
        d_food_pre = (self.snake[0][0] - self.food[0]) ** 2 + (self.snake[0][1] - self.food[1]) ** 2
        self.isDead = self.frameForward(opt) # 前进一步，所有的信息都会被刷新
        self.food = self.frameGenFood()
        d_food_post = (self.snake[0][0] - self.food[0]) ** 2 + (self.snake[0][1] - self.food[1]) ** 2
        if d_food_pre > d_food_post:
            award += 5
        else:
            award -= 5
        if self.isDead:
            award -= 100
        if self.isEat:
            award += 20
        return self.get_state(), award, self.isDead




    def toNumpy(self):
        Graph = np.zeros((3, 12, 12))
        body_count = 0
        for body in self.snake[1:]:
            self.color_block(Graph,(body[0], body[1]), (0, 255 * math.exp(-body_count), 0))
            body_count += 0.02
        self.color_block(Graph,(self.snake[0][0], self.snake[0][1]), (0, 0, 255))
        self.color_block(Graph,(self.food[0], self.food[1]), (255, 255, 0))
        return Graph

    def color_block(self,Graph, block_this, color_this):
        Graph[0][block_this[0]][block_this[1]] = color_this[0]
        Graph[1][block_this[0]][block_this[1]] = color_this[1]
        Graph[2][block_this[0]][block_this[1]] = color_this[2]
        return Graph

    def render(self):
        self.isRender = True
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 50)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('贪吃蛇')
        self.screen.fill((0, 0, 0))

    def show(self):  # self.isRender=True,显示窗口
        self.screen.fill(color['black'])
        # snack = graph[0]
        # food = graph[1][0]
        # head = snack[0]
        body_count = 0
        for body in self.snake[1:]:
            pygame.draw.rect(self.screen, (0, 255 * math.exp(-body_count), 0),
                             pygame.Rect(position(body[0], body[1]), block))
            body_count += 0.02
        pygame.draw.rect(self.screen, color['yellow'], pygame.Rect(position(self.food[0][0], self.food[0][1]), block))
        pygame.draw.rect(self.screen, color['blue'], pygame.Rect(position(self.snake[0][0], self.snake[0][1]), block))
        score_text = self.font.render(f'Score: {self.score}', True, color['white'])
        self.screen.blit(score_text, (10, 0))
        pygame.display.flip()


    def frameForward(self,direction):  # make sure operation legal, before using frameForward
        self.direction=direction
        self.isEat = False
        head = self.snake[0]
        if self.direction == Direction.UP:
            head = (head[0], head[1] - 1)
        elif self.direction == Direction.DOWN:
            head = (head[0], head[1] + 1)
        elif self.direction == Direction.LEFT:
            head = (head[0] - 1, head[1])
        elif self.direction == Direction.RIGHT:
            head = (head[0] + 1, head[1])
        if head in self.food:
            self.food.clear()
            self.isEat = True
        if head[0] > X_max - 1 or head[0] < 0 or head[1] > Y_max - 1 or head[1] < 0:
            self.isDead = True
            #print("game over reason: Exceed boundary")
            return self.isDead
        if head in self.snake:
            self.isDead = True
            #print("game over reason: Head meet body")
            return self.isDead
        self.snake.insert(0, head)
        if not self.isEat:
            self.snake.pop(-1)  # 删除尾部一节
        else:
            self.food=[]

        return self.isDead


def position(x, y):
    return x * block_size, y * block_size


class GameManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):  # singleton pattern
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GameManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance  # instance exits

    def __init__(self):
        self.games = []
        self.num_game = 0
        self.total_render = True

    def set_game_num(self, num_game):
        self.games = [GameAPI() for _ in range(num_game)]
        self.num_game = num_game

    def get_game(self, index):
        return self.games[index]

    def __getitem__(self, index):
        return self.get_game(index)

    def __iter__(self):
        return iter(self.games)

    def setRender(self, render):
        # using setRender() after set_game_num(), actually need after self.games = [GameAPI() for _ in range(num_game)]
        self.total_render = render
        for game in self.games:
            game.render()

    def resetAll(self):  # reset all games
        for game in self.games:
            game.reset()