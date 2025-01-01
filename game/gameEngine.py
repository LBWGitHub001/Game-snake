import pygame
from utils.utils import *
import torch
import random
from enum import Enum, auto

color = {'black': (0, 0, 0), 'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
         'yellow': (255, 255, 0), 'purple': (255, 0, 255), 'cyan': (0, 255, 255), 'white': (255, 255, 255)}


class GameEngine:
    def __init__(self, config):
        self.config = config
        self.gameConfig = self.config['game']
        self.xTotal = self.gameConfig['xpx']
        self.yTotal = self.gameConfig['ypx']
        self.blockSize = self.gameConfig['blockSize']
        # 游戏内容
        self.state = torch.zeros((3, self.xTotal, self.yTotal), dtype=torch.float32)
        self.food = list()
        self.snake = list()
        self.snake.append((random.randint(0, self.xTotal-1), random.randint(0, self.yTotal-1)))
        self.direction = random.randint(0, 3)
        self.genFood()
        self.award = 0

    def reset(self):
        self.state = torch.zeros((3, self.xTotal, self.yTotal), dtype=torch.float32)
        self.food = list()
        self.snake = list()
        self.snake.append((random.randint(0, self.xTotal-1), random.randint(0, self.yTotal-1)))
        self.direction = random.randint(0, 3)
        self.genFood()
        self.award = 0

    def genFood(self):
        if not self.food:  # if food Empty then Add one
            total = self.xTotal * self.yTotal
            food = random.randint(0, total - 1)
            x = food % self.xTotal
            y = food // self.yTotal
            while (x, y) in self.snake:
                food = food + 1
                food %= total
                x = food % self.xTotal
                y = food // self.yTotal
            self.food.append((x, y))
            return x, y
        else:
            return self.food

    def calDistance(self):
        self.genFood()
        return (self.snake[0][0] - self.food[0][0]) ** 2 + (self.snake[0][1] - self.food[0][1]) ** 2

    def SetAction(self, action):  # action 是一个有3个数的tensor
        _, command = torch.max(action, dim=0)
        command = command[0].item()
        command -= 1
        self.direction += command
        if self.direction < 0:
            self.direction = 3
        elif self.direction > 3:
            self.direction = 0

        # 计算现在蛇头距离食物的距离
        dis_prev = self.calDistance()

        snackLen = len(self.snake)
        head = self.snake[snackLen - 1]
        if self.direction == 0:
            self.snake.insert(0,(head[0], head[1] - 1))
        elif self.direction == 1:
            self.snake.insert(0,(head[0] + 1, head[1]))
        elif self.direction == 2:
            self.snake.insert(0,(head[0], head[1] + 1))
        elif self.direction == 3:
            self.snake.insert(0,(head[0] - 1, head[1]))
        self.snake.pop(-1)

        dis_now = 0
        if self.snake[0][0] != self.food[0][0] and self.snake[0][1] != self.food[0][1]:
            dis_now += self.calDistance()
        else:
            self.award = 200
            self.snake.append(self.food[0])
        if self.snake[0][0] > self.config['game']['xpx'] or self.snake[0][0] < 0:
            if self.snake[0][1] > self.config['game']['ypx'] or self.snake[0][1] < 0:
                self.reset()
                self.award -= 300
        self.award += dis_prev - dis_now

    def getState(self):
        self.state = torch.zeros((3, self.xTotal, self.yTotal), dtype=torch.float32)
        for i in self.food:
            self.state[0][i[0]][i[1]] = color['yellow'][0]
            self.state[1][i[0]][i[1]] = color['yellow'][1]
            self.state[2][i[0]][i[1]] = color['yellow'][2]
        for i in range(len(self.snake)):
            colorSnake='green'
            if i == 0:
                colorSnake='blue'
            body=self.snake[i]
            self.state[0][body[0]][body[1]] = color[colorSnake][0] * 0.99**i
            self.state[1][body[0]][body[1]] = color[colorSnake][1] * 0.99**i
            self.state[2][body[0]][body[1]] = color[colorSnake][2] * 0.99**i
        return self.state

    def getAward(self):
        return self.award * torch.ones(1, dtype=torch.float32)
