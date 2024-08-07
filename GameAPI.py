import numpy as np
import pygame
import math
import random
import time
from threading import Lock
# import threading
from enum import Enum, auto

screen_width = 1200
screen_height = 1200
block_size = 100
block = (block_size, block_size)
X_max = screen_width // block_size
Y_max = screen_height // block_size
total = X_max * Y_max
score = 0
fps = 300
id_prop = {0: 'null', 1: 'body', 2: 'head', 3: 'food'}
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
        self.Graph = np.zeros((3, 12, 12))
        self.pre_Graph = np.zeros((3, 12, 12))
        self.award = 0
        self.frame = None
        self.isDead = False
        self.isEat = False
        self.isRender = False
        self.food = []
        self.snake_pre = []
        self.snake_post = []
        self.snake = [(6, 6), (5, 6)]  # save tuple for the position of snake body&head
        self.food = []  # save tuple for the position of food
        self.graph = [[0 for i in range(X_max)] for i in range(Y_max)]
        self.direction = Direction.RIGHT

    def reset(self):  # 初始化,重新开始游戏,返回值:12*12图像
        self.Frame()
        self.frame = None
        return self.get_now_graph()

    def set_action(self, opt):  # 从模型获取操作opt,并进行处理,返回值12*12图像(相当于最后调用一下get_next_graph)
        opts = operation[opt]
        self.pre_Graph = self.Graph
        dir = opts[self.frame.getDirection()]
        self.frame.setDirection(dir)
        self.frame.forward()
        if self.render:
            show(self.screen, self.frame.getGraph(), self.font)
            time.sleep(1)
        return self.get_next_graph()

    def get_now_graph(self):  # 获取操作前的图像,返回值:12*12图像
        self.food = self.get_food()
        self.get_len('pre')
        return self.toNumpy(self.frame.getGraph())

    def get_next_graph(self):  # 获取操作后的图像,返回值:12*12图像
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

    def Frame(self):
        self.snake = [(6, 6), (5, 6)]  # save tuple for the position of snake body&head
        self.food = []  # save tuple for the position of food
        self.graph = [[0 for i in range(X_max)] for i in range(Y_max)]
        self.direction = Direction.RIGHT

    def frameSetDirection(self, direction):
        isLegal = True
        if direction == Direction.UP and self.direction == Direction.DOWN:
            isLegal = False
        if direction == Direction.LEFT and self.direction == Direction.RIGHT:
            isLegal = False
        if direction == Direction.DOWN and self.direction == Direction.UP:
            isLegal = False
        if direction == Direction.RIGHT and self.direction == Direction.LEFT:
            isLegal = False
        if isLegal:
            self.direction = direction
        else:
            print('Illegal Operation! ')

    def frameForward(self):
        global isDead
        isEat = False
        head = self.snake[0]
        if self.direction == Direction.UP:
            head = head[0], head[1] - 1
        elif self.direction == Direction.DOWN:
            head = head[0], head[1] + 1
        elif self.direction == Direction.LEFT:
            head = head[0] - 1, head[1]
        elif self.direction == Direction.RIGHT:
            head = head[0] + 1, head[1]
        if head in self.food:
            self.food.clear()
            isEat = True
        if head[0] > X_max - 1 or head[0] < 0 or head[1] > Y_max - 1 or head[1] < 0:
            isDead = True
            return isDead
        if head in self.snake:
            isDead = True
            return isDead
        self.snake.insert(0, head)
        if not isEat:
            self.snake.pop(-1)  # 删除尾部一节
        else:
            global score
            score += 100
        return isDead

    def frameClearGraph(self):
        self.graph = [[0 for i in range(X_max)] for i in range(Y_max)]
        # self.isColor.clear()

    def frameUpdate(self):
        for body in self.snake:
            self.graph[body[0]][body[1]] = 1
            # self.isColor.append((body[0],body[1]))
        self.graph[self.snake[0][0]][self.snake[0][1]] = 2
        food = self.frameGenFood()
        self.graph[self.food[0][0]][self.food[0][1]] = 3
        # self.isColor.append((self.food[0][0],self.food[0][1]))

    def frameGenFood(self):
        if not self.food:  # if food Empty then Add one
            food = random.randint(0, total - 1)
            x = food % X_max
            y = food // X_max
            i = 1
            m = 1
            while self.graph[x][y] != 0:
                food = food + (-1) ** m * i ** 2
                food %= total
                x = food % X_max
                y = food // X_max
                i += 1
                m += 1
            self.food.append((x, y))
            return x, y
        else:
            return None

    def frameGetGraph(self):
        self.frameClearGraph()
        self.frameUpdate()
        return self.snake, self.food, self.direction

    def frameGetDirection(self):
        return self.direction


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

    def set_game_num(self, num_game):
        self.games = [GameAPI() for _ in range(num_game)]
        self.num_game = num_game

    def get_game(self, index):
        return self.games[index]

    def __getitem__(self, index):
        return self.get_game(index)

    def __iter__(self):
        return iter(self.games)


def position(x, y):
    return x * block_size, y * block_size


def show(screen, graph, font):
    screen.fill(color['black'])
    body_count = 0
    snack = graph[0]
    food = graph[1][0]
    head = snack[0]
    pygame.draw.rect(screen, color['yellow'], pygame.Rect(position(food[0], food[1]), block))
    pygame.draw.rect(screen, color['blue'], pygame.Rect(position(head[0], head[1]), block))
    for body in snack[1:]:
        pygame.draw.rect(screen, (0, 255 * math.exp(-body_count), 0), pygame.Rect(position(body[0], body[1]), block))
        body_count += 0.02

    global score
    score_text = font.render(f'Score: {score}', True, color['white'])
    screen.blit(score_text, (10, 0))
    pygame.display.flip()


def main(game_api):
    # 初始化 Pygame
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 50)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('贪吃蛇')
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()

    # 设置定时器事件类型（这里使用自定义事件类型）
    TIMER_EVENT_TYPE = pygame.USEREVENT + 1
    # 设置定时器，每 1000 毫秒（1 秒）触发一次 TIMER_EVENT_TYPE 事件
    global fps
    pygame.time.set_timer(TIMER_EVENT_TYPE, fps)

    global isDead
    while not isDead:
        for event in pygame.event.get():
            if event.type == TIMER_EVENT_TYPE:
                isDead = game_api.frameForward()
                if isDead:
                    break
                show(screen, game_api.frameGetGraph(), font)
            elif event.type == pygame.QUIT:
                isDead = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    print('up')
                    game_api.frameSetDirection(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    print('down')
                    game_api.frameSetDirection(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    print('left')
                    game_api.frameSetDirection(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    print('right')
                    game_api.frameSetDirection(Direction.RIGHT)

    print('Game Over!')
    isExit = False
    while not isExit:  # after game over, need to press any key to exit
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                isExit = True
                break

    pygame.quit()


if __name__ == '__main__':
    fps = 300
    gameAPI = GameAPI()
    main(gameAPI)
