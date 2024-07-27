import math
import random
import pygame
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


class Frame:
    def __init__(self):
        self.snake = [(6, 6), (5, 6)]  # turple for the position of snake body&head
        self.food = []  # turple for the position of food
        self.graph = [[0 for i in range(X_max)] for i in range(Y_max)]
        self.direction = Direction.RIGHT

    def setDirection(self, direction):
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
            print('Illegal Operation !')

    def forward(self):
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
            global isDead
            isDead = True
            return isDead
        if head in self.snake:
            isDead = True
            return isDead
        self.snake.insert(0, head)
        if isEat == False:
            self.snake.pop(-1)  # 删除尾部一节
        else:
            global score
            score += 100
        return isDead

    def clearGraph(self):
        self.graph = [[0 for i in range(X_max)] for i in range(Y_max)]
        # self.isColor.clear()

    def update(self):
        for body in self.snake:
            self.graph[body[0]][body[1]] = 1
            # self.isColor.append((body[0],body[1]))
        self.graph[self.snake[0][0]][self.snake[0][1]] = 2
        food = self.genFood()
        self.graph[self.food[0][0]][self.food[0][1]] = 3
        # self.isColor.append((self.food[0][0],self.food[0][1]))

    def genFood(self):
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

    def getGraph(self):
        self.clearGraph()
        self.update()
        return self.snake, self.food, self.direction


def position(x, y):
    return x * block_size, y * block_size


def show(screen, graph, font):
    screen.fill(color['black'])
    body_count = 0
    snack=graph[0]
    food=graph[1][0]
    head = snack[0]
    pygame.draw.rect(screen, color['yellow'], pygame.Rect(position(food[0], food[1]), block))
    pygame.draw.rect(screen, color['blue'], pygame.Rect(position(head[0], head[1]), block))
    for body in snack[1:]:
        pygame.draw.rect(screen, (0, 255 * math.exp(-body_count), 0), pygame.Rect(position(body[0], body[1]), block))
        body_count += 0.02

    global score
    score_text = font.render(f'Score: {score}', True, color['white'])
    screen.blit(score_text, (10,0))
    pygame.display.flip()


def main():
    # 初始化 Pygame
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 50)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('贪吃蛇')
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()

    frame = Frame()
    # 设置定时器事件类型（这里使用自定义事件类型）
    TIMER_EVENT_TYPE = pygame.USEREVENT + 1
    # 设置定时器，每 1000 毫秒（1 秒）触发一次 TIMER_EVENT_TYPE 事件
    pygame.time.set_timer(TIMER_EVENT_TYPE, fps)

    global isDead
    while not isDead:
        for event in pygame.event.get():
            if event.type == TIMER_EVENT_TYPE:
                isDead = frame.forward()
                if isDead:
                    break
                show(screen, frame.getGraph(),font)
            elif event.type == pygame.QUIT:
                pygame.quit()
                isDead = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    print('up')
                    frame.setDirection(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    print('down')
                    frame.setDirection(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    print('left')
                    frame.setDirection(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    print('right')
                    frame.setDirection(Direction.RIGHT)

    print('Game Over!')


if __name__ == '__main__':
    main()
