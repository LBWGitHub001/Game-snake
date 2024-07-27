import time
import pygame
from GameManager import GameManager
from map import Map
from snake import Snake


def test():
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Snake')
    screen.fill((0, 0, 0))
    rect = pygame.Rect(0, 0, 50, 50)  # x=50, y=50, width=150, height=100
    pygame.draw.rect(screen, (255, 0, 0), rect)
    pygame.draw.rect(screen, (255, 255, 0), (10,10,5,20))
    pygame.draw.rect(screen, (255, 255, 0), (30,10,5,20))
    pygame.display.flip()


if __name__ == '__main__':
    pygame.init()
    # 设置窗口大小
    screen_width = 3000
    screen_height = 4000

    # 游戏运行标志
    running = True
    # 游戏管理类
    # MM = GameManager(screen_width, screen_height)
    test()
    # 游戏循环
    while running:
        time.sleep(1)

    # 退出pygame
    pygame.quit()
