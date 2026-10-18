import time
import pygame
from client import GameAPI, Direction, COLOR


def main():
    """Interactive human playable Snake game using arrow keys."""
    env = GameAPI()
    env.render()

    running = True
    game_over = False
    fps = 10

    # Opposite directions to prevent 180-degree instant suicide
    opposite = {
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP,
        Direction.LEFT: Direction.RIGHT,
        Direction.RIGHT: Direction.LEFT,
    }

    current_dir = env.direction

    print("==================================================")
    print(" 贪吃蛇小游戏 (Snake Game - Human Play)")
    print(" - 方向键 [↑ ↓ ← →]: 控制移动")
    print(" - [R] 或 [空格键]: 死亡后重新开始")
    print(" - [ESC] 或 关闭窗口: 退出游戏")
    print("==================================================")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif game_over:
                    if event.key in (pygame.K_SPACE, pygame.K_r):
                        env.reset()
                        current_dir = env.direction
                        game_over = False
                else:
                    new_dir = None
                    if event.key == pygame.K_UP:
                        new_dir = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        new_dir = Direction.DOWN
                    elif event.key == pygame.K_LEFT:
                        new_dir = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        new_dir = Direction.RIGHT

                    if new_dir is not None and new_dir != opposite.get(current_dir):
                        current_dir = new_dir

        if not game_over and running:
            dead = env.frameForward(current_dir)
            if dead:
                game_over = True
            else:
                if env.isEat:
                    env.score += 1
                    env.food = env.frameGenFood()

            env.show(fps=fps)

            if game_over:
                # Display game over banner
                overlay = env.font.render(
                    "GAME OVER! Press [R] or [SPACE] to Restart", True, (255, 100, 100)
                )
                text_rect = overlay.get_rect(
                    center=(env.screen_width // 2, env.screen_height // 2)
                )
                env.screen.blit(overlay, text_rect)
                pygame.display.flip()

    env.close()


if __name__ == "__main__":
    main()
