import math
import random
from enum import Enum, auto
from threading import Lock
import numpy as np
import pygame

# Default configurations
GRID_WIDTH = 12
GRID_HEIGHT = 12
DEFAULT_BLOCK_SIZE = 50
TOTAL_CELLS = GRID_WIDTH * GRID_HEIGHT

COLOR = {
    'black': (15, 15, 18),
    'grid': (30, 30, 35),
    'snake_head': (65, 105, 225),   # Royal Blue
    'snake_body': (46, 204, 113),   # Green
    'food': (231, 76, 60),          # Red
    'white': (240, 240, 240)
}


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


# Relative actions: 0: Turn Left, 1: Go Straight, 2: Turn Right
operation = [
    {
        Direction.UP: Direction.LEFT,
        Direction.DOWN: Direction.RIGHT,
        Direction.LEFT: Direction.DOWN,
        Direction.RIGHT: Direction.UP,
    },
    {
        Direction.UP: Direction.UP,
        Direction.DOWN: Direction.DOWN,
        Direction.LEFT: Direction.LEFT,
        Direction.RIGHT: Direction.RIGHT,
    },
    {
        Direction.UP: Direction.RIGHT,
        Direction.DOWN: Direction.LEFT,
        Direction.LEFT: Direction.UP,
        Direction.RIGHT: Direction.DOWN,
    },
]


class GameAPI:
    """Snake game environment for RL training and human play."""

    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, block_size=DEFAULT_BLOCK_SIZE):
        self.width = width
        self.height = height
        self.total_cells = width * height
        self.block_size = block_size
        self.screen_width = width * block_size
        self.screen_height = height * block_size

        self.snake = []
        self.direction = Direction.RIGHT
        self.food = None
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.isDead = False
        self.isEat = False
        self.isRender = False

        self.screen = None
        self.clock = None
        self.font = None

        self.reset()

    def reset(self):
        """Reset game state and return initial 3x12x12 state matrix."""
        mid_x = self.width // 2
        mid_y = self.height // 2
        self.snake = [(mid_x, mid_y), (mid_x - 1, mid_y)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.isDead = False
        self.isEat = False
        self.food = self.frameGenFood()
        return self.get_state()

    def frameGenFood(self):
        """Randomly generate food on an empty grid cell."""
        snake_set = set(self.snake)
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in snake_set
        ]
        if not empty_cells:
            return None  # Win condition (snake fills board)
        return random.choice(empty_cells)

    def get_state(self):
        """Return 3-channel state representation (3, 12, 12) as float32:

        Channel 0: Snake Head (1.0)
        Channel 1: Snake Body (decaying gradient from 1.0 down along body)
        Channel 2: Food (1.0)
        """
        graph = np.zeros((3, self.height, self.width), dtype=np.float32)

        # 1. Body channel
        body_count = 0.0
        for bx, by in self.snake[1:]:
            graph[1, by, bx] = max(0.2, math.exp(-body_count))
            body_count += 0.05

        # 2. Head channel
        hx, hy = self.snake[0]
        graph[0, hy, hx] = 1.0

        # 3. Food channel
        if self.food is not None:
            fx, fy = self.food
            graph[2, fy, fx] = 1.0

        return graph

    def get_action_mask(self):
        """Return boolean mask of shape (3,) indicating safe relative actions.
        0: Turn Left, 1: Go Straight, 2: Turn Right
        """
        hx, hy = self.snake[0]
        mask = [True, True, True]

        for a in range(3):
            target_dir = operation[a][self.direction]
            if target_dir == Direction.UP:
                nh = (hx, hy - 1)
            elif target_dir == Direction.DOWN:
                nh = (hx, hy + 1)
            elif target_dir == Direction.LEFT:
                nh = (hx - 1, hy)
            elif target_dir == Direction.RIGHT:
                nh = (hx + 1, hy)
            else:
                nh = (hx, hy)

            # 1. Boundary collision check
            if nh[0] < 0 or nh[0] >= self.width or nh[1] < 0 or nh[1] >= self.height:
                mask[a] = False
                continue

            # 2. Body collision check (vacating tail is safe if not eating food this step)
            body_check = (
                self.snake[:-1]
                if (self.food is not None and nh != self.food)
                else self.snake
            )
            if nh in body_check:
                mask[a] = False
                continue

        # If all moves are fatal, allow all moves so agent takes a step and terminates
        if not any(mask):
            mask = [True, True, True]

        return np.array(mask, dtype=np.bool_)

    def stepOver(self, action):
        """Step the environment forward by 1 tick using a relative action (0: left, 1: straight, 2: right).

        Returns:
            state (np.ndarray): next state (3, 12, 12)
            reward (float): reward gained
            done (bool): whether episode is finished
        """
        self.steps += 1
        self.steps_without_food += 1

        # Determine target direction
        target_dir = operation[action][self.direction]

        # Manhattan distance to food before moving
        d_food_pre = (
            abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
            if self.food
            else 0
        )

        # Move forward
        self.isDead = self.frameForward(target_dir)

        # Reward shaping
        reward = 0.0
        if self.isDead:
            reward = -10.0
        elif self.isEat:
            reward = 10.0
            self.score += 1
            self.steps_without_food = 0
            self.food = self.frameGenFood()
            if self.food is None:  # Filled entire board
                reward += 50.0
                self.isDead = True
        else:
            # Distance reward: encourage moving toward food
            if self.food:
                d_food_post = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
                reward += 1.0 if d_food_post < d_food_pre else -1.0

            # Prevent infinite non-eating loops (starvation limit)
            max_starve_steps = 100 + len(self.snake) * 4
            if self.steps_without_food >= max_starve_steps:
                self.isDead = True
                reward = -5.0

        return self.get_state(), reward, self.isDead

    def frameForward(self, direction):
        """Advance the snake one step in the specified absolute direction."""
        self.direction = direction
        self.isEat = False
        hx, hy = self.snake[0]

        if self.direction == Direction.UP:
            new_head = (hx, hy - 1)
        elif self.direction == Direction.DOWN:
            new_head = (hx, hy + 1)
        elif self.direction == Direction.LEFT:
            new_head = (hx - 1, hy)
        elif self.direction == Direction.RIGHT:
            new_head = (hx + 1, hy)
        else:
            new_head = (hx, hy)

        # Boundary collision check
        if (
            new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
        ):
            self.isDead = True
            return self.isDead

        # Body collision check
        if new_head in self.snake:
            self.isDead = True
            return self.isDead

        # Check if food is eaten
        if self.food is not None and new_head == self.food:
            self.isEat = True

        # Move snake
        self.snake.insert(0, new_head)
        if not self.isEat:
            self.snake.pop()

        return self.isDead

    def render(self):
        """Initialize Pygame display."""
        if not self.isRender:
            self.isRender = True
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Snake AI - PPO")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 22)

    def show(self, fps=20):
        """Render the current frame to the Pygame window."""
        if not self.isRender:
            self.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self.screen.fill(COLOR['black'])

        # Draw grid lines
        for x in range(0, self.screen_width, self.block_size):
            pygame.draw.line(self.screen, COLOR['grid'], (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.block_size):
            pygame.draw.line(self.screen, COLOR['grid'], (0, y), (self.screen_width, y))

        # Draw food
        if self.food is not None:
            fx, fy = self.food
            rect = pygame.Rect(
                fx * self.block_size + 2,
                fy * self.block_size + 2,
                self.block_size - 4,
                self.block_size - 4,
            )
            pygame.draw.rect(self.screen, COLOR['food'], rect, border_radius=4)

        # Draw snake body
        for bx, by in self.snake[1:]:
            rect = pygame.Rect(
                bx * self.block_size + 2,
                by * self.block_size + 2,
                self.block_size - 4,
                self.block_size - 4,
            )
            pygame.draw.rect(self.screen, COLOR['snake_body'], rect, border_radius=3)

        # Draw snake head
        hx, hy = self.snake[0]
        head_rect = pygame.Rect(
            hx * self.block_size + 1,
            hy * self.block_size + 1,
            self.block_size - 2,
            self.block_size - 2,
        )
        pygame.draw.rect(self.screen, COLOR['snake_head'], head_rect, border_radius=5)

        # Draw Score & Steps
        info_text = f"Score: {self.score}  Steps: {self.steps}"
        surface = self.font.render(info_text, True, COLOR['white'])
        self.screen.blit(surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def close(self):
        """Close Pygame window safely."""
        if self.isRender:
            pygame.quit()
            self.isRender = False


class GameManager:
    """Manager supporting single or multi-game environments for backward compatibility."""

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GameManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'games'):
            self.games = [GameAPI()]
            self.num_game = 1

    def set_game_num(self, num_game):
        self.num_game = max(1, num_game)
        self.games = [GameAPI() for _ in range(self.num_game)]

    def get_game(self, index=0):
        return self.games[index]

    def __getitem__(self, index):
        return self.get_game(index)

    def __iter__(self):
        return iter(self.games)

    def resetAll(self):
        return [game.reset() for game in self.games]


class ParallelGameEnv:
    """Vectorized / Multi-threaded environment running N concurrent Snake games."""

    def __init__(self, num_envs=8, width=GRID_WIDTH, height=GRID_HEIGHT):
        from concurrent.futures import ThreadPoolExecutor

        self.num_envs = num_envs
        self.envs = [GameAPI(width=width, height=height) for _ in range(num_envs)]
        self.executor = ThreadPoolExecutor(max_workers=num_envs)

    def reset(self):
        """Reset all environments and return array of states and action masks.

        Returns:
            states (np.ndarray): shape (num_envs, 3, 12, 12)
            masks (np.ndarray): shape (num_envs, 3)
        """
        states = [env.reset() for env in self.envs]
        masks = [env.get_action_mask() for env in self.envs]
        return np.array(states, dtype=np.float32), np.array(masks, dtype=np.bool_)

    def _step_single(self, args):
        env_idx, action = args
        env = self.envs[env_idx]
        next_state, reward, done = env.stepOver(action)
        info = {
            "score": env.score,
            "steps": env.steps,
            "is_dead": done,
        }
        if done:
            # Auto-reset sub-environment to continue collecting seamless experience
            next_state = env.reset()
        next_mask = env.get_action_mask()
        return env_idx, next_state, reward, done, next_mask, info

    def step(self, actions):
        """Execute actions across all environments in parallel threads.

        Args:
            actions: list or 1D array of length num_envs

        Returns:
            next_states (np.ndarray): shape (num_envs, 3, 12, 12)
            rewards (np.ndarray): shape (num_envs,)
            dones (np.ndarray): shape (num_envs,)
            next_masks (np.ndarray): shape (num_envs, 3)
            infos (list[dict]): length num_envs
        """
        futures = self.executor.map(
            self._step_single, [(i, int(actions[i])) for i in range(self.num_envs)]
        )
        results = list(futures)
        results.sort(key=lambda x: x[0])

        next_states = np.array([r[1] for r in results], dtype=np.float32)
        rewards = np.array([r[2] for r in results], dtype=np.float32)
        dones = np.array([r[3] for r in results], dtype=np.bool_)
        next_masks = np.array([r[4] for r in results], dtype=np.bool_)
        infos = [r[5] for r in results]

        return next_states, rewards, dones, next_masks, infos

    def close(self):
        """Shut down thread pool and close environments."""
        self.executor.shutdown(wait=False)
        for env in self.envs:
            env.close()