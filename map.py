import random
import threading


class Map():
    def __init__(self, X_max, Y_max):
        self.X_max = X_max
        self.Y_max = Y_max
        self.food = []

    def add_food(self, food):
        self.food.append(food)

    def eat_food(self, food):
        self.food.remove(food)
    @property
    def layer(self):
        block_nums = self.X_max * self.Y_max
        return random.randint(1, block_nums)
