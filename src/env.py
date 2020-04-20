import math
from typing import List

import numpy as np
import cv2
import src.utils as utils

np.random.seed(0)


class MovingObstacle:
    def __init__(self, x: int, y: int, radius=10):
        self.x = x
        self.y = y
        self.radius = radius

        vx = np.random.uniform(0, 1.0)
        vy = np.random.uniform(0, 1.0)
        self.velocity = np.array([vx, vy])

    def draw(self, map: np.array) -> np.array:
        x = int(self.x)
        y = int(self.y)
        cv2.circle(map, (x, y), self.radius, color=(0, 0, 1), thickness=-1, lineType=cv2.LINE_AA)
        return map

    def update(self, map_w: int, map_h: int):
        if self.x > map_w or self.x < 0:
            self.velocity[0] *= -1
        if self.y > map_h or self.y < 0:
            self.velocity[1] *= -1

        self.x += self.velocity[0]
        self.y += self.velocity[1]


class Environment:
    def __init__(self, width=300, height=500, n_moving_obstacles=5):
        self.static_obstacles = []
        self.moving_obstacles: List[MovingObstacle] = []
        self.width = width
        self.height = height

        # initialize static obstacles as black boxes in the form (x, y, w, h)

        n_cols = 3
        n_rows = 9
        dx = self.width // (n_cols + 1)
        dy = self.height // (n_rows + 1)
        for r in range(n_rows):
            x_shift = np.random.randint(-20, 20)

            for c in range(n_cols):
                y_shift = np.random.randint(-5, 5)

                w = dx//2 + np.random.randint(-10, 20)
                h = dy//2 + np.random.randint(-10, 20)
                x = (c + 1) * dx - w//2 + x_shift
                y = (r + 1) * dy - h//2 + y_shift

                self.static_obstacles.append((x, y, w, h))

        # initialize dynamic obstacles red circles
        for _ in range(n_moving_obstacles):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            radius = 7
            self.moving_obstacles.append(MovingObstacle(x, y, radius))

    def step(self):
        for obs in self.moving_obstacles:
            obs.update(self.width, self.height)

    def check_static_collision(self, x: float, y: float, radius: float) -> bool:
        """checks if the given circle collides with any static obstacle on the map """

        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True

        for (rx, ry, w, h) in self.static_obstacles:
            if utils.circle_touches_rect(x, y, radius, rx, ry, w, h):
                return True

        return False

    def draw(self) -> np.array:
        map = np.ones((self.height, self.width, 3))
        self._draw_static_obstacles(map)
        self._draw_moving_obstacles(map)
        return map

    def _draw_static_obstacles(self, map: np.array):
        for (x, y, w, h) in self.static_obstacles:
            map = cv2.rectangle(map, (x, y), (x + w, y + h), color=0, thickness=-1)
        return map

    def _draw_moving_obstacles(self, map: np.array):
        for obs in self.moving_obstacles:
            map = obs.draw(map)
        return map

    def _draw_triangle(self, map, x: int, y: int):
        d = 16
        center = (x, y)
        vertices = np.array([
            [x, y - d / math.sqrt(3)],
            [x + d/2, y + d / (2 * math.sqrt(3))],
            [x - d/2, y + d / (2 * math.sqrt(3))]
        ])
        cv2.fillPoly(map, np.int32([vertices]), color=(0, 0.8, 0), lineType=cv2.LINE_AA)
