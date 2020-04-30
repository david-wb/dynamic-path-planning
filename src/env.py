from typing import List, Optional

import cv2
import numpy as np

import src.utils as utils

np.random.seed(1)


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

    def get_pos(self):
        return np.array([self.x, self.y], dtype=np.float)


class Environment:
    def __init__(self, width=300, height=500, n_moving_obstacles=10):
        self.static_obstacles = []
        self.moving_obstacles: List[MovingObstacle] = []
        self.temp_moving_obstacles = []
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

    def check_dynamic_collision(self, x: float, y: float, radius: float) -> bool:
        """checks if the given circle collides with given dynamic obstacle """

        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True

        for (rx, ry, rradius) in self.temp_moving_obstacles:
            if utils.circle_touches_circle(x, y, radius, rx, ry, rradius):
                return True

        return False

    def check_collision(self, x: float, y: float, radius: float) -> bool:
        return self.check_dynamic_collision(x, y, radius) or self.check_static_collision(x, y, radius)

    def add_dynamic_obstacle(self,x: float, y: float, radius: float):
        self.temp_moving_obstacles.append((x, y, radius))

    def detect_moving_obstacles(self, x: float, y: float, distance=20.0) -> Optional[MovingObstacle]:
        for obs in self.moving_obstacles:
            dx = x - obs.x
            dy = y - obs.y
            d = np.linalg.norm([dx, dy])

            if d <= distance:
                return obs

    def draw(self) -> np.array:
        map = np.ones((self.height, self.width, 3))
        self._draw_static_obstacles(map)
        self._draw_moving_obstacles(map)
        return map

    def draw_path(self, map: np.array, tree, start, goal):

        current_node = goal
        while current_node is not start:
            map = cv2.line(map, (int(current_node[0]), int(current_node[1])),(int(tree[current_node][0]), int(tree[current_node][1])), color=(1, 0, 0), thickness=3, lineType=cv2.LINE_AA)
            map = cv2.circle(map, (int(current_node[0]), int(current_node[1])), 5, color=(1, 0, 1), thickness=-1, lineType=cv2.LINE_AA)
            current_node = tree[current_node]

        # for node in tree:
        #     map = cv2.circle(map, (int(node[0]), int(node[1])), 10, color=(1, 0, 1), thickness=-1, lineType=cv2.LINE_AA)

        return map

    def _draw_static_obstacles(self, map: np.array):
        for (x, y, w, h) in self.static_obstacles:
            map = cv2.rectangle(map, (x, y), (x + w, y + h), color=0, thickness=-1)
        return map

    def _draw_moving_obstacles(self, map: np.array):
        for obs in self.moving_obstacles:
            map = obs.draw(map)
        return map

