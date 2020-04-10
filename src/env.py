import math

import numpy as np
import cv2


class Environment:
    def __init__(self, width=200, height=400, n_dynamic_obstacles=5):
        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.width = width
        self.height = height

        # initialize obstacles as boxes in the form (x, y, w, h)

        n_cols = 3
        n_rows = 10
        dx = self.width // (n_cols + 1)
        dy = self.height // (n_rows + 1)
        for r in range(10):
            for c in range(3):
                w = dx//2
                h = dy//2
                x = (c + 1) * dx - w//2
                y = (r + 1) * dy - h//2

                self.static_obstacles.append((x, y, w, h))

        for _ in range(n_dynamic_obstacles):
            self.dynamic_obstacles.append((np.random.randint(0, self.width), np.random.randint(0, self.height)))

    def step(self):
        pass

    def check_collision(self, x: float, y: float) -> bool:
        raise NotImplementedError()

    def draw(self) -> np.array:
        map = np.ones((self.height, self.width, 3))
        self._draw_static_obstacles(map)
        self._draw_dynamic_obstacles(map)
        return map

    def _draw_static_obstacles(self, map: np.array):
        for (x, y, w, h) in self.static_obstacles:
            map = cv2.rectangle(map, (x, y), (x + w, y + h), color=0, thickness=-1)
        return map

    def _draw_dynamic_obstacles(self, map: np.array):
        for (x, y) in self.dynamic_obstacles:
            self._draw_triangle(map, x, y)
        return map

    def _draw_triangle(self, map, x: int, y: int):
        d = 15
        center = (x, y)
        vertices = np.array([
            [x, y - d / math.sqrt(3)],
            [x + d/2, y + d / (2 * math.sqrt(3))],
            [x - d/2, y + d / (2 * math.sqrt(3))]
        ])
        cv2.polylines(map, np.int32([vertices]), isClosed=True, color=0, thickness=1)
        cv2.polylines
        cv2.fillPoly(map, np.int32([vertices]), color=(0, 255, 0), lineType=cv2.LINE_AA)