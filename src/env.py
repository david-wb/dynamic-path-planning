import numpy as np
import cv2


class Environment:
    def __init__(self, width: int, height: int):
        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.width = width
        self.height = height

    def step(self):
        pass

    def check_collision(self, x: float, y: float) -> bool:
        raise NotImplementedError()

    def draw(self) -> np.array:
        map = np.ones((self.height, self.width))
        cv2.imshow("environment", map)
        cv2.waitKey(1)
        return map
