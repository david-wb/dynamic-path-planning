from typing import Tuple, List

import cv2
import numpy as np

from src.env import Environment
from src.rrt import RRT


class Robot:
    def __init__(self, env: Environment, start: Tuple[int, int], goal: Tuple[int, int], velocity=2):
        self.env = env
        self.rrt_planner = RRT(env, 20, 10000, 10)
        self.start = start
        self.goal = goal

        self.rrt_planner.plan(start, goal)
        self.path = self.rrt_planner.get_path()

        self.current_node_i = 0
        self.current_pos = self.path[0]
        self.velocity = velocity
        self.steps = 0
        self.reached_goal = False

    def act(self, env: Environment):
        current_node = self.path[self.current_node_i]

        if current_node == self.path[-1]:
            print(f'Goal reached in {self.steps} time steps.')
            self.reached_goal = True
            return

        next_node_i = self.current_node_i + 1
        next_node = self.path[next_node_i]

        ax, ay = self.current_pos
        bx, by = next_node

        direction = np.array([bx - ax, by - ay])
        distance = np.linalg.norm(direction)
        direction /= (distance + 1e-8)

        obstacle = env.detect_moving_obstacles(x=self.current_pos[0], y=self.current_pos[1])

        if obstacle is not None:
            # TODO: re-plan the path
            print('detected moving obstacle')
        else:
            if distance <= self.velocity:
                self.current_node_i += 1
                self.current_pos = next_node
            else:
                self.current_pos += direction * self.velocity

        self.steps += 1

    def draw(self, img):

        [prev_x, prev_y] = self.path[0]
        for [x, y] in self.path[1:]:

            img = cv2.line(img,
                           (int(prev_x), int(prev_y)),
                           (int(x), int(y)),
                           color=(1, 0, 0),
                           thickness=3,
                           lineType=cv2.LINE_AA)

            img = cv2.circle(img,
                             (int(prev_x), int(prev_y)),
                             5,
                             color=(1, 0, 1),
                             thickness=-1,
                             lineType=cv2.LINE_AA)

            prev_x = x
            prev_y = y

        img = cv2.circle(img,
                         (int(prev_x), int(prev_y)),
                         5,
                         color=(1, 0, 1),
                         thickness=-1,
                         lineType=cv2.LINE_AA)

        img = cv2.circle(img,
                         (int(self.current_pos[0]), int(self.current_pos[1])),
                         5,
                         color=(0, 1, 0),
                         thickness=-1,
                         lineType=cv2.LINE_AA)
        return img





