from typing import Tuple, List

import cv2
import numpy as np

from src.dynamic_rrt import DynamicRRT, Node
from src.env import Environment, MovingObstacle
from src.rrt import RRT


class Robot:
    def __init__(self, env: Environment,
                 start: Tuple[int, int],
                 goal: Tuple[int, int],
                 velocity=2,
                 strategy='lookahead'):
        self.env = env
        self.radius = 5
        self.rrt_planner = DynamicRRT(env, 10, 10000, self.radius + 5)
        self.start = start
        self.goal = goal
        self.strategy = strategy
        self.rrt_planner.plan(start, goal)
        self.path: List[Node] = self.rrt_planner.get_path()
        self.current_node_i = 0
        self.current_pos = np.copy(self.path[0].value)
        self.velocity = velocity
        self.steps = 0
        self.reached_goal = False
        self.replan_wait = 0

    def act(self, env: Environment):
        current_node = self.path[self.current_node_i]

        if current_node == self.path[-1]:
            print(f'Goal reached in {self.steps} time steps.')
            self.reached_goal = True
            return

        next_node = self.path[self.current_node_i + 1]

        ax, ay = self.current_pos
        bx, by = next_node.value

        direction = np.array([bx - ax, by - ay])
        distance = np.linalg.norm(direction)
        direction /= (distance + 1e-8)

        obstacle = env.detect_moving_obstacles(x=self.current_pos[0], y=self.current_pos[1], distance=40)

        if obstacle is not None and self.replan_wait == 0:
            self.replan_lookahead(obstacle)

        if distance <= self.velocity:
            self.current_node_i += 1
            self.current_pos = np.copy(next_node.value)
        else:
            self.current_pos += direction * self.velocity

        if self.replan_wait > 0:
            self.replan_wait -= 1
        self.steps += 1

    def get_future_positions(self, steps=1):
        result = []

        node_i = self.current_node_i
        node = self.path[self.current_node_i]
        pos = np.copy(self.current_pos)

        for i in range(steps):
            if node == self.path[-1]:
                break

            next_node = self.path[node_i + 1]
            ax, ay = pos
            bx, by = next_node.value

            direction = np.array([bx - ax, by - ay])
            distance = np.linalg.norm(direction)
            direction /= (distance + 1e-8)
            if distance <= self.velocity:
                node_i += 1
                pos = np.copy(next_node.value)
            else:
                pos += direction * self.velocity

            result.append(pos)

        return result

    def replan_lookahead(self, obstacle: MovingObstacle):
        future_robot_poses = self.get_future_positions(5)
        future_obs: List[MovingObstacle] = []
        obs_pos = obstacle.get_pos()
        for i in range(5):
            obs_pos += obstacle.velocity
            future_obs.append(MovingObstacle(obs_pos[0], obs_pos[1], obstacle.radius))

        collision_obs = []
        for (r_p, obs) in zip(future_robot_poses, future_obs):
            distance = np.linalg.norm(r_p - obs.get_pos())
            if distance < obstacle.radius + self.radius:
                collision_obs.append(obs)

        if collision_obs:
            print('replanning...')
            self.replan_wait = 5
            self.rrt_planner.replan(collision_obs,
                                    self.path[self.current_node_i],
                                    np.copy(self.current_pos))
            self.path = self.rrt_planner.get_path()

    def draw(self, img):
        [prev_x, prev_y] = self.path[0].value
        for node in self.path[1:]:
            [x, y] = node.value

            img = cv2.line(img,
                           (int(prev_x), int(prev_y)),
                           (int(x), int(y)),
                           color=(1, 0, 0),
                           thickness=3,
                           lineType=cv2.LINE_AA)

            img = cv2.circle(img,
                             (int(prev_x), int(prev_y)),
                             2,
                             color=(1, 0, 1),
                             thickness=-1)

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
                         self.radius,
                         color=(0, 1, 0),
                         thickness=-1,
                         lineType=cv2.LINE_AA)
        return img





