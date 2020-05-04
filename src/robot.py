from typing import Tuple, List

import cv2
import numpy as np
from enum import Enum

from src.dynamic_rrt import DynamicRRT, PointSampling
from src.dynamic_rrt_star import DynamicRRTStar, Node
from src.env import Environment, MovingObstacle


class Strategy(Enum):
    CONTINUE = 'CONTINUE'
    WAIT = 'WAIT'
    LOOK_AHEAD = 'LOOK_AHEAD'


class RobotMetrics:
    def __init__(self,
                 reached_goal: bool,
                 time_steps: int,
                 distance_traveled: float,
                 num_collisions: int):
        self.reached_goal = reached_goal
        self.time_steps = time_steps
        self.distance_traveled = distance_traveled
        self.num_collisions = num_collisions
        self.replan_num_nodes = []


class Robot:
    def __init__(self,
                 env: Environment,
                 start: Tuple[int, int],
                 goal: Tuple[int, int],
                 point_sampling: PointSampling,
                 velocity=2,
                 strategy: Strategy = Strategy.LOOK_AHEAD,
                 lookahead_steps=10):
        self.env = env
        self.start = start
        self.goal = goal
        self.velocity = velocity
        self.strategy = strategy
        self.lookahead_steps = lookahead_steps
        self.radius = 5
        self.rrt_planner = DynamicRRT(environment=env,
                                      point_sampling=point_sampling,
                                      delta_q=10,
                                      max_nodes=2000,
                                      collision_tolerance=self.radius + 5)
        self.rrt_planner.plan(start, goal)
        self.path: List[Node] = self.rrt_planner.get_path()
        self.current_node_i = 0
        if self.path:
            self.current_pos = np.copy(self.path[0].value)
        else:
            self.current_pos = None
        self.replan_wait = 0

        # Metrics to track
        self.metrics = RobotMetrics(False, 0, 0, 0)
        self.done = False

    def act(self, env: Environment):
        if not self.path:
            # failed to find path
            self.done = True
            return

        current_node = self.path[self.current_node_i]
        if current_node == self.path[-1]:
            self.metrics.reached_goal = True
            self.done = True
            return

        obstacles = env.detect_moving_obstacles(x=self.current_pos[0], y=self.current_pos[1], distance=50)

        next_node = self.path[self.current_node_i + 1]
        ax, ay = self.current_pos
        bx, by = next_node.value
        direction = np.array([bx - ax, by - ay])
        distance = np.linalg.norm(direction)
        direction /= (distance + 1e-8)

        if self.strategy == Strategy.LOOK_AHEAD:
            if len(obstacles) > 0 and self.replan_wait == 0:
                self.replan_lookahead(obstacles)

            if distance <= self.velocity:
                self.current_node_i += 1
                next_pos = np.copy(next_node.value)
            else:
                next_pos = self.current_pos + direction * self.velocity
        elif self.strategy == Strategy.WAIT:
            next_pos = self.wait_strategy(obstacles, next_node, distance, direction)
        else:  # continue strategy
            next_pos = self.continue_strategy(next_node, distance, direction)

        # Update metrics
        self.metrics.distance_traveled += np.linalg.norm(next_pos - self.current_pos)
        self.metrics.time_steps += 1

        if self.env.check_dynamic_collision(self.current_pos[0], self.current_pos[1], self.radius):
            self.metrics.num_collisions += 1

        if self.replan_wait > 0:
            self.replan_wait -= 1
        self.current_pos = next_pos

    def continue_strategy(self,
                          next_node:
                          Node, distance: float,
                          direction: np.ndarray) -> np.ndarray:
        if distance <= self.velocity:
            self.current_node_i += 1
            next_pos = np.copy(next_node.value)
        else:
            next_pos = self.current_pos + direction * self.velocity
        return next_pos

    def wait_strategy(self,
                      obstacles: List[MovingObstacle],
                      next_node:
                      Node, distance: float,
                      direction: np.ndarray) -> np.ndarray:
        if len(obstacles) > 0:
            next_pos = self.current_pos
        elif distance <= self.velocity:
            self.current_node_i += 1
            next_pos = np.copy(next_node.value)
        else:
            next_pos = self.current_pos + direction * self.velocity
        return next_pos

    def get_future_positions(self, steps=1):
        result = []

        node_i = self.current_node_i
        pos = np.copy(self.current_pos)

        for i in range(steps):
            if node_i == len(self.path) - 1:
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

    def replan_lookahead(self, obstacles: List[MovingObstacle]):
        future_robot_poses = self.get_future_positions(self.lookahead_steps)
        future_obs: List[MovingObstacle] = []
        for obs in obstacles:
            future_obs += obs.get_future_positions(self.lookahead_steps)

        collision_obs = []
        for i, robot_pos in enumerate(future_robot_poses):
            for obs in future_obs:

                distance_current = np.linalg.norm(self.current_pos - obs.get_pos())
                distance_future = np.linalg.norm(robot_pos - obs.get_pos())

                if distance_current < obs.radius + self.radius + 5:
                    continue

                if distance_future < obs.radius + self.radius + 3:
                    collision_obs.append(obs)

        if collision_obs:
            print('replanning...')
            self.replan_wait = 10
            num_nodes = self.rrt_planner.replan(collision_obs,
                                                self.path[self.current_node_i],
                                                np.copy(self.current_pos))
            self.metrics.replan_num_nodes.append(num_nodes)
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
