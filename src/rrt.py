from typing import List

import src.env as env
import cv2
import numpy as np
from sklearn.neighbors import KDTree
import math
from .env import Environment


class RRT:
    def __init__(self, environment: Environment, delta_q, max_node, collision_tolerance):
        self.environment = environment
        self.delta_q = delta_q
        self.max_node = max_node
        self.collision_tolerance = collision_tolerance
        self.tree = {}
        self.start = None
        self.goal = None

    def plan(self, start, goal):
        self.start = start
        self.goal = goal

        for i in range(self.max_node):
            self.tree[start] = None
            random_point = self.random_point()
            nearest_node, nearest_distance = self.nearest_node(random_point)
            new_node = self.new_node(random_point, nearest_node, nearest_distance)
            if new_node:
                self.tree[new_node] = nearest_node
                if math.hypot(new_node[0]-goal[0], new_node[1] - goal[1]) < 30:
                    self.tree[goal] = new_node
                    break
        return self.tree

    def get_path(self) -> List[np.ndarray]:
        """Returns a list of numpy (x, y) positions from start to goal."""
        assert self.start is not None and self.goal is not None

        path = []
        node = self.goal
        while node is not None:
            path.append(node)
            node = self.tree.get(node)

        # reverse
        path = path[::-1]
        assert path[0] == self.start
        return path

    def random_point(self):
        np.random.seed()
        valid_point = False
        while not valid_point:
            x = np.random.uniform() * self.environment.width
            y = np.random.uniform() * self.environment.height
            point = (x, y)
            # check if the random point is in free space
            if not self.environment.check_static_collision(x,y,self.collision_tolerance):
                valid_point = True
        return point

    def nearest_node(self,point):
        nodes = list(self.tree.keys())
        np_nodes = np.asarray(nodes)
        kd_tree = KDTree(np_nodes)
        nearest_distance, nearest_neighbor = kd_tree.query([point], k=1)
        return nodes[nearest_neighbor[0][0]],nearest_distance[0][0]

    def new_node(self, random_point, nearest_node, nearest_distance):
        if nearest_distance < self.delta_q:
            return random_point
        new_node_direction = math.atan2((random_point[1] - nearest_node[1]), (random_point[0] - nearest_node[0]))
        new_node_x = nearest_node[0] + math.cos(new_node_direction) * self.delta_q
        new_node_y = nearest_node[1] + math.sin(new_node_direction) * self.delta_q
        new_node = (new_node_x, new_node_y)
        if not self.environment.check_static_collision(new_node_x, new_node_y, self.collision_tolerance):
            return new_node

