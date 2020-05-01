import math
from typing import List

import numpy as np
from sklearn.neighbors import KDTree

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
        self.path = None

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

    def replan(self, obstacle, current_pos, current_node):
        # get all nodes within the tree and search for node impacted by the obstacle
        nodes = list(self.tree.keys())
        np_nodes = np.asarray(nodes)
        kd_tree = KDTree(np_nodes)
        obstructed_index = kd_tree.query_radius([[obstacle.x, obstacle.y]], r=2*obstacle.radius)

        # search for the first remaining path node after separation
        obstructed = False
        orphan_node_index = -1
        for index in obstructed_index[0]:
            if nodes[index] in self.path:
                obstructed = True
                if nodes[index] == self.goal:
                    print("obstacle is at goal, impossible to re-plan")
                else:
                    self.tree.pop(nodes[index])
                    orphan_node_index = max(orphan_node_index, index)
        if not obstructed:
            return self.path
        orphan_node = self.path[self.path.index(nodes[orphan_node_index])]

        # add the dynamic obstacle obstacle into map
        obstacle_current_x = obstacle.x
        obstacle_current_y = obstacle.y
        for i in range(5):
            self.environment.add_dynamic_obstacle(obstacle_current_x, obstacle_current_y, obstacle.radius)
            obstacle_current_x += obstacle.velocity[0]
            obstacle_current_y += obstacle.velocity[1]

        # connect robot current location to the tree
        neighbor_node_index = kd_tree.query_radius([[current_pos[0], current_pos[1]]], r=self.delta_q)
        for index in neighbor_node_index[0]:
            self.tree[nodes[index]] = current_pos

        replanner = RRT(self.environment, self.delta_q, self.max_node, self.collision_tolerance)
        replanner.plan((current_pos[0], current_pos[1]), orphan_node)
        replanned_path = replanner.get_path()

        pre_path = self.path[:self.path.index(current_node)+1]
        post_path = self.path[self.path.index(orphan_node):]
        return pre_path + replanned_path + post_path

    def get_path(self) -> List[np.ndarray]:
        """Returns a list of numpy (x, y) positions from start to goal."""
        assert self.start is not None and self.goal is not None

        path = []
        node = self.goal
        while node is not None:
            path.append(node)
            print(node)
            node = self.tree.get(node)

        # reverse
        path = path[::-1]
        assert path[0] == self.start
        self.path = path
        return path

    def random_point(self):
        np.random.seed()
        valid_point = False
        while not valid_point:
            x = np.random.uniform() * self.environment.width
            y = np.random.uniform() * self.environment.height
            point = (x, y)
            # check if the random point is in free space
            if not self.environment.check_collision(x,y,self.collision_tolerance):
                valid_point = True
        return point

    def nearest_node(self,point):
        nodes = list(self.tree.keys())
        np_nodes = np.asarray(nodes)
        kd_tree = KDTree(np_nodes)
        nearest_distance, nearest_neighbor = kd_tree.query([point], k=1)
        return nodes[nearest_neighbor[0][0]],  nearest_distance[0][0]

    def new_node(self, random_point, nearest_node, nearest_distance):
        if nearest_distance < self.delta_q:
            return random_point
        new_node_direction = math.atan2((random_point[1] - nearest_node[1]), (random_point[0] - nearest_node[0]))
        new_node_x = nearest_node[0] + math.cos(new_node_direction) * self.delta_q
        new_node_y = nearest_node[1] + math.sin(new_node_direction) * self.delta_q
        new_node = (new_node_x, new_node_y)
        if not self.environment.check_collision(new_node_x, new_node_y, self.collision_tolerance):
            return new_node

