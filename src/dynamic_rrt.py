from typing import List, Optional

import numpy as np

from .env import Environment, MovingObstacle


class Node:
    def __init__(self, value: np.ndarray, parent=None, children=None):
        self.value = value
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children

    def distance(self, other) -> float:
        return np.linalg.norm(self.value - other.value)


class DynamicRRT:
    def __init__(self, environment: Environment, delta_q=20, max_nodes=1000, collision_tolerance=10):
        self.delta_q = delta_q
        self.environment = environment
        self.max_nodes = max_nodes
        self.collision_tolerance = collision_tolerance
        self.root = None
        self.goal_node = None
        self.goal_pos = None
        self.nodes = []
        self.path = None

    def plan(self, start, goal):
        self.root = Node(value=np.array(start, dtype=np.float))
        nodes = [self.root]
        self.goal_pos = np.array(goal, dtype=np.float)

        while len(nodes) < self.max_nodes:
            free_point = self.sample_point()
            nearest_node = DynamicRRT.nearest_node(nodes, free_point)
            new_node = self.get_new_node(nearest_node, free_point)

            if new_node:
                nodes.append(new_node)

                dist_to_goal = np.linalg.norm(new_node.value - goal)
                if dist_to_goal < 5:
                    self.goal_node = new_node
                    return

        print('failed to find path')

    def get_new_node(self, nearest_node: Node, point: np.ndarray,
                     extra_obs: Optional[List[MovingObstacle]] = None) -> Optional[Node]:
        direction = point - nearest_node.value
        distance = np.linalg.norm(direction)
        direction /= distance

        new_point = nearest_node.value + direction * self.delta_q

        if extra_obs:
            for obs in extra_obs:
                if np.linalg.norm(obs.get_pos() - new_point) <= self.collision_tolerance + obs.radius:
                    return

        if not self.environment.check_static_collision(new_point[0], new_point[1], self.collision_tolerance):
            new_node = Node(value=new_point, parent=nearest_node)
            nearest_node.children.append(new_node)
            return new_node

    @staticmethod
    def nearest_node(nodes: List[Node], point: np.ndarray) -> Node:
        min_dist = float('inf')
        nearest = None
        for node in nodes:
            dist = np.linalg.norm(point - node.value)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        assert nearest is not None
        return nearest

    def replan(self, future_obs: List[MovingObstacle], start_node, start_pos: np.ndarray):
        current_node = Node(value=start_pos, parent=start_node)
        start_node.children = [current_node]
        nodes = [start_node, current_node]

        while len(nodes) < self.max_nodes:
            free_point = self.sample_point(future_obs)
            nearest_node = DynamicRRT.nearest_node(nodes, free_point)
            new_node = self.get_new_node(nearest_node, free_point, future_obs)

            if new_node:
                nodes.append(new_node)

                dist_to_goal = np.linalg.norm(new_node.value - self.goal_pos)
                if dist_to_goal < 5:
                    self.goal_node = new_node
                    return
        print('failed to replan')

    def reachable_nodes(self, node: Node):
        if not node.children:
            return [node]

        result = [node]
        for child in node.children:
            result += self.reachable_nodes(child)
        return result

    def get_path(self) -> List[Node]:
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node)
            node = node.parent

        path = path[::-1]
        self.path = path
        return path

    def sample_point(self, extra_obs: Optional[List[MovingObstacle]] = None) -> np.ndarray:
        while True:
            x = np.random.uniform() * self.environment.width
            y = np.random.uniform() * self.environment.height

            valid = True
            if extra_obs:
                for obs in extra_obs:
                    if np.linalg.norm(obs.get_pos() - (x, y)) <= self.collision_tolerance + obs.radius:
                        valid = False
                        break

            if valid and not self.environment.check_static_collision(x, y, self.collision_tolerance):
                return np.array((x, y))