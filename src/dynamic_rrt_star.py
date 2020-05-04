from typing import List, Optional

import numpy as np
import scipy.stats as stats

from .env import Environment, MovingObstacle


class Node:
    def __init__(self, value: np.ndarray, parent=None, children=None, cost=0):
        self.value = value
        self.parent = parent
        self.cost = cost
        if children is None:
            self.children = []
        else:
            self.children = children

    def distance(self, other) -> float:
        return np.linalg.norm(self.value - other.value)


class DynamicRRTStar:
    def __init__(self, environment: Environment, delta_q=20, max_nodes=1000, collision_tolerance=10):
        self.replan_max_nodes = 1000
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
        self.goal_pos = np.array(goal, dtype=np.float)
        nodes = [self.root]
        self.goal_pos = np.array(goal, dtype=np.float)

        while len(nodes) < self.max_nodes:
            free_point = self.sample_point()
            nearest_node = DynamicRRTStar.nearest_node(nodes, free_point)
            new_node = self.get_new_node(nodes, nearest_node, free_point)

            if new_node:
                nodes.append(new_node)

                dist_to_goal = np.linalg.norm(new_node.value - goal)
                if dist_to_goal < 5 and (self.goal_node is None or new_node.cost < self.goal_node.cost):
                    self.goal_node = new_node
                    print(len(nodes))

        print('RRT Star plan completed')
        return

    def get_new_node(self, nodes: List[Node], nearest_node: Node, point: np.ndarray,
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
            # check the neighbors for best parent node checking
            neighbor_list, neighbor_dist = DynamicRRTStar.nearest_nodes(nodes,new_point,20)
            best_cost = nearest_node.cost + self.delta_q
            best_parent = nearest_node
            for neighbor_i in range(len(neighbor_list)):
                neighbor_cost = neighbor_list[neighbor_i].cost + neighbor_dist[neighbor_i]
                if neighbor_cost < best_cost:
                    best_cost = neighbor_cost
                    best_parent = neighbor_list[neighbor_i]

            # added best cost and best parent into the new node
            new_node = Node(value=new_point, parent=best_parent, cost=best_cost)
            best_parent.children.append(new_node)

            # rewiring
            for neighbor_i in range(len(neighbor_list)):
                if neighbor_list[neighbor_i] is not new_node.parent:
                    neighbor_current_cost = neighbor_list[neighbor_i].cost
                    neighbor_new_node_cost = new_node.cost + neighbor_dist[neighbor_i]
                    if neighbor_new_node_cost < neighbor_current_cost:
                        new_node.children.append(neighbor_list[neighbor_i])
                        neighbor_list[neighbor_i].parent = new_node
                        neighbor_list[neighbor_i].cost = neighbor_new_node_cost
            return new_node

    @staticmethod
    def nearest_nodes(nodes: List[Node], point: np.ndarray, max_radius):
        neighbor_list = []
        neighbor_dist = []
        for node in nodes:
            dist = np.linalg.norm(point - node.value)
            if dist < max_radius:
                neighbor_list.append(node)
                neighbor_dist.append(dist)

        return neighbor_list,neighbor_dist

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
        self.goal_node.cost = 10000000000
        nodes = [current_node]

        node = self.goal_node.parent
        valid_goal_ancestors = [self.goal_node]
        while node is not None:
            valid = True
            for obs in future_obs:
                if obs.check_collision(node.value, self.collision_tolerance):
                    valid = False
                    break
            if valid:
                valid_goal_ancestors.append(node)
            node = node.parent

        while len(nodes) < self.replan_max_nodes:
            free_point = self.sample_point(future_obs)
            nearest_node = DynamicRRTStar.nearest_node(nodes, free_point)
            new_node = self.get_new_node(nodes,nearest_node, free_point, future_obs)

            if new_node:
                nodes.append(new_node)
                dist_to_goal = np.linalg.norm(new_node.value - self.goal_pos)
                if dist_to_goal < self.delta_q and new_node.cost < self.goal_node.cost:
                    print(len(nodes))
                    self.goal_node = new_node
        print('RRT Star re-plan completed')

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
        x_upper = self.environment.width
        y_upper = self.environment.height
        x_mu = self.goal_pos[0]
        y_mu = self.goal_pos[1]

        sigma = 200
        X = stats.truncnorm(-x_mu / sigma, (x_upper - x_mu) / sigma, loc=x_mu, scale=sigma)
        Y = stats.truncnorm(-y_mu / sigma, (y_upper - y_mu) / sigma, loc=y_mu, scale=sigma)

        max_samples = 10000
        n = 0
        while n < max_samples:
            n += 1
            x = X.rvs(1)[0]
            y = Y.rvs(1)[0]
            valid = True
            if extra_obs:
                for obs in extra_obs:
                    if np.linalg.norm(obs.get_pos() - (x, y)) <= self.collision_tolerance + obs.radius:
                        valid = False
                        break

            if valid and not self.environment.check_static_collision(x, y, self.collision_tolerance):
                return np.array((x, y))
