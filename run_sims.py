import numpy as np
from tqdm import tqdm

from src.env import Environment
from src.robot import Robot, Strategy


def run_sim(robot: Robot, env: Environment, max_steps=2000):
    for i in range(max_steps):
        env.step()  # Update environment
        robot.act(env)  # Do robot action
        if robot.metrics.reached_goal:
            break


def run_sims(n_sims: int, strategy: Strategy, n_moving_obstacles: int):
    reached_goal_count = 0
    total_distance = 0
    num_collisions = 0
    time_steps = 0
    time_steps_to_goal = []

    for i in tqdm(range(n_sims)):
        env = Environment(n_moving_obstacles=n_moving_obstacles)
        robot = Robot(env=env, start=(1, 1), goal=(295, 400), strategy=strategy)
        run_sim(robot, env)

        if robot.metrics.reached_goal:
            reached_goal_count += 1
            time_steps_to_goal.append(robot.metrics.time_steps)
        total_distance += robot.metrics.distance_traveled
        num_collisions += robot.metrics.num_collisions
        time_steps += robot.metrics.time_steps

    print('Strategy: ', strategy)
    print('Goal reached rate: ', reached_goal_count/n_sims)
    print('Average distance traveled: ', total_distance/n_sims)
    print('Average number of collisions: ', num_collisions/n_sims)
    print('Average time to goal: ', np.mean(time_steps_to_goal))


run_sims(n_sims=20, strategy=Strategy.LOOK_AHEAD, n_moving_obstacles=5)
run_sims(n_sims=20, strategy=Strategy.CONTINUE, n_moving_obstacles=5)
run_sims(n_sims=20, strategy=Strategy.WAIT, n_moving_obstacles=5)
