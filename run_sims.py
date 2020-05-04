import numpy as np
from tqdm import tqdm
import pandas as pd

from src.dynamic_rrt import PointSampling
from src.env import Environment
from src.robot import Robot, Strategy


def run_sim(robot: Robot, env: Environment, max_steps=2000):
    for i in range(max_steps):
        env.step()  # Update environment
        robot.act(env)  # Do robot action
        if robot.done:
            break


def run_sims(n_sims: int, strategy: Strategy, n_moving_obstacles: int, csv_filename: str):
    reached_goal = []
    distances = []
    time_steps = []
    collision_counts = []
    time_steps_to_goal = []
    avg_replan_nodes = []

    for i in tqdm(range(n_sims)):
        env = Environment(n_moving_obstacles=n_moving_obstacles)
        robot = Robot(env=env, start=(1, 1), goal=(295, 400), point_sampling=PointSampling.UNIFORM, strategy=strategy)
        run_sim(robot, env)

        if robot.metrics.reached_goal:
            reached_goal.append(1)
            time_steps_to_goal.append(robot.metrics.time_steps)
        else:
            reached_goal.append(0)

        distances.append(robot.metrics.distance_traveled)
        collision_counts.append(robot.metrics.num_collisions)
        time_steps.append(robot.metrics.time_steps)
        if robot.metrics.replan_num_nodes:
            avg_replan_nodes.append(np.mean(robot.metrics.replan_num_nodes))

    print('Strategy: ', strategy.value)
    print('Goal reached rate: ', np.mean(reached_goal))
    print('Average distance traveled: ', np.mean(distances))
    print('Average number of collisions: ', np.mean(collision_counts))
    print('Average time to goal: ', np.mean(time_steps_to_goal))
    print('Average replan nodes: ', np.mean(avg_replan_nodes))

    df = pd.DataFrame(data={
        "reached_goal": reached_goal,
        "distance_traveled": distances,
        "num_collisions": collision_counts,
        "time_steps": time_steps,
        "num_moving_obstacles": [n_moving_obstacles] * len(time_steps)
    })

    df.to_csv(csv_filename)


run_sims(n_sims=50, strategy=Strategy.LOOK_AHEAD, n_moving_obstacles=5, csv_filename="look_ahead.csv")
run_sims(n_sims=50, strategy=Strategy.CONTINUE, n_moving_obstacles=5, csv_filename="continue.csv")
run_sims(n_sims=50, strategy=Strategy.WAIT, n_moving_obstacles=5, csv_filename="wait.csv")
