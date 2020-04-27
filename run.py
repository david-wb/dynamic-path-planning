from time import sleep

import cv2

from src.env import Environment
from src.robot import Robot
from src.rrt import RRT

# Create environment
env = Environment()

robot = Robot(env=env, start=(1, 1), goal=(295, 400))

i = 0
while True:
    i += 1
    print('step: ', i)

    sleep(0.01)

    # Update environment
    env.step()

    # Do robot action
    robot.act(env)

    # Draw environment
    img = env.draw()

    # Draw robot
    robot.draw(img)

    # Draw goal
    img = cv2.circle(img, (295, 400), 8, color=(1, 0.9, 0), thickness=-1, lineType=cv2.LINE_AA)

    cv2.imshow("environment", img)

    if robot.reached_goal:
        break

    cv2.waitKey(1)
