from time import sleep
import cv2
from src.env import Environment

env = Environment()

while True:
    sleep(1/60)
    env.step()
    map = env.draw()

    cv2.imshow("environment", map)
    cv2.waitKey(1)