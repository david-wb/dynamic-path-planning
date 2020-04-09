from time import sleep

from src.env import Environment

env = Environment(width=90, height=160)

while True:
    print("test")
    sleep(0.1)
    env.step()
    env.draw()