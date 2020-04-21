from time import sleep
import cv2
from src.env import Environment
from src.rrt import RRT

env = Environment()

# while True:
sleep(0.01)
env.step()
map = env.draw()
rrt_planner = RRT(env,20,10000,10)
start = (1,1)
goal = (295,400)
tree = rrt_planner.plan(start,goal)
map = cv2.circle(map, (295,400), 15, color=(1, 0.9, 0), thickness=-1, lineType=cv2.LINE_AA)
env.draw_path(map,tree,start,goal)
cv2.imshow("environment", map)
cv2.waitKey()