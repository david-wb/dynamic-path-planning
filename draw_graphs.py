import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from src.env import Environment
from src.robot import Robot, Strategy


wait_df = pd.read_csv("./wait.csv")
continue_df = pd.read_csv("./continue.csv")
lookahead_df = pd.read_csv("./look_ahead.csv")

fig = plt.figure()
lines = []
for frame in [wait_df, continue_df, lookahead_df]:
    frame = frame[frame['reached_goal'] == 1]
    line, = plt.plot(frame['time_steps'], 'o')
    lines.append(line)
plt.legend(lines, ['wait', 'continue', 'Look-ahead replanning'])
plt.ylabel('Time Steps')
plt.ylabel('Simulation')
plt.title('Time Steps Taken')
plt.show()

fig2 = plt.figure()
lines = []
for frame in [wait_df, continue_df, lookahead_df]:
    frame = frame[frame['reached_goal'] == 1]
    line, = plt.plot(frame['num_collisions'], 'o')
    lines.append(line)
plt.legend(lines, ['wait', 'continue', 'Look-ahead replanning'])
plt.ylabel('Collisions')
plt.ylabel('Simulation')
plt.title('Number of collisions')
plt.show()

## bar chart
labels = ['Average Time Steps', 'Distance']
wait = [614.04, 520.65]
continu = [283.91, 491.157]
lookahead = [298.06, 538.83]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig3, ax = plt.subplots()
rects1 = ax.bar(x - width, wait, width, label='Wait', align='center')
rects2 = ax.bar(x, continu, width, label='Continue', align='center')
rects3 = ax.bar(x + width, lookahead, width, label='Look-ahead Replanning', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('Time and distance by strategy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig3.tight_layout()
plt.ylim(0, 1000)
plt.show()

## bar chart Collisions

labels = ['Collisions']
wait = [5.2]
continu = [3.98]
lookahead = [0.5]

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig3, ax = plt.subplots()
rects1 = ax.bar(x - 0.2, wait, width, label='Wait', align='center')
rects2 = ax.bar(x, continu, width, label='Continue', align='center')
rects3 = ax.bar(x + 0.2, lookahead, width, label='Look-ahead Replanning', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('Number of collisions by strategy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig3.tight_layout()
plt.ylim(0, 6)
plt.show()

## RRT vs RRT*

labels = ['Time', 'Distance']
rrt = [414.74, 651.15]
rrt_star = [298.06, 538.83]
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig3, ax = plt.subplots()
rects1 = ax.bar(x - 0.1, rrt, width, label='RRT', align='center')
rects2 = ax.bar(x + 0.1, rrt_star, width, label='RRT_STAR', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('RRT vs RRT_STAR Average Time and Distance')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig3.tight_layout()
plt.ylim(0, 700)
plt.show()

## RRT vs RRT*

labels = ['Collisions']
rrt = [0.78]
rrt_star = [0.5]
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig3, ax = plt.subplots()
rects1 = ax.bar(x - 0.1, rrt, width, label='RRT', align='center')
rects2 = ax.bar(x + 0.1, rrt_star, width, label='RRT_STAR', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('RRT vs RRT_STAR Average Collisions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig3.tight_layout()
plt.ylim(0, 1)
plt.show()

## RRT uniform vs guided point sampling

labels = ['Replan Nodes']
rrt_guided = [341.05]
rrt_uniform = [665.65]
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig3, ax = plt.subplots()
rects1 = ax.bar(x - 0.1, rrt_guided, width, label='Guided', align='center')
rects2 = ax.bar(x + 0.1, rrt_uniform, width, label='Uniform', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('RRT Uniform vs Guided Point Sampling')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig3.tight_layout()
plt.ylim(0, 800)
plt.show()