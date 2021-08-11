# importing libraries
import numpy as np

from environment import Environment

# Make an environment object
env = Environment()

score = 0

while not env.game_over:
    env.step(0)
    score = env.render()

print(score)
