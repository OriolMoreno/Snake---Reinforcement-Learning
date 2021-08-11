import gym

from stable_baselines.common.env_checker import check_env
from environment import Environment

env = Environment()
check_env(env)

