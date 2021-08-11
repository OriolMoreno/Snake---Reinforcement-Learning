from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from environment import Environment

import time


env = Environment()

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, done, info = obs.step(action)
    obs.render()

    if done:
        break

print(env.player.snake_body)
