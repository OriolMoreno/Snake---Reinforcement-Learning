import pygame
import time
import random
from player import Player
from environment import Environment
# import tensorflow as tf
# from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import pickle


env = Environment()

trainName = 'aaav25sizePenal'

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(400))
actor.add(Activation('relu'))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(env.action_space_size))
actor.add(Activation('softsign'))
print(actor.summary())

action_input = Input(shape=(env.action_space_size,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)

x = Concatenate()([action_input, flattened_observation])
x = Dense(400)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=env.action_space_size, theta=0.6, mu=0, sigma=0.3)
agent = DDPGAgent(nb_actions=env.action_space_size, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=2000, nb_steps_warmup_actor=10000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=0.001, clipnorm=1.), metrics=['mae'])

agent.load_weights('../weights/{}_weights.h5f'.format(trainName))

agent.test(env, nb_episodes=20, visualize=True, verbose=0, nb_max_episode_steps=1000000)
