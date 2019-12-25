import sys
sys.path.append('../')
import random
import gym
from gym_maze.envs.maze_env import *
import numpy as np

random.seed(1)
maze_width, maze_height = 5, 5
env = MazeEnv(maze_size=(maze_width, maze_height))

def get_state_action_pairs(states, actions):
    return [(state, action) for state in states for action in actions]

def is_terminal_state(env, state):
    env.reset(state)
    some_action = env.action_space.sample()
    observation, reward, done, info = env.step(some_action)
    return done

def initialize(env, state):
    if is_terminal_state(env, state):
        return 0
    return random.random()

# How to best initialize values?
def initialize_action_values(env, states, actions):
    state_action_pairs = get_state_action_pairs(states, actions)
    Q = {}
    for state, action in state_action_pairs:
        try:
            Q[state][action] = initialize(env, state)
        except KeyError:
            Q[state] = {}
            Q[state][action] = initialize(env, state)
    return Q

def sarsa_control(env, states, actions):


for i_episode in range(200):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        # action = env.action_space.sample()
        action = policy[tuple(observation)]
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
