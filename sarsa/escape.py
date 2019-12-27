import sys
sys.path.append('../')
import random
import gym
from gym_maze.envs.maze_env import *
import numpy as np

random.seed(1)
maze_width, maze_height = 3, 3
env = MazeEnv(maze_size=(maze_width, maze_height))

def all_states(width, height):
    return [(x,y) for x in range(width) for y in range(height)]

def all_actions():
    return [0,1,2,3]

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
    # return 0

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

def epsilon_greedy_action(Q, state, epsilon):
    # The action with max value has prob 1 - epsilon
    # The other ones each have prob = epsilon/(num_actions - 1)
    num_actions = len(Q[state])
    action_to_values = sorted(Q[state].items(), key = lambda pair:pair[1], reverse=True)
    greedy_action, greedy_value = action_to_values[0]
    distribution = [1 - epsilon if val is greedy_value else epsilon/(num_actions-1) for action,val in action_to_values]
    # pick random action with epsilon-determined distribution
    actions = [action for action, value in action_to_values]
    # print('state:', state, 'action_to_values:', action_to_values, 'distribution:', distribution)
    action = random.choices(actions,distribution)[0]
    # print('action', action)
    return action


# Qns: guide to picking step_size value
def using_sarsa(env, states, actions, num_episodes=500, step_size=0.5, gamma=0.999, epsilon=0.8):
    # Initialize S (for now, always start from (0,0))
    Q = initialize_action_values(env, states, actions)
    # print('initial Q: ',Q)
    # Learning
    for _ in range(num_episodes):
        current_obs = env.reset()
        current_obs = tuple(current_obs)
        current_action = epsilon_greedy_action(Q, current_obs, epsilon)
        done = False
        while not done:
            next_obs, reward, done, info = env.step(current_action)
            next_obs = tuple(next_obs)
            next_action = epsilon_greedy_action(Q, next_obs, epsilon)
            Q[current_obs][current_action] += step_size*(reward + gamma*Q[next_obs][next_action] - Q[current_obs][current_action])
            current_obs = next_obs
            current_action = next_action
        print("Training episode ", _, " complete")

    # Defining optimal policy
    optimal_policy = {state: max(Q[state].items(), key=lambda pair: pair[1])[0] for state in Q}
    # print('Q:',Q)
    # print('optimal policy:', optimal_policy)
    return optimal_policy

if __name__ == '__main__':
    states, actions = all_states(maze_width, maze_height), all_actions()
    policy = using_sarsa(env, states, actions)
    test_trials = 200
    for i_episode in range(test_trials):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = policy[tuple(observation)]
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
