import gym
from gym_maze.envs.maze_env import *
import numpy as np

# env = gym.make('maze-sample-3x3-v0')
maze_width, maze_height = 5, 5
env = MazeEnv(maze_size=(maze_width, maze_height))

def average(arr):
    return sum(arr)/len(arr)

def all_states(width, height):
    return [(x,y) for x in range(width) for y in range(height)]

def all_actions():
    return [0,1,2,3]

def get_state_action_pairs(states, actions):
    return [(state, action) for state in states for action in actions]


def initialize_policy(env, states):
    return {state: env.action_space.sample() for state in states}


def initialize_averages(states, actions):
    return {state:{action:0 for action in actions} for state in states}

def initialize_returns(states, actions):
    return {(state,action):[] for state in states for action in actions}

def update_count(experience_count, observation, action):
    if (observation, action) in experience_count:
        experience_count[(observation, action)] += 1
    else:
        experience_count[(observation, action)] = 1
    return experience_count

states = all_states(maze_width,maze_height)
actions = all_actions()
state_action = get_state_action_pairs(states, actions)
policy = initialize_policy(env, states)
Q = initialize_averages(states, actions)
returns = initialize_returns(states, actions)

num_episodes, gamma = 900, 1

for _ in range(num_episodes):
    start_state, start_action = state_action[np.random.choice(len(states) * len(actions))]
    # Generate episode from start_state, start_action
    env.reset(start_state)
    env.state = np.array(start_state)
    experience, experience_count = [], {(start_state, start_action):1}
    observation, reward, done, info = env.step(start_action)
    print("state:",start_state, "action:", start_action, "next_state:",observation)
    experience.append((start_state, start_action, reward))
    # If we're starting the policy arbitrarily, what guarantees that we'll reach the goals
    while not done:
        observation = tuple(observation)
        # action = policy[observation]
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        print("state:",observation, "action:", action, "next_state:", next_observation)
        experience.append((observation, action, reward))
        experience_count = update_count(experience_count, observation, action)
        observation = next_observation
    print(experience)

    g = 0

    for t in range(len(experience)-1, -1, -1):
        state, action, reward = experience[t]
        g = gamma*g + reward
        # unless pair S_t, A_t appears in S_0, A_0, S_1, A_1, ...., S_t-1, A_t-1
        if experience_count[(state, action)] > 1:
            experience_count[(state, action)] -= 1
        else:
            returns[(state, action)].append(g)
            temp = average(returns[(state, action)])
            Q[state][action] = temp
            action, value = max(Q[state].items(), key = lambda pair: pair[1])
            policy[state] = action


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
