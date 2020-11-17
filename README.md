# Maze-Escape
A personal project dedicated to applying both model-free reinforcement learning (RL) techniques, specifically Monte-carlo control, Q-learning, and SARSA, to both successfully and efficiently finding the way out of square mazes of different dimensions

# Requirements
* [numpy](https://www.scipy.org/install.html#pip-install) - Computation toolkit
* [gym](https://gym.openai.com/docs/#installation) - Environment toolkit

# Organization of repository:
* gym-maze: Contains code using OpenAI's gym toolkit to simulate mazes.
* model-free: Contains code with the model-free RL algorithms
    * monte-carlo: escape.py
    * q-learning: escape.py
    * sarsa: escape.py

# Acknowledgments
- The environment used as the test-bed for those techniques was developed by MattChanTK.  
- I used the textbook _Reinforcement Learning: an Introduction_ (MIT Press, 2018)
