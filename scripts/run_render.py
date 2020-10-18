import warnings

warnings.filterwarnings('ignore')

import gym
from tqdm import trange
import simulation
from pathlib import Path

import sys

sys.path.insert(0, './scripts/')

from simulation.agents import NoneAgent, HeuristicAgent, RandomAgent, RLAgent

params = {
    'policy_frequency':     1/4,
    'simulation_frequency': 10,
    'save_history':         True,
    'offscreen_rendering':  False,
    'max_demand':           15000
}

env = gym.make('highway-v0', **params)
state, done = env.reset(), False

#agent = NoneAgent(env)
agent = HeuristicAgent(env)
# agent = RLAgent(env)

for step in trange(env.total_steps - 1, desc="Running..."):
    action = agent.act(state)
    state, reward, done, info = env.step(action)

    if done:
        break

    env.render('human')

env.close()

filename = Path('..') / 'metrics' / 'exp1' / 'baseline1.csv'
#env.history.to_csv(filename, index=False)
