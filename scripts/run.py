import warnings

import gym
from tqdm import trange
import simulation
from pathlib import Path

import sys
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, './scripts/')

from simulation.agents import NoneAgent, HeuristicAgent, RandomAgent

params = {
    'policy_frequency':     1 / 4,
    'simulation_frequency': 15,
    'save_history':         True,
    'offscreen_rendering':  True,
    'max_demand':           15000
}

env = gym.make('highway-v0', **params)

#agent = NoneAgent(env)
agent = HeuristicAgent(env)

EPISODES = 1

results = []

for episode in range(EPISODES):

    state, done = env.reset(), False

    print(f'running episode {episode}')

    for step in trange(env.total_steps - 1, desc="Running..."):
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        if done:
            break

    episode_results = env.history
    episode_results['episode'] = episode
    results.append(episode_results)

env.close()

results_df = pd.concat(results, ignore_index=True)

filename = Path('..') / 'metrics' / 'exp1' / 'baseline4.csv'
results_df.to_csv(filename, index=False)
