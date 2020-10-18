import warnings

warnings.filterwarnings('ignore')

import gym
from tqdm import trange
import simulation
from pathlib import Path

import sys

sys.path.insert(0, './scripts/')
from utils import capture_intermediate_frames, record_videos

from simulation.agents import NoneAgent, HeuristicAgent, RandomAgent

params = {
    'steps_per_episode':    20,
    'policy_frequency':     1 / 4,
    'simulation_frequency': 15,
    'save_history':         True,
    'offscreen_rendering':  False,
    'demand_multiplier':    2.5
}

env = gym.make('highway-v0', **params)
# env = record_videos(env)
state, done = env.reset(), False
# capture_intermediate_frames(env)

# agent = NoneAgent(env)
agent = HeuristicAgent(env)

for step in trange(env.steps_per_episode - 1, desc="Running..."):
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    if done:
        break

env.close()

filename = Path('..') / 'metrics' / 'simulation_metrics.csv'
history_df = env.get_history(as_dataframe=True)
history_df.to_csv(filename, index=False)
