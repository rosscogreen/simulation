import warnings

warnings.filterwarnings('ignore')
import gym
import sys

sys.path.insert(0, './scripts/')
from simulation.agents import RLAgent

params = {
    'policy_frequency':     0.25,
    'simulation_frequency': 15,
    'save_history':         False,
    'offscreen_rendering':  False,
    'state_as_array':       True,
    'amplitude':            1000
}

env = gym.make('highway-v0', **params)
agent = RLAgent(env)
agent.train(steps=1000, visualise=True)
