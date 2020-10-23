import gym
from tqdm import trange
from pathlib import Path
import sys
sys.path.insert(0, './scripts/')

import simulation
from simulation.agents import NoneAgent, HeuristicAgent, RandomAgent

MODEL_PATH = Path('models') / 'dqn_model'


params = {
    'policy_frequency':     0.5,
    'simulation_frequency': 15,
    'save_history':         True,
    'offscreen_rendering':  False,
}

env = gym.make('highway-v0', **params)
obs, done = env.reset(), False

baseline1 = NoneAgent(env)
baseline2 = RandomAgent(env)
baseline3 = HeuristicAgent(env)
model = DQN.load(MODEL_PATH)

for step in trange(env.total_steps - 1, desc="Running..."):
    action, _ = model.predict(obs)
    state, reward, done, info = env.step(action)
    if done:
        break
    env.render()

env.close()