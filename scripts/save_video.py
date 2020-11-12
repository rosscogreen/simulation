import gym
from gym.wrappers import Monitor
from tqdm import trange
from pathlib import Path
import sys

sys.path.insert(0, '/')

from stable_baselines3 import DQN

import simulation
from simulation.agents import NoneAgent, HeuristicAgent, RandomAgent

MODEL_PATH = Path('../models') / 'dqn_model'
BASE_PATH = Path('..')
EVALUATION_RESULTS_FILENAME = BASE_PATH / 'evaluation_results.csv'
HISTORY_FILENAME = BASE_PATH / 'history.csv'

VIDEO_DIR_PATH = BASE_PATH / 'videos'

params = {
    'total_steps':          100,
    'policy_frequency':     1.0,
    'simulation_frequency': 15,
    'save_history':         True,
    'offscreen_rendering':  False,
}

MODEL_NAME = 'baseline1'
VIDEO_PATH = VIDEO_DIR_PATH / MODEL_NAME

env = gym.make('highway-v0', **params)
env = Monitor(env, VIDEO_PATH, force=True, video_callable=lambda episode: True)
obs, done = env.reset(), False

baseline1 = NoneAgent(env)
baseline2 = RandomAgent(env)
baseline3 = HeuristicAgent(env)
# dqn_agent = DQN.load(MODEL_PATH)

model = baseline1

for step in trange(env.total_steps - 1, desc="Running..."):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()

# Save Results
results_df = env.evaluate
print(results_df)

history_df = env.history
path = BASE_PATH / 'baseline1_history.csv'
history_df.to_csv(path, index=False)
