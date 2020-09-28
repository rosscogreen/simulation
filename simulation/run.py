import gym
import simulation
from simulation.agents import NoneAgent
from pathlib import Path
from datetime import datetime

METRIC_DIR = Path('..') / 'metrics'


env = gym.make('highway-v0')

agent = NoneAgent(env)

done = False
obs = env.reset()

while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    print(obs)
    env.render(mode='human')

env.close()

# Save Metrics Dataframe to csv file
filename = METRIC_DIR / f'simulation_{int(datetime.now().timestamp())}_metrics.csv'
history_df = env.history
history_df.to_csv(filename, index=False)



