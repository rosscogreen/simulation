import gym
import simulation
from simulation.agents import NoneAgent, RandomAgent
from pathlib import Path
from datetime import datetime
from PIL import Image
from gym.wrappers import Monitor
from tqdm import trange

output_dir = '/Users/rossgreen/PycharmProjects/simulation/output'
video_dir = '/Users/rossgreen/PycharmProjects/simulation/video'

def record_videos(env, path=video_dir):
    return Monitor(env, path, force=True, video_callable=lambda episode: True)

def capture_intermediate_frames(env):
    env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame

def save_img(data, step):
    img = Image.fromarray(data, 'RGB')
    img.save(f'{output_dir}/step{step}.png')

METRIC_DIR = Path('..') / 'metrics'

params = {
    'policy_frequency': 0.2,
    'save_history': True,
    'steps_per_episode': 5
}

env = gym.make('highway-v0', **params)
env = record_videos(env)
obs, done = env.reset(), False
capture_intermediate_frames(env)

agent = NoneAgent(env)

for step in trange(params['steps_per_episode'], desc="Running..."):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)


#env.render()

# step = 0
# while not done:
#     action = agent.act(obs)
#     obs, reward, done, info = env.step(action)
#     #print(obs)
#
#
#     #data = env.render('rgb_array')
#     #save_img(data, step)
#
#     step += 1


env.close()

# Save Metrics Dataframe to csv file
# filename = METRIC_DIR / f'simulation_{int(datetime.now().timestamp())}_metrics.csv'
# history_df = env.get_history(as_dataframe=True)
# history_df.to_csv(filename, index=False)



