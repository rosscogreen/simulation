import gym
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import simulation

TRAINING_STEPS = 10000

MODELS_DIR_PATH = Path('models')
MODEL_NAME = 'dqn_model'
MODEL_PATH = MODELS_DIR_PATH / MODEL_NAME
LOG_PATH = MODELS_DIR_PATH / 'logs/'
CHECKPOINT_PATH = MODELS_DIR_PATH / 'checkpoints/'
MONITOR_PATH = MODELS_DIR_PATH / 'monitoring/'

config = {
    'simulation_frequency': 15,
    'policy_frequency':     0.5,
    'demand_amplitude':     15000,
    'total_steps':          100,
}

env = gym.make('highway-v0', **config)

checkpoint_callback = CheckpointCallback(save_freq=1000,
                                         save_path=CHECKPOINT_PATH,
                                         name_prefix=MODEL_NAME,
                                         verbose=1)
env = Monitor(env, MONITOR_PATH)

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=LOG_PATH, learning_starts=100, target_update_interval=500)
model.learn(total_timesteps=TRAINING_STEPS, callback=checkpoint_callback, log_interval=4)
model.save(MODEL_PATH)