# A Reinforcement Learning Approach to Dynamic Lane Reversal
## UWA: Masters of Data Science Research Project 


### Implementation Code

#### Directory
```
/simulation
```
Contains all python source code for the simulation envionment

```
/scripts
```
Contains code for running simulation, training model, etc

/scripts/run_experiments is the main code for generating experiment results

```
/models
```
Contains trained DQN model

```
/results
```
Contains results from experiments


```
/example_results
```
Shows some example results for a single run of the experiment

```
/videos
```
Contains videos of running simulation

### Examples

#### Run Simulation using trained model

```python
import gym
from stable_baselines3 import DQN
import simulation

# environment settings
params = {
    'policy_frequency':     0.25,
    'simulation_frequency': 15,
    'save_history':         True,
    'offscreen_rendering':  True,
    'max_demand':           15000
}

# create environment
env = gym.make('highway-v0', **params)

# Setup Agent
model_path = "models/dqn_model"
agent = DQN.load(model_path)

# Initial State
obs, done = env.reset(), False

while not done:
    action, _states = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()

# Result Dataframes
evaluation_results_dataframe = env.evaluate
environment_metrics_history_dataframe = env.history

```

#### Train Agent
```python
import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import simulation

TRAINING_STEPS = 1000

MODEL_PATH = "models/dqn_model"
LOG_PATH = "models/logs"
CHECKPOINT_PATH = "models/checkpoints"
MONITOR_PATH = "models/monitoring"

params = {
    'simulation_frequency': 15,
    'policy_frequency':     0.5,
    'demand_amplitude':     15000,
    'total_steps':          100,
}

env = gym.make('highway-v0', **params)

checkpoint_callback = CheckpointCallback(save_freq=1000,
                                         save_path=CHECKPOINT_PATH,
                                         name_prefix=MODEL_NAME,
                                         verbose=1)
env = Monitor(env, MONITOR_PATH)

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=LOG_PATH, learning_starts=100, target_update_interval=500)

model.learn(total_timesteps=TRAINING_STEPS, callback=checkpoint_callback, log_interval=4)

# Save model
model.save(MODEL_PATH)

```