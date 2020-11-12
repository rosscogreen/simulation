# Masters of Data Science Research Project

## A Reinforcement Learning Approach to Dynamic Lane Reversal

### Simulation

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
/videos
```
Contains videos of running simulation

### Example

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