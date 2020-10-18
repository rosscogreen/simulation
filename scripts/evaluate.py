import gym
from tqdm.auto import trange
from pathlib import Path
import simulation
import matplotlib.pyplot as plt
import pandas as pd
from simulation.agents import NoneAgent, RandomAgent, HeuristicAgent
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

config = {
    'simulation_frequency': 15,
    'policy_frequency':     0.5,
    'demand_amplitude':     15000,
    'demand_periods':       2,
    'total_steps':          100
}
env = gym.make('highway-v0', **config)

baseline1 = NoneAgent(env)
baseline2 = RandomAgent(env)
baseline3 = HeuristicAgent(env)
model = DQN(MlpPolicy, env, verbose=1)

SEED = 1

SEEDS = [1,5,3,8,67,43,78,41,34,67]

# %% Train Model
STEPS = 1000
model.learn(total_timesteps=STEPS, log_interval=4)
model.save("dqn_highway")

# %% Load Model
model = DQN.load("dqn_highway")

# %%
def run_episode(env, model, visualise=True, save_history=True, close_env=True):
    if visualise:
        env.offscreen = False
    env.save_history = save_history

    obs, done = env.reset(), False

    while True:
    #for i in trange(env.total_steps - 1):
        # deterministic = True
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
        if visualise:
            env.render()

    if close_env:
        env.close()

    return env.evaluate, env.history if save_history else None


def run_evaluation(env, model, model_id, experiment_id, episodes=5, visualise=False, save_history=True):
    full_evaluation = []
    full_history = None

    for episode in trange(episodes):
        env.seed(SEEDS[episode])

        evaluation_results, history = run_episode(env, model, visualise, save_history, close_env=False)

        evaluation_results['episode'] = episode
        evaluation_results['model'] = model_id
        evaluation_results['experiment'] = experiment_id
        full_evaluation.append(evaluation_results)

        if save_history:
            history['episode'] = episode
            history['model'] = model_id
            history['experiment'] = experiment_id
            full_history = history if full_history is None else pd.concat([full_history, history])

    env.close()

    return pd.DataFrame(full_evaluation), full_history if save_history else None

def run_experiment(env, experiment_id, episodes=5, visualise=False, save_history=True, basedir='results'):
    ev1, h1 = run_evaluation(env, baseline1, 'baseline_1', experiment_id, episodes, visualise, save_history)
    ev2, h2 = run_evaluation(env, baseline1, 'baseline_2', experiment_id, episodes, visualise, save_history)
    ev3, h3 = run_evaluation(env, baseline1, 'baseline_3', experiment_id, episodes, visualise, save_history)
    ev4, h4 = run_evaluation(env, baseline1, 'dqn_model', experiment_id, episodes, visualise, save_history)

    evaluation_filename = Path(basedir) / f'{experiment_id}_evaluation_results.csv'
    evaluation_results_df = pd.concat([ev1, ev2, ev3, ev4])

    if save_history:
        history_filename = Path(basedir) / f'{experiment_id}_history.csv'
        history_results_df = pd.concat([h1, h2, h3, h4])





# %%
evaluation_baseline1, history_baseline1 = run_evaluation(env, baseline1, 'baseline_1')

# %%
evaluation_baseline2, history_baseline2 = run_evaluation(
        env, baseline2, 'baseline_2', episodes=1, visualise=True, save_history=True)

# %%
evaluation_baseline3, history_baseline3 = run_evaluation(
        env, baseline3, 'baseline_3', episodes=1, visualise=True, save_history=True)

# %%
evaluation_model, history_model = run_evaluation(
        env, model, 'dqn_model', episodes=1, visualise=True, save_history=True)

# %%
action = env.action_space.sample()
print(action)

# %%

obs = model.env.observation_space.sample()
print(obs)

# %%
from simulation.envs.common import Demand

demand = Demand()

demand.generate(15000, 2, 6000)

plt.plot(demand.upstream)
plt.plot(demand.downstream)
plt.show()

# %%
env._init_demand()
plt.plot(env._demand.upstream)
plt.plot(env._demand.downstream)
plt.show()
# %%

obs = env.reset()
print(obs)

# %%

obs, reward, done, info = env.step(1)
data = env.render('rgb_array')
show(data)
# %%
