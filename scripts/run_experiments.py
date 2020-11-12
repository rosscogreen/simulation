import gym
from tqdm.auto import trange
from pathlib import Path
import simulation
import pandas as pd
from simulation.agents import NoneAgent, RandomAgent, HeuristicAgent
from stable_baselines3 import DQN

EPISODES = 10
SEEDS = [1, 5, 3, 8, 67, 23, 32, 54, 17, 78]

MODEL_PATH = Path('../models') / 'dqn_model'
RESULTS_PATH = Path('../results')
EVALUATION_RESULTS_FILENAME = RESULTS_PATH / 'evaluation_results.csv'
HISTORY_FILENAME = RESULTS_PATH / 'history.csv'
SENSITIVITY_FILENAME = RESULTS_PATH / 'sensitivity.csv'


def run_episode(env, model, visualise=True):
    if visualise:
        env.offscreen = False

    obs, done = env.reset(), False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if visualise:
            env.render()

    return env.evaluate, env.history


def run_evaluation(env, model, episodes=EPISODES, visualise=False):
    evaluation_results, histories = [], []

    for episode in trange(episodes):
        env.seed(SEEDS[episode])
        evaluation_result, history = run_episode(env, model, visualise)
        evaluation_result['episode'] = episode
        evaluation_results.append(evaluation_result)
        history['episode'] = episode
        histories.append(history)

    env.close()

    return pd.DataFrame(evaluation_results), pd.concat(histories)


def run_experiment(env, experiment_id, episodes=EPISODES, visualise=False):
    evaluation_results, histories = [], []
    models = [NoneAgent(env), RandomAgent(env), HeuristicAgent(env), DQN.load(MODEL_PATH)]
    model_names = ['baseline_1', 'baseline_2', 'baseline_3', 'dqn_model']

    env.experiment = experiment_id

    for model, model_id in zip(models, model_names):
        print(f'EXPERIMENT: {experiment_id}, MODEL: {model_id}')

        evaluation_result, history = run_evaluation(env, model, episodes, visualise)

        evaluation_result['model'] = model_id
        evaluation_result['experiment'] = experiment_id
        evaluation_results.append(evaluation_result)

        history['model'] = model_id
        history['experiment'] = experiment_id
        histories.append(history)

    return pd.concat(evaluation_results), pd.concat(histories)


def run_all_experiments(episodes=EPISODES, visualise=False):
    all_results, all_histories = [], []

    config = {
        'simulation_frequency': 15,
        'policy_frequency':     0.5,
        'demand_amplitude':     15000,
        'total_steps':          100,
    }

    env = gym.make('highway-v0', **config)

    for experiment_id in ['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4']:
        results_df, histories_df = run_experiment(env, experiment_id, episodes, visualise)
        all_results.append(results_df)
        all_histories.append(histories_df)

    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(EVALUATION_RESULTS_FILENAME)

    all_histories_df = pd.concat(all_histories)
    all_histories_df.to_csv(HISTORY_FILENAME)


def run_sensitivity():
    model = DQN.load(MODEL_PATH)
    results = []

    for policy in [0.1, 0.25, 0.5, 1, 2]:
        config = {
            'simulation_frequency': 15,
            'demand_amplitude':     15000,
            'total_steps':          100,
            'policy_frequency':     policy
        }
        env = gym.make('highway-v0', **config)
        result, history = run_episode(env, model, False)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(SENSITIVITY_FILENAME)





#%%

run_all_experiments()

#%%

run_sensitivity()

#%%