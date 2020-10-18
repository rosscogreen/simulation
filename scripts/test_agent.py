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
    'amplitude':            10000
}
env = gym.make('highway-v0', **params)
env.reset()

#%%
agent = RLAgent(env)
agent.load_model('/Users/rossgreen/PycharmProjects/simulation/scripts/models/dqn_params.h5f')



#%%
print(env.get_state())

#%%
agent.dqn.forward(env.get_state())

#%%
agent.dqn.forward()

#agent.test(nb_episodes=1, visualize=True, load_weights=True)
