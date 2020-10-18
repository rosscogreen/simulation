from rl.agents.dqn import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from simulation.agents.basic import Agent


class RLAgent(Agent):

    def __init__(self, env, training_steps=1000, learning_rate=0.01, window_length=4):
        self.env = env
        self.states = env.observation_space.shape[0]
        self.actions = env.action_space.n
        self.learning_rate = learning_rate
        self.window_length = window_length
        self.training_steps = training_steps

        self.dqn = self.build_agent()

        self.model_path = 'models/dqn_params.h5f'
        self.checkpoint_path = 'checkpoints/dqn_weights_{step}.h5f'
        self.log_path = 'logs/dqn_log.json'

    def build_agent(self):
        model = self.build_model()

        memory = SequentialMemory(limit=50000, window_length=1)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)


        dqn = DQNAgent(model=model,
                       nb_actions=self.actions,
                       memory=memory,
                       nb_steps_warmup=10,
                       policy=policy)

        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        return dqn

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1, self.states)))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.actions, activation='linear'))
        return model

    def train(self, steps=50000, interval=100, visualise=False):
        callbacks = [
            ModelIntervalCheckpoint(self.checkpoint_path, interval=10),
            FileLogger(self.log_path, interval=100),
        ]

        self.dqn.fit(env=self.env,
                     nb_steps=steps,
                     visualize=visualise,
                     callbacks=callbacks,
                     log_interval=interval,
                     nb_max_episode_steps=self.env.total_steps - 1)

        self.dqn.save_weights(self.model_path, overwrite=True)

    def test(self, nb_episodes=5, visualize=True, load_weights=True):
        if load_weights:
            self.dqn.load_weights(self.model_path)
        scores = self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=visualize)
        return scores

    def act(self, obs) -> int:
        return self.dqn.forward(obs)

    def load_checkpoint(self):
        self.dqn.load_weights(self.checkpoint_path)

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.dqn.load_weights(path)
