from abc import ABC, abstractmethod
from typing import Tuple

class Agent(ABC):
    ADD_UPSTREAM = 0
    HOLD = 1
    ADD_DOWNSTREAM = 2

    env = None

    @abstractmethod
    def predict(self, obs) -> Tuple:
        raise NotImplementedError()


class NoneAgent(Agent):

    def __init__(self, env):
        self.env = env

    def predict(self, obs) -> Tuple:
        return self.HOLD, None


class RandomAgent(Agent):

    def __init__(self, env):
        self.env = env

    def predict(self, obs) -> Tuple:
        return self.env.action_space.sample(), None