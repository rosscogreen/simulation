from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def act(self, obs) -> str:
        raise NotImplementedError()


class NoneAgent(Agent):

    def __init__(self, env):
        super(NoneAgent, self).__init__(env)

    def act(self, obs) -> int:
        return 0


class RandomAgent(Agent):

    def __init__(self, env):
        super(RandomAgent, self).__init__(env)

    def act(self, obs) -> int:
        random_action = self.env.action_space.sample()
        return random_action


class HeuristicAgent(Agent):

    def __init__(self, env):
        super(HeuristicAgent, self).__init__(env)

    def act(self, obs) -> int:
        return 0


class LinearAgent(Agent):

    def __init__(self, env):
        super(LinearAgent, self).__init__(env)

    def act(self, obs) -> int:
        return 0


class ReinforcementLearningAgent(Agent):

    def __init__(self, env):
        super(ReinforcementLearningAgent, self).__init__(env)

    def act(self, obs) -> int:
        return 0
