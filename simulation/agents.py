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
        diff = obs.get('upstream space mean speed', 0) - obs.get('downstream space mean speed',0)
        margin = 5
        if diff > margin:
            return 2
        elif diff < -margin:
            return 1
        else:
            return 0

    def act2(self, obs) -> int:
        diff = obs['upstream avg flow'] - obs['downstream avg flow']
        if diff > 100:
            return 1
        elif diff < -100:
            return 2
        else:
            return 0



class ReinforcementLearningAgent(Agent):

    def __init__(self, env):
        super(ReinforcementLearningAgent, self).__init__(env)

    def act(self, obs) -> int:
        return 0
