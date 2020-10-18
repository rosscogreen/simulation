import warnings

from rl.core import Processor

warnings.filterwarnings('ignore')


class RLProcessor(Processor):

    def process_observation(self, observation):
        return observation.obs()
