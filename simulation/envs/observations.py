from typing import Dict
from collections import defaultdict
from abc import ABC, abstractmethod


class Observation(ABC):
    env = None

    @abstractmethod
    def observe(self, dt) -> Dict:
        """Get an observation of the environment state."""
        raise NotImplementedError()


class LaneReversalObservation(Observation):

    def __init__(self, env):
        self.env = env

    def observe(self, dt) -> Dict:
        obs = {}
        obs['current_step'] = self.env.current_step
        obs['upstream_lanes'] = self.env.road.upstream_lane_count
        obs['downstream_n_lanes'] = self.env.road.downstream_lane_count
        obs['upstream_demand'] = self.env.up
        obs['downstream_demand'] = self.env.downstream_demand

        obs.update(self.env.road.detectors.report(dt))
        return obs
