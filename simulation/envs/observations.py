from typing import Dict
from collections import defaultdict

class ObservationType(object):

    def observe(self) -> Dict:
        """Get an observation of the environment state."""
        raise NotImplementedError()

class LaneReversalObservation(ObservationType):

    def __init__(self, env):
        self.env = env

    def observe(self) -> Dict:
        obs = defaultdict(dict)
        for u, detectors in self.env.road.detectors.items():
            for x, d in detectors.items():
                obs[x]['car_count'] = d.car_count_history[-1]
                obs[x]['total_speed'] = d.car_speed_history[-1]
                obs[x]['n_lanes'] = d.lane_count_history[-1]

        return obs

class LaneReversalObservation2(ObservationType):

    def __init__(self, env):
        self.env = env

    @property
    def observe(self) -> Dict:
        return {}
        # return {
        #     'steps':               self.env.steps,
        #     'time':                self.env.time,
        #     'upstream_lanes':      len(self.env.road.upstream_lanes),
        #     'downstream_lanes':    len(self.env.road.downstream_lanes),
        #     'upstream_demand':     self.env.road.upstream_demand,
        #     'downstream_demand':   self.env.road.downstream_demand,
        #     'upstream_density':    self.env.road.upstream_density,
        #     'downstream_density':  self.env.road.downstream_density,
        #     'upstream_velocity':   self.env.road.upstream_space_mean_speed,
        #     'downstream_velocity': self.env.road.downstream_space_mean_speed,
        #     'upstream_flow':       self.env.road.upstream_flow,
        #     'downstream_flow':     self.env.road.downstream_flow,
        #     'upstream_queue':      len(self.env.road.upstream_queue_in),
        #     'downstream_queue':    len(self.env.road.downstream_queue_in),
        # }