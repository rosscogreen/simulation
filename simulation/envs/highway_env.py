import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
from gym import spaces
from gym.envs.registration import register

from simulation.envs.common import BaseEnv, Demand
from simulation.road_factory import create_road
from simulation.states import State

logger = logging.getLogger(__name__)


# 30 cars per mile per lane
# 19 cars per km per lane
# 10 cars per lane

class HighwayEnv(BaseEnv):
    # Actions
    ADD_UPSTREAM = 0
    HOLD = 1
    ADD_DOWNSTREAM = 2

    GAP_THRESHOLD = 0.1

    # Penalties
    NO_REVERSE_LANE_PENALTY = -0.9
    SLOW_REVERSE_PENALTY = -0.4
    QUEUE_PENALTY = -0.2

    # Rewards
    HIGH_SPEED_REWARD = 1

    LANE_CHOICE_REWARD = 1

    # Constants
    ROAD_CAPACITY = 560
    LANE_CAPACITY = 70
    FREE_FLOW_SPEED = 100.

    DENSITY_THRESHOLD_PER_LANE = 10  # per lane

    def __init__(self, **params):
        super(HighwayEnv, self).__init__()
        self.params = params
        self.offscreen = params.get('offscreen_rendering', True)
        self.total_steps = params.get('total_steps', 100)
        self.simulation_frequency = params.get('simulation_frequency', 15)
        self.policy_frequency = params.get('policy_frequency', 1)
        self.dt = 1 / self.simulation_frequency
        self.updates_per_step = int(self.simulation_frequency / self.policy_frequency)
        self.total_simulation_steps = self.total_steps * self.updates_per_step
        self.experiment = params.get('experiment_id', "experiment_1")

        # Spaces
        self.state = State(self)
        self.action_space = spaces.Discrete(3)
        self.observation_space = self.state.observation_space

        self._demand = Demand(self)

        self.previous_density = None
        self.previous_speed = None
        self.total_reward = 0
        self.road = None

        self.save_history = params.get('save_history', True)
        self.history_log = []


    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.done = False
        self.steps = 0
        self.simulation_steps = 0
        self.total_reward = 0
        self.previous_density = None
        self.previous_speed = None
        self.history_log = []
        self.road = create_road()
        self.road.np_random = self.np_random
        self._demand.generate()
        self.frames = []
        return self.state.initial_state

    def step(self, action: int = None):
        self.steps += 1

        self._act(action)
        reward = self._density_reward(action)

        self._update_rendering()
        self._simulate()

        obs = self._obs()
        # reward = self._reward(action)
        done = self._done()
        info = self._info(action, reward)

        return obs, reward, done, info

    def _simulate(self) -> None:
        for i in range(self.updates_per_step):
            self.simulation_steps += 1
            self._inflow()
            self.road.step(dt=self.dt)
            self._render()

        self.enable_auto_render = False

    def _act(self, action: Optional[int] = None):
        if action is None or self.road.is_lane_reversal_in_progress:
            return
        if action == self.ADD_UPSTREAM:
            self.road.initiate_lane_reversal(upstream=True)
        elif action == self.ADD_DOWNSTREAM:
            self.road.initiate_lane_reversal(upstream=False)

    def _obs(self):
        return self.state.observation(3600 / self.policy_frequency)

    def _density_reward(self, action):
        nlanes_up = self.road.upstream_lane_count
        nlanes_down = self.road.downstream_lane_count

        if (nlanes_up == 7 and action == self.ADD_UPSTREAM) or (nlanes_down == 7 and action == self.ADD_DOWNSTREAM):
            return self.NO_REVERSE_LANE_PENALTY

        ncars_up = len(self.road.upstream_cars)
        ncars_down = len(self.road.downstream_cars)

        density_upstream = ncars_up / nlanes_up
        density_downstream = ncars_down / nlanes_down

        try:
            gap = (density_downstream - density_upstream) / (density_downstream + density_upstream)
        except ZeroDivisionError:
            return 0

        current_diff = np.abs(density_downstream - density_upstream)

        if gap >= self.GAP_THRESHOLD:

            # Downstream is higher density -> Want to add downstream

            if action == self.ADD_DOWNSTREAM:
                return 0.1

            best_diff = max(0.0, (ncars_down / (nlanes_down + 1)) - (ncars_up / (max(1.0, nlanes_up - 1))))

            if action == self.HOLD:
                try:
                    return (best_diff - current_diff) / max(best_diff, current_diff)
                except ZeroDivisionError:
                    return 0

            if action == self.ADD_UPSTREAM:
                worst_diff = (ncars_down / (max(1.0, nlanes_down - 1))) - (ncars_up / (nlanes_up + 1))
                try:
                    return (best_diff - worst_diff) / max(best_diff, worst_diff)
                except ZeroDivisionError:
                    return 0

        elif gap <= -self.GAP_THRESHOLD:

            # Upstream is higher density -> Want to add upstream

            if action == self.ADD_UPSTREAM:
                return 0.1

            best_diff = max(0.0, (ncars_up / (nlanes_up + 1)) - (ncars_down / (max(1, nlanes_down - 1))))

            if action == self.HOLD:
                try:
                    return (best_diff - current_diff) / max(best_diff, current_diff)
                except ZeroDivisionError:
                    return 0

            if action == self.ADD_DOWNSTREAM:
                worst_diff = (ncars_up / (max(1, nlanes_up - 1))) - (ncars_down / (nlanes_down + 1))
                try:
                    return (best_diff - worst_diff) / max(best_diff, worst_diff)
                except ZeroDivisionError:
                    return 0

        else:

            if action == self.HOLD:
                return 0.1

            if action == self.ADD_UPSTREAM:
                return (nlanes_down - 4) / 4

            if action == self.ADD_DOWNSTREAM:
                return (nlanes_up - 4) / 4

    def _reward(self, action):
        cars = self.road.cars

        speed = np.mean(np.array([car.speed for car in cars]) * 3.6) / self.FREE_FLOW_SPEED
        density = len(cars) / self.ROAD_CAPACITY

        previous_density = self.previous_density
        self.previous_density = density

        previous_speed = self.previous_speed
        self.previous_speed = speed

        if self.road.upstream_lane_count == 7 and action == self.ADD_UPSTREAM:
            return self.NO_REVERSE_LANE_PENALTY

        if self.road.downstream_lane_count == 7 and action == self.ADD_DOWNSTREAM:
            return self.NO_REVERSE_LANE_PENALTY

        if self.road.is_lane_reversal_in_progress:
            return (self.road.steps_in_current_reversal / self.simulation_frequency) * self.SLOW_REVERSE_PENALTY

        density_cost = density - previous_density if previous_density is not None else 0
        speed_reward = speed - previous_speed if previous_speed is not None else 0

        total_queue_length = self.road.calc_total_queue_length()
        queue_cost = total_queue_length * self.QUEUE_PENALTY

        reward = density_cost + speed_reward + queue_cost

        return reward

    def _done(self):
        return self.steps >= self.total_steps - 1

    def _update_rendering(self):
        if self.viewer is not None:
            self.viewer.update_metrics()

    def _inflow(self):
        if self.experiment == 'experiment_4':
            self.inflow_exp4()
            return

        upstream_source_lanes = self.road.network.get_source_lanes_for_direction(True)
        downstream_source_lanes = self.road.network.get_source_lanes_for_direction(False)

        self.np_random.shuffle(upstream_source_lanes)
        self.np_random.shuffle(downstream_source_lanes)

        n_up, n_down = self._demand.counts_for_step

        n_lanes_up = len(upstream_source_lanes)
        for i in range(int(n_up)):
            lane = upstream_source_lanes[i % n_lanes_up]
            self.road.add_car(lane)

        n_lanes_down = len(downstream_source_lanes)
        for i in range(int(n_down)):
            lane = downstream_source_lanes[i % n_lanes_down]
            self.road.add_car(lane)

    def inflow_exp4(self):
        n_up_ramp2 = self._demand.generate_count(self._demand.upstream_bottleneck[self.simulation_steps])
        n_down_ramp2 = self._demand.generate_count(self._demand.downstream_bottleneck[self.simulation_steps])
        n_up, n_down = self._demand.counts_for_step

        filter_highway = lambda lanes: [l for l in lanes if l.index[2] > 0]
        filter_onramp = lambda lanes: [l for l in lanes if l.index[2] == 0 and 'onramp1_runway_start' in l.index[0]][0]

        upstream_source_lanes = self.road.network.get_source_lanes_for_direction(True)
        downstream_source_lanes = self.road.network.get_source_lanes_for_direction(False)
        upstream_onramp = filter_onramp(upstream_source_lanes)
        downstream_onramp = filter_onramp(downstream_source_lanes)
        upstream_highway_source_lanes = filter_highway(upstream_source_lanes)
        downstream_highway_source_lanes = filter_highway(downstream_source_lanes)

        n = len(upstream_highway_source_lanes)
        for i in range(int(n_up)):
            lane = upstream_highway_source_lanes[i % n]
            self.road.add_car(lane)

        n = len(downstream_highway_source_lanes)
        for i in range(int(n_down)):
            lane = downstream_highway_source_lanes[i % n]
            self.road.add_car(lane)

        for i in range(int(n_up_ramp2)):
            self.road.add_car(upstream_onramp)

        for i in range(int(n_down_ramp2)):
            self.road.add_car(downstream_onramp)

    def _info(self, action, reward) -> Dict:
        state = self.state
        self.total_reward += reward
        info = {
            'step':               self.steps,
            'action':             action,
            'upstream_demand':    self._demand.demand_up,
            'downstream_demand':  self._demand.demand_down,
            'upstream_flow':      state.q_up,
            'downstream_flow':    state.q_down,
            'upstream_speed':     state.v_up,
            'downstream_speed':   state.v_down,
            'upstream_density':   state.k_up,
            'downstream_density': state.k_down,
        }

        if self.save_history:
            self.history_log.append(info)

        return info

    @property
    def demand_up(self):
        return self._demand.demand_up

    @property
    def demand_down(self):
        return self._demand.demand_down

    @property
    def history(self):
        return pd.DataFrame(self.history_log)

    @property
    def evaluate(self):
        return {
            'average_travel_time':                        self.road.average_travel_time,
            'average_deviation_from_free_flow_speed_kmh': self.road.average_deviation_from_free_flow_speed * 3.6,
            'total_throughput':                           self.road.outflow
        }

    def print_results(self):
        print(f'')


register(id='highway-v0', entry_point='simulation.envs:HighwayEnv')
