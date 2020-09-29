import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
from simulation.graphics import EnvViewer
from simulation.road_factory import create_road

logger = logging.getLogger(__name__)

METRIC_DIR = Path('../..') / 'metrics'


class HighwayEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    HOLD = 0
    ADD_UPSTREAM = 1
    ADD_DOWNSTREAM = 2

    def __init__(self, **params):
        self.params = params

        self.save_history: bool = params.get('save_history', False)

        self.simulation_frequency = params.get('simulation_frequency', 20)
        self.policy_frequency = params.get('policy_frequency', 1)
        self.steps_per_episode = params.get('steps_per_episode', 100)

        self.time_per_update = 1 / self.simulation_frequency
        self.updates_per_step = int(self.simulation_frequency / self.policy_frequency)
        self.time_per_step = self.updates_per_step * self.time_per_update

        self.demand_conversion = 3600 * self.simulation_frequency

        # Seeding
        self.seed = params.get('seed', 42)
        self.np_random = seeding.np_random(self.seed)[0]

        # Spaces
        n_actions = 3
        self.action_space = spaces.Discrete(n_actions)

        self.observation_space = None

        # Running
        self.done = False
        self.current_step = 0
        self.updates = 0
        self.simulation_time = 0

        # Experiment
        self.road = None
        self.upstream_demand = []
        self.downstream_demand = []
        self.upstream_demand_for_step = 0
        self.downstream_demand_for_step = 0
        self.n_upstream_for_step = 0
        self.n_downstream_for_step = 0

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        self.history_log = []

        self.reset()

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.done = False
        self.current_step = 0
        self.updates = 0
        self.simulation_time = 0
        self.road = create_road()
        self.init_demand()
        obs = self.observe(0)

        return obs

    def step(self, action: int = None):
        self.current_step += 1

        # Car inflow Based on demand and poisson probability
        self.upstream_demand_for_step = self.upstream_demand[self.current_step]
        self.downstream_demand_for_step = self.downstream_demand[self.current_step]

        self.act(action)
        self.simulate()

        obs = self.observe_cars(action)
        reward = self.get_reward(action)
        done = self.is_done()
        info = {'action': action}

        if self.save_history:
            self.history_log.append(obs)

        return obs, reward, done, info

    def inflow(self):
        n_up = np.random.poisson(self.upstream_demand_for_step / self.demand_conversion)
        self.road.spawn(n_up, True)

        n_down = np.random.poisson(self.downstream_demand_for_step / self.demand_conversion)
        self.road.spawn(n_down, False)

    def simulate(self) -> None:
        """
            Perform several steps of simulation with constant action
        """
        for i in range(self.updates_per_step):
            self.updates += 1
            self.simulation_time += self.time_per_update
            self.road.step(dt=self.time_per_update)
            self.inflow()
            self.render_update()

            # if self.terminal():
            #     break

        self.enable_auto_render = False

    def render(self, mode: str = 'human'):
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        if self.should_update_rendering:
            self.viewer.display()

        self.viewer.handle_events()

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

        self.should_update_rendering = False

    def render_update(self):
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback is not None:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def close(self):
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def init_demand(self):
        demand_periods = self.params.get('demand_periods', 2 * np.pi)
        demand_steps = np.linspace(0, demand_periods, self.steps_per_episode + 1)
        total_demand = self.params.get('total_demand', 20000)
        min_demand = self.params.get('min_demand', 2000)
        demand_range = (total_demand - 2 * min_demand) / 2
        demand_offset = total_demand / 2

        self.upstream_demand = (demand_range * np.sin(demand_steps) + demand_offset).astype('int')
        self.downstream_demand = total_demand - self.upstream_demand

    def act(self, action: Optional[int] = None):
        if action is None or action == self.HOLD:
            return
        if self.road.is_lane_reversal_in_progress:
            logger.debug("Lane reversal is already in progress, can't perform action")
            return
        if action == self.ADD_UPSTREAM:
            self.road.initiate_lane_reversal(upstream=True)
        elif action == self.ADD_DOWNSTREAM:
            self.road.initiate_lane_reversal(upstream=False)
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    def get_reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        return 0

    def terminal(self):
        return False

    def get_history(self, as_dataframe=False) -> pd.DataFrame:
        history = self.history_log
        if as_dataframe:
            history = pd.DataFrame(history)
        return history

    def is_done(self):
        return self.done or self.current_step >= self.steps_per_episode

    def observe(self, action) -> Dict:
        return {
            'current_step':       self.current_step,
            'upstream_lanes':     self.road.upstream_lane_count,
            'downstream_n_lanes': self.road.downstream_lane_count,
            'upstream_demand':    self.upstream_demand_for_step,
            'downstream_demand':  self.downstream_demand_for_step,
            **self.road.detectors.report(self.time_per_step)
        }

    def observe_cars(self, action) -> List:
        return [c.report() for c in self.road.cars]


register(id='highway-v0', entry_point='simulation.envs:HighwayEnv')
