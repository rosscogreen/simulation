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
from simulation.car import Car
from pathlib import Path

logger = logging.getLogger(__name__)

METRIC_DIR = Path('../..') / 'metrics'

base = Path('../..')
# FLOW_FILE_PATH = base / 'resources' / 'flows.csv'
FLOW_FILE_PATH = '/Users/rossgreen/PycharmProjects/simulation/resources/flows.csv'


class HighwayEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    HOLD = 0
    ADD_UPSTREAM = 1
    ADD_DOWNSTREAM = 2

    def __init__(self, **params):
        self.params = params

        self.demand_per_step = []
        self.init_demand()

        self.offscreen = params.get('offscreen_rendering', False)
        self.save_history: bool = params.get('save_history', False)

        #self.steps_per_episode = params.get('steps_per_episode', 84)


        self.simulation_frequency = params.get('simulation_frequency', 15)
        self.policy_frequency = params.get('policy_frequency', 1)

        self.steps_per_episode = len(self.demand_per_step)  - 1

        self.updates_per_step = int(self.simulation_frequency / self.policy_frequency)

        self.dt = 1 / self.simulation_frequency

        self.demand_conversion = 900 * self.simulation_frequency

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
        self.real_time = 0

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
        self.real_time = 0

        self.road = create_road()

        self.init_demand()
        # self.create_vehicles()

        obs = self.observe(0)

        return obs

    def create_vehicles(self):
        initial_demand_up, initial_demand_down = self.demand_per_step[0]
        print(initial_demand_up, initial_demand_down)
        self._create_vehicles(True, initial_demand_up)
        self._create_vehicles(False, initial_demand_down)

    def _create_vehicles(self, upstream, demand, sms=95, scale=10):
        density = demand / sms
        lane_density = int(density / 4)

        for i in ([1, 2, 3, 4]):
            if upstream:
                li = ('west', 'east', i) if i > 1 else ('west', 'upstream_onramp0_merge_start', i)
            else:
                li = ('east', 'west', i) if i > 1 else ('east', 'downstream_onramp0_merge_start', i)
            lane = self.road.network.get_lane(li)
            adds = np.random.normal(lane_density, scale, int(lane_density))
            s_list = np.cumsum(adds)
            for s in s_list:
                Car.make_on_lane(self.road, lane, longitudinal=s)

    def step(self, action: int = None):
        self.current_step += 1

        self.act(action)
        self.simulate()

        obs = self.observe(action)
        reward = self.get_reward(action)
        done = self.is_done()
        info = {'action': action}

        if self.save_history:
            self.history_log.append(obs)

        return obs, reward, done, info

    def simulate(self) -> None:
        """
            Perform several steps of simulation with constant action
        """
        self.upstream_demand_for_step, self.downstream_demand_for_step = self.demand_per_step[self.current_step]
        n_up = np.random.poisson(self.upstream_demand_for_step / self.demand_conversion, self.updates_per_step)
        n_down = np.random.poisson(self.downstream_demand_for_step / self.demand_conversion, self.updates_per_step)

        for i in range(self.updates_per_step):
            self.updates += 1
            self.simulation_time += self.dt

            self.road.step(dt=self.dt)

            self.road.spawn(n_up[i], True)
            self.road.spawn(n_down[i], False)

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

        if not self.offscreen:
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
        df = pd.read_csv(FLOW_FILE_PATH, parse_dates=['Quarter Hour'], index_col='Quarter Hour')
        self.demand_per_step = df.values

    def init_demand2(self):
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
        return self.is_done()

    def get_history(self, as_dataframe=False) -> pd.DataFrame:
        history = self.history_log
        if as_dataframe:
            history = pd.DataFrame(history)
        return history

    def is_done(self):
        return self.done or self.current_step >= self.steps_per_episode

    def report(self):
        return {
            'step':              self.current_step,
            'upstream demand':   self.upstream_demand_for_step,
            'downstream demand': self.downstream_demand_for_step,
        }

    def observe(self, action):
        return {
            **self.report(),
            **self.road.report(),
            **self.road.detectors.report(900 / self.policy_frequency)
        }

    def observe_cars(self, action) -> List:
        return [c.report() for c in self.road.cars]


register(id='highway-v0', entry_point='simulation.envs:HighwayEnv')
