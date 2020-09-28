from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding

from simulation.envs.metrics import EnvMetrics
from simulation.envs.observations import LaneReversalObservation
from simulation.experiment import Experiment
from simulation.graphics.env import EnvViewer
from simulation.road import Road
from simulation.road_factory.roads import create_road
from simulation.road_objects import Obstacle

METRIC_DIR = Path('../..') / 'metrics'


class HighwayEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    ACTIONS = {
        0: 'HOLD',
        1: 'INCREASE_UPSTREAM',
        2: 'INCREASE_DOWNSTREAM'
    }

    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    def __init__(self, experiment=None):

        self.experiment = experiment or Experiment()
        self.metrics = EnvMetrics()

        self.dt = 1 / self.experiment.SIMULATION_FREQUENCY
        self.n_intermediate_steps = int(self.experiment.SIMULATION_FREQUENCY // self.experiment.POLICY_FREQUENCY)

        # Seeding
        self.np_random = None
        self.seed(42)

        self.road = None

        # Spaces
        self.observation = LaneReversalObservation(self)
        self.action_space = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.steps = 0  # actions performed
        self.time = 0  # simulation time

        # Experiment
        self.demand_up = []
        self.demand_down = []
        self._init_demand()

        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = True


        self.controller = None

        self._history = []

    def _init_demand(self):
        demand_steps = np.linspace(0, self.experiment.NUM_PERIODS, self.experiment.DURATION + 1)
        demand_range = (self.experiment.TOTAL_DEMAND - (2 * self.experiment.MIN_DEMAND)) / 2
        demand_offset = self.experiment.TOTAL_DEMAND / 2

        self.demand_up = (demand_range * np.sin(demand_steps) + demand_offset).astype('int')
        self.demand_down = self.experiment.TOTAL_DEMAND - self.demand_up

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.steps = 0
        self.time = 0
        self.done = False
        self.define_spaces()
        self._create_road()
        self._create_vehicles()
        self._create_detectors()
        return self.observation.observe()

    def _create_road(self):
        self.road: "Road" = create_road()

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        return

    def _create_detectors(self):
        agg_period = self.experiment.TIME_WARP//self.experiment.POLICY_FREQUENCY
        for x in [10, 490]:
            self.road.add_detector('upstream', x, agg_period)
            self.road.add_detector('downstream', x, agg_period)

    def define_spaces(self):
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # if "observation" not in self.config:
        #    raise ValueError("The observation configuration must be defined")
        # self.observation = observation_factory(self, self.config["observation"])
        # self.observation_space = self.observation.space()

    def step(self, action=None) -> Tuple[Dict, float, bool, dict]:
        """
            Perform an action and step the environment dynamics.

        :param int action: the action performed by the ego-vehicles
        :return: a tuple (observation, reward, terminal, info)
        """
        self.steps += 1
        self._simulate()



        #obs = self.observation.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = {'action': action}

        obs = self.road.update_detectors_period()
        print(obs)

        self._save(obs)

        return obs, reward, terminal, info

    def _simulate(self, action=None) -> None:
        """
            Perform several steps of simulation with constant action
        """
        for i in range(self.n_intermediate_steps):
            if action is not None \
                    and self.time % self.n_intermediate_steps == 0 \
                    and not self.road.lane_reversal_in_progress:
                self._act(action)

            self.road.step(dt=self.dt)
            self.time += 1

            self.inflow()
            self.outflow()

            self._automatic_rendering()

            if self.done or self._is_terminal():
                break

        self.enable_auto_render = False

    def _act(self, action: Optional[int] = None):
        action = self.ACTIONS[action]

        if action == 'INCREASE_UPSTREAM':
            self.road.initiate_lane_reversal(upstream=True)

        elif action == 'INCREASE_DOWNSTREAM':
            self.road.initiate_lane_reversal(upstream=False)

    def inflow(self):
        self.spawn_upstream()
        self.spawn_downstream()

    def spawn_upstream(self):
        demand = self.demand_up[self.steps]
        for _ in range(self.n_vehicles_to_spawn(demand)):
            if self.spawn_vehicle(self.road.network.upstream_source_lanes):
                self.metrics.total_upstream_in += 1

    def spawn_downstream(self):
        demand = self.demand_down[self.steps]
        for _ in range(self.n_vehicles_to_spawn(demand)):
            if self.spawn_vehicle(self.road.network.downstream_source_lanes):
                self.metrics.total_downstream_in += 1

    def spawn_vehicle(self, lanes):
        source_lanes = [lane for lane in lanes if not lane.forbidden]
        lane = self.np_random.choice(source_lanes)
        return self.road.spawn_car_on_lane(lane)

    def outflow(self):
        for c in self.road.cars:
            if c.should_kill:
                c.kill()
                if c.lane.upstream:
                    self.metrics.total_upstream_out += 1
                else:
                    self.metrics.total_downstream_out += 1

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        return self.steps >= self.experiment.DURATION

    def _cost(self, action):
        """
            A constraint metric, for budgeted MDP.

            If a constraint is defined, it must be used with an alternate reward that doesn't contain it
            as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        return 0

    def get_mean_arrival_rate(self, demand):
        return demand / 3600 / self.experiment.SIMULATION_FREQUENCY

    def n_vehicles_to_spawn(self, demand):
        mean_arrival_rate = self.get_mean_arrival_rate(demand)
        return np.random.poisson(mean_arrival_rate)

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.done = True

        if self.viewer is not None:
            self.viewer.close()

        self.viewer = None

    def render(self, mode: str = 'human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """

        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        self.viewer.handle_events()

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

        self.should_update_rendering = False

    def _automatic_rendering(self):
        """
            Automatically render the intermediate frames while an action is still ongoing.
            This allows to render the whole video and not only single steps corresponding to agent decision-making.

            If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
            such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback is not None:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def get_display_metrics(self):
        env_metrics = {
            "Steps": self.steps,
            "Time":  self.time,
        }

        env_metrics = [f'{k} = {v}' for k, v in env_metrics.items()]

        upstream_metrics, downstream_metrics, global_metrics = self.road.get_display_metrics()
        return env_metrics, upstream_metrics, downstream_metrics, global_metrics

    def _save(self, obs):
        self._history.append(obs)

    def save_history(self):
        filename = METRIC_DIR / f'simulation_{int(datetime.now().timestamp())}_metrics.csv'
        self.history.to_csv(filename, index=False)

    @property
    def current_state(self):
        if len(self._history) == 0:
            return None
        return self._history[-1]


register(id='highway-v0', entry_point='simulation.envs:HighwayEnv')
