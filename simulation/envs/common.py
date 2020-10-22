from typing import List
import pandas as pd
import numpy as np
import gym
from gym.utils import seeding

from simulation.rendering import Viewer

FLOW_FILE_PATH = '/Users/rossgreen/PycharmProjects/simulation/resources/flows.csv'


class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.np_random = None

        # Running
        self.done = False
        self.steps = 0
        self.simulation_steps = 0

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        self.seed()

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'human'):
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = Viewer(self)

        self.enable_auto_render = True

        if self.should_update_rendering:
            self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

        self.should_update_rendering = False

    def _render(self):
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


class Demand:

    def __init__(self, env):
        self.env = env
        self.steps = env.total_simulation_steps
        self.amplitude = 15000

        self.upstream = []
        self.downstream = []
        self.upstream_bottleneck = []
        self.downstream_bottleneck = []

    @property
    def demand_up(self):
        return self.upstream[self.env.simulation_steps]

    @property
    def demand_down(self):
        return self.downstream[self.env.simulation_steps]

    @property
    def demand_for_step(self):
        return self.demand_up, self.demand_down

    def generate_count(self, demand):
        rate = (demand / 3600) * self.env.dt
        return self.env.np_random.poisson(rate)

    @property
    def counts_for_step(self):
        return self.generate_count(self.demand_up), self.generate_count(self.demand_down)

    def generate(self):
        self.amplitude = self.env.params.get('demand_amplitude', 15000)
        experiment = self.env.experiment
        if experiment == 'experiment_1':
            self.generate_experiement_1()
        elif experiment == 'experiment_2':
            self.generate_experiement_2()
        elif experiment == 'experiment_3':
            self.generate_experiement_3()
        elif experiment == 'experiment_4':
            self.generate_experiement_4()
        else:
            self.generate_experiement_1()

    def generate_random(self):
        freq = np.abs(self.env.np_random.normal(2, 1)) * 2 * np.pi
        amplitude = self.env.np_random.normal(10000, 2000)
        x = np.linspace(0, freq, self.steps)

        self.upstream = self.env.np_random.normal(0.5, 0.1) * (
                    np.sin(x) + self.env.np_random.normal(2, 0.2)) * amplitude
        self.downstream = np.min(self.upstream) + (np.max(self.upstream) - self.upstream)

    def generate_from_file(self, filename, demand_multiplier):
        df = pd.read_csv(filename, parse_dates=['Quarter Hour'], index_col='Quarter Hour')
        demand_per_step = df.values * demand_multiplier
        demand_per_step = demand_per_step.astype(int)
        return demand_per_step

    def gen_sine(self, period, amplitude):
        freq = period * 2 * np.pi
        x = np.linspace(0, freq, self.steps)
        y1 = ((np.sin(x) + 1) / 2) * amplitude
        y2 = amplitude - y1
        return y1, y2

    def generate_experiement_1(self):
        """ Peak hour traffic"""
        self.upstream, self.downstream = self.gen_sine(period=1, amplitude=self.amplitude)

    def generate_experiement_2(self):
        """ Higher freq peak hour traffic"""
        self.upstream, self.downstream = self.gen_sine(period=3, amplitude=self.amplitude)

    def generate_experiement_3(self):
        """ Random Traffic"""
        self.upstream = self.get_random_series()
        self.downstream = self.get_random_series()

    def generate_experiement_4(self):
        """ Bottleneck """
        self.upstream = self.downstream = np.zeros([self.steps]) + 8000
        self.upstream_bottleneck, self.downstream_bottleneck = self.gen_sine(period=2, amplitude=8000)
        #self.upstream_bottleneck = self.downstream_bottleneck = np.zeros([self.steps]) + 6000

    def get_random_series(self):
        return pd.Series(
                self.env.np_random.uniform(0, self.amplitude, self.env.total_steps)
        ).repeat(self.env.updates_per_step).values
