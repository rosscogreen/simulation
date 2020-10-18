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

    def __init__(self, np_random=None):
        self.np_random = np_random if np_random is not None else np.random
        self.upstream = []
        self.downstream = []

    def demand_up(self, i):
        return self.upstream[i]

    def demand_down(self, i):
        return self.downstream[i]

    def demand_for_step(self, step):
        return self.demand_up(step), self.demand_down(step)

    def generate_count(self, demand, dt):
        rate = (demand / 3600) * dt
        return self.np_random.poisson(rate)

    def counts_for_step(self, step, dt):
        up = self.generate_count(self.demand_up(step), dt)
        down = self.generate_count(self.demand_down(step), dt)
        return up, down

    def generate(self, amplitude, period, steps):
        x = np.linspace(0, period * 2 * np.pi, steps)

        self.upstream = ((np.sin(x) + 1) / 2) * amplitude
        self.downstream = amplitude - self.upstream

    def generate_random(self, steps):
        freq = np.abs(self.np_random.normal(2, 1)) * 2 * np.pi
        amplitude = self.np_random.normal(10000, 2000)
        x = np.linspace(0, freq, steps)

        self.upstream = self.np_random.normal(0.5, 0.1) * (np.sin(x) + self.np_random.normal(2, 0.2)) * amplitude
        self.downstream = np.min(self.upstream) + (np.max(self.upstream) - self.upstream)

    def generate_from_file(self, filename, demand_multiplier):
        df = pd.read_csv(filename, parse_dates=['Quarter Hour'], index_col='Quarter Hour')
        demand_per_step = df.values * demand_multiplier
        demand_per_step = demand_per_step.astype(int)
        return demand_per_step
