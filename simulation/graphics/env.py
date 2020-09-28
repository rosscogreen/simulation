import os

import numpy as np
import pygame
from pygame.locals import *
from simulation.graphics.common import WorldSurface
from typing import Union

from simulation.config import SCREEN_WIDTH, SCREEN_HEIGHT, SCALING, DISPLAY_METRICS, LANE_WIDTH, NUM_LANES
from simulation.graphics.common import GREY, WHITE, BLACK, GREEN
from simulation.graphics.road import RoadGraphics

TOP_LEFT = (0, 0)

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

FLOW_DEMAND_STEP_INCREASE = 100


class EnvViewer(object):

    def __init__(self, env):
        self.env = env

        # Initialise Screen
        pygame.init()
        pygame.display.set_caption("Traffic Simulation")

        main_display_size = (SCREEN_WIDTH, SCREEN_HEIGHT)

        # Screen display surface
        self.main_display_surface = pygame.display.set_mode(size=main_display_size)
        self.main_display_surface.fill(GREEN)

        # Game display surface
        self.game_display_surface = WorldSurface(main_display_size, SCALING)
        self.game_display_surface.fill(color=GREEN)

        # Road display surface
        road_height = (8 + NUM_LANES) * LANE_WIDTH * SCALING
        self.road_display_surface = WorldSurface((SCREEN_WIDTH, road_height), SCALING)

        # Metric Displays

        self.road_graphics = RoadGraphics(road=self.env.road)

        self.metric_car = None

        self.title_font = pygame.font.Font('freesansbold.ttf', 20)
        self.default_font = pygame.font.Font('freesansbold.ttf', 16)

        y_metric_start = SCREEN_HEIGHT * (3 / 8)  # self.ROAD_HEIGHT
        x_metric_start = 20
        metric_width = (SCREEN_WIDTH - (2 * x_metric_start)) / 5

        self.metric_display_pos = (0, y_metric_start)

        self.metric_display_surface = WorldSurface((SCREEN_WIDTH, int((SCREEN_HEIGHT / 2) - (road_height / 2))),
                                                   SCALING)
        self.metric_display_surface.fill(WHITE)

        self.metric_display_positions = [(x_metric_start + (i * metric_width), 10) for i in range(5)]

        # Timing
        self.clock = pygame.time.Clock()

    def display(self):
        """
            Main method:
            Display the road_network and vehicles on a pygame window.
        """

        # update road lanes if any changes made to lanes
        self.road_graphics.draw(surface=self.road_display_surface)

        # Display lanes on game display
        self.game_display_surface.blit(source=self.road_display_surface, dest=TOP_LEFT)

        # Update vehicle positions and display
        dirty = self.env.road.cars.draw(self.game_display_surface)

        if DISPLAY_METRICS:
            self._update_metric_display()
            self.game_display_surface.blit(source=self.metric_display_surface, dest=self.metric_display_pos)

        # Update game display on screen
        self.main_display_surface.blit(source=self.game_display_surface, dest=(0, (2.5 / 8) * SCREEN_HEIGHT))

        # Display screen changes
        pygame.display.flip()
        # pygame.display.update(dirty)

        # Update game clock
        self.clock.tick(self.env.experiment.SIMULATION_FREQUENCY)

    def _update_metric_display(self):
        state = self.env.current_state
        if state is None:
            return

        env_metrics = {
            "Steps": state['steps'],
            "Time":  state['time'],
        }

        upstream_metrics = {
            "Lanes":                    state['upstream_lanes'],
            "Demand (veh/hr)":          state['upstream_demand'],
            "Queue In":                 state['upstream_queue'],
            "Density (veh/km)":         state['upstream_density'],
            "Space Mean Speed (km/hr)": state['upstream_velocity'],
            "Flow":                     state['upstream_flow'],
        }

        downstream_metrics = {
            "Lanes":                    state['downstream_lanes'],
            "Demand (veh/hr)":          state['downstream_demand'],
            "Queue In":                 state['downstream_queue'],
            "Density (veh/km)":         state['downstream_density'],
            "Space Mean Speed (km/hr)": state['downstream_velocity'],
            "Flow":                     state['downstream_flow'],
        }

        global_metrics = {
            "Total Flow": state['upstream_flow'] + state['downstream_flow'],
        }

        self.metric_display_surface.fill(WHITE)

        p1, p2, p3, p4, p5 = self.metric_display_positions

        # Simulation Metrics
        self._display_metrics(metrics=env_metrics, pos=p1, title="Simulation")

        # Upstream Metrics
        self._display_metrics(metrics=upstream_metrics, pos=p2, title="Upstream Metrics")

        # Downstream Metrics
        self._display_metrics(metrics=downstream_metrics, pos=p3, title="Downstream Metrics")

        # Global Road Metrics
        self._display_metrics(metrics=global_metrics, pos=p4, title="Road Metrics")

        # Selected Vehicle Metrics
        if self.metric_car and self.env.road.cars.has(self.metric_car):
            self._display_metrics(metrics=self.metric_car.metrics(), pos=p5, title="Vehicle Metrics")

    def _display_metrics(self, metrics, pos, title=None):
        if title:
            pos = self._display_text(text=title, pos=pos, font=self.title_font, y_increase=2)

        for k, v in metrics.items():
            if isinstance(v, float):
                v = round(v, 3)
            metric_txt = f'{k} = {v}'
            pos = self._display_text(metric_txt, pos)

    def _display_text(self, text: Union[str, pygame.font.FontType], pos, font=None, y_increase=1.5):
        if isinstance(text, str):
            font = font or self.default_font
            text = font.render(text, True, BLACK)

        rect = text.get_rect(topleft=pos)

        self.metric_display_surface.blit(text, rect)

        x, y, w, h = rect
        y += h * y_increase

        return x, y

    def close(self):
        """
            Close the pygame window.
        """
        pygame.quit()

    def handle_events(self):
        """
            Handle pygame events by forwarding them to the display and environment vehicles.
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                self.env.close()

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.env.close()
                elif event.key == K_UP:
                    self.env._act(1)
                elif event.key == K_DOWN:
                    self.env._act(2)

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.game_display_surface)
        return np.moveaxis(data, 0, 1)
