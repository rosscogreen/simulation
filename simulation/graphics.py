import os
from typing import Union, Tuple

import numpy as np
import pygame
from numpy import deg2rad, rad2deg
from pygame.locals import *
from pygame import Surface

from simulation.car import Car
from simulation.config import DISPLAY_METRICS, LANE_WIDTH, NUM_LANES
from simulation.custom_types import CL, NL, SL

MIN_ANGLE = deg2rad(2)

TOP_LEFT = (0, 0)

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 1050
SCALING = 3.36

ROAD_HEIGHT = (8 + NUM_LANES) * LANE_WIDTH * SCALING
METRIC_HEIGHT = int((SCREEN_HEIGHT / 2) - (ROAD_HEIGHT / 2))

ROAD_POS = (0, (2.5 / 8) * SCREEN_HEIGHT)

############## Pygame Colors ######################
BLACK = (0, 0, 0)
GREY = (100, 100, 100)
GREEN = (0, 100, 0)
YELLOW = (200, 200, 0)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
BLUE = (100, 200, 255)

STRIPE_SPACING = 5.0
STRIPE_LENGTH = 3.0
STRIPE_WIDTH = 0.3


class WorldSurface(Surface):
    """
        A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area.
    """

    def __init__(self, size: Tuple[int, int], scaling: Union[int, float], flags=0):
        super(WorldSurface, self).__init__(size, flags, Surface(size))
        self.origin = np.array([0, 0])
        self.scaling = scaling

    def pix(self, length):
        """
            Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x, y):
        """
            Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec):
        """
             Convert a world position [m] into a position in the surface [px].
        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])


class Cars(pygame.sprite.RenderUpdates):
    IMAGE = None

    def draw(self, surface: WorldSurface, offscreen):
        dirty = self.lostsprites
        self.lostsprites = []

        dirty_append = dirty.append
        rot = pygame.transform.rotate
        to_pix = surface.pos2pix
        blit = surface.blit

        for sprite in self.sprites():

            if sprite.image is None:
                self.load_image(sprite, surface, offscreen)

            sprite_rect = self.spritedict[sprite]

            x, y = sprite.position
            rect = to_pix(x - sprite.LENGTH / 2, y - sprite.LENGTH / 2)

            # Rotate surface to match vehicle rotation
            rotate_angle = rad2deg(-sprite.heading if abs(sprite.heading) > MIN_ANGLE else 0)
            rotated = rot(sprite.image, rotate_angle)

            newrect = blit(rotated, rect)
            sprite.rect = newrect

            if sprite_rect:
                if newrect.colliderect(sprite_rect):
                    dirty_append(newrect.union(sprite_rect))
                else:
                    dirty_append(newrect)
                    dirty_append(sprite_rect)
            else:
                dirty_append(newrect)

            self.spritedict[sprite] = newrect

        return dirty

    def load_image(self, sprite: "Car", surface: "WorldSurface", offscreen=False):
        image_path = getattr(sprite, 'IMAGE_PATH', None)

        if image_path is not None:
            width = surface.pix(sprite.LENGTH)
            height = surface.pix(sprite.WIDTH)

            if self.IMAGE is None:
                if not offscreen:
                    self.IMAGE = pygame.image.load(str(image_path)).convert()
                else:
                    self.IMAGE = pygame.image.load(str(image_path))

            vehicle_surface = pygame.Surface((width, width), pygame.SRCALPHA)
            image = pygame.transform.scale(self.IMAGE, (width, height))
            vehicle_surface.blit(image, (0, (width / 2) - (height / 2)))

        else:
            width = surface.pix(3)
            height = surface.pix(2)

            vehicle_surface = pygame.Surface((width, height))
            vehicle_surface.fill(RED)
            vehicle_surface.set_alpha(255)

        sprite.image = vehicle_surface
        sprite.rect = vehicle_surface.get_rect()

    # def load_rect(self, surface, v):
    #     tire_length, tire_width = 1, 0.3
    #     length = v.LENGTH + 2 * tire_length
    #     vehicle_surface = pygame.Surface((surface.pix(length), surface.pix(length)),
    #                                      flags=pygame.SRCALPHA)  # per-pixel alpha
    #     rect = (
    #     surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2), surface.pix(v.LENGTH), surface.pix(v.WIDTH))
    #     pygame.draw.rect(vehicle_surface, RED, rect, 0)
    #     pygame.draw.rect(vehicle_surface, BLACK, rect, 1)
    #
    #     tire_positions = [[surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
    #                       [surface.pix(tire_length), surface.pix(length / 2 + v.WIDTH / 2)],
    #                       [surface.pix(length - tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
    #                       [surface.pix(length - tire_length), surface.pix(length / 2 + v.WIDTH / 2)]]
    #     tire_angles = [0, 0, v.action["steering"], v.action["steering"]]
    #     for tire_position, tire_angle in zip(tire_positions, tire_angles):
    #         tire_surface = pygame.Surface((surface.pix(tire_length), surface.pix(tire_length)), pygame.SRCALPHA)
    #         rect = (0, surface.pix(tire_length / 2 - tire_width / 2), surface.pix(tire_length), surface.pix(tire_width))
    #         pygame.draw.rect(tire_surface, cls.BLACK, rect, 0)
    #         cls.blit_rotate(vehicle_surface, tire_surface, tire_position, np.rad2deg(-tire_angle))
    #
    #     # Centered rotation
    #     h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
    #     position = [*surface.pos2pix(v.position[0], v.position[1])]
    #     if not offscreen:
    #         # convert_alpha throws errors in offscreen mode
    #         # see https://stackoverflow.com/a/19057853
    #         vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
    #     cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))


class EnvViewer(object):

    def __init__(self, env):
        self.env = env
        self.offscreen = env.offscreen

        pygame.init()
        pygame.display.set_caption("Dynamic Lane Reversal Simulation")

        if not self.offscreen:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.screen.fill(GREEN)

        # Draw road and cars
        self.road_surface = WorldSurface((SCREEN_WIDTH, ROAD_HEIGHT), SCALING)
        self.lane_surface = WorldSurface((SCREEN_WIDTH, ROAD_HEIGHT), SCALING)

        self.clock = pygame.time.Clock()


    def display(self):
        self.draw_road()
        self.draw_cars()

        if not self.offscreen:
            self.update_screen()
            self.clock.tick(self.env.simulation_frequency)

    def draw_road(self):
        if self.env.road.redraw:
            self.lane_surface.fill(GREY)
            for lane in self.env.road.network:
                self.draw_lane(lane)

            self.draw_detectors()

        self.road_surface.blit(self.lane_surface, (0, 0))

    def draw_cars(self):
        self.env.road.cars.draw(self.road_surface, self.offscreen)

    def draw_detectors(self):
        surface = self.lane_surface
        y1 = 0
        y2 = surface.get_height()

        for x1, x2, x in self.env.road.detectors.x_ranges:
            self.draw_striped_line(surface=surface,
                                   color=YELLOW,
                                   start_pos=np.array([x, y1]),
                                   end_pos=np.array([x, y2]))

    def update_screen(self):
        self.screen.blit(self.road_surface, ROAD_POS)
        pygame.display.flip()

    def draw_lane(self, lane: "AbstractLane"):
        x1, y1 = lane.start_pos
        x2, y2 = lane.end_pos
        lw = lane.width / 2
        length = lane.length
        starts = (x1, y1 - lw), (x1, y1 + lw)
        ends = (x2, y2 - lw), (x2, y2 + lw)
        surface = self.lane_surface
        rs = (-lw, lw)

        for line_type, start, end, r in zip(lane.line_types, starts, ends, rs):
            if line_type == NL:
                continue
            if line_type == CL:
                self.draw_line(surface, lane.position(0, r), lane.position(lane.length, r))
            else:
                s_starts = np.arange(length // STRIPE_SPACING) * STRIPE_SPACING
                s_length = STRIPE_LENGTH if line_type == SL else STRIPE_SPACING
                s_ends = s_starts + s_length
                for s1, s2 in zip(s_starts, s_ends):
                    self.draw_line(surface, lane.position(s1, r), lane.position(s2, r))

    def draw_striped_line(self, surface, start_pos, end_pos,
                          stripe_spacing=2, stripe_length=1, width=1, color=WHITE):
        x1, y1 = start_pos
        x2, y2 = end_pos
        length = y2 - y1
        starts = np.arange(length // stripe_spacing) * stripe_spacing
        ends = starts + stripe_length
        for y1, y2 in zip(starts, ends):
            self.draw_line(surface, (x1, y1), (x1, y2), color=color, width=width)

    def draw_line(self, surface, start_pos, end_pos, width=1, color=WHITE):
        pygame.draw.line(surface=surface,
                         color=color,
                         start_pos=surface.vec2pix(start_pos),
                         end_pos=surface.vec2pix(end_pos),
                         width=width)

    def close(self):
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.env.close()
            elif event.type == KEYDOWN:
                self.handle_key_event(event.key)

    def handle_key_event(self, key):
        if key == K_ESCAPE:
            self.env.close()
        if key == K_UP:
            self.env.act(1)
        elif key == K_DOWN:
            self.env.act(2)

    def get_image(self):
        data = pygame.surfarray.array3d(self.road_surface)
        return np.moveaxis(data, 0, 1)


class EnvViewer2(object):

    def __init__(self, env):
        self.env = env

        TITLE_FONT = pygame.font.Font('freesansbold.ttf', 20)
        DEFAULT_FONT = pygame.font.Font('freesansbold.ttf', 16)

        # Initialise Screen
        pygame.init()
        pygame.display.set_caption("Dynamic Lane Reversal Simulation")

        main_display_size = (SCREEN_WIDTH, SCREEN_HEIGHT)

        # Screen display surface
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        # self.screen.fill(GREEN)

        # Game display surface
        self.game_display = WorldSurface((SCREEN_WIDTH, SCREEN_HEIGHT), SCALING)
        self.game_display.fill(color=GREEN)

        # Road display surface

        self.road_display = WorldSurface((SCREEN_WIDTH, ROAD_HEIGHT), SCALING)
        self.road_graphics = RoadGraphics(road=self.env.road, surface=self.road_display)

        # Metrics Displays
        # self.metric_display = WorldSurface((SCREEN_WIDTH, METRIC_HEIGHT), SCALING)
        # self.metric_display.fill(WHITE)
        # self.init_metrics()

        # Timing
        self.clock = pygame.time.Clock()

    def init_metrics(self):
        y_metric_start = SCREEN_HEIGHT * (3 / 8)  # self.ROAD_HEIGHT
        x_metric_start = 20
        metric_width = (SCREEN_WIDTH - (2 * x_metric_start)) / 5
        self.metric_display_pos = (0, y_metric_start)
        self.metric_display_positions = [(x_metric_start + (i * metric_width), 10) for i in range(5)]

    def display(self):
        self.draw_road()
        self.draw_cars()
        self.update_screen()

    def display2(self):
        """
            Main method:
            Display the road_network and vehicles on a pygame window.
        """

        # update road lanes if any changes made to lanes
        # self.road_graphics.draw(surface=self.road_display_surface)
        self.road_graphics.draw()

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

    def draw_road(self):
        self.road_graphics.draw()
        self.game_display.blit(self.road_display, (0, 0))

    def draw_cars(self):
        dirty = self.env.road.cars.draw(self.game_display)

    def update_screen(self, dirty=None):
        self.screen.blit(self.game_display, (0, 0))
        if dirty is not None:
            pygame.display.update(dirty)
        else:
            pygame.display.flip()

    def draw_metrics(self):
        if not DISPLAY_METRICS:
            return

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

        self.me.fill(WHITE)

        p1, p2, p3, p4, p5 = self.metric_display_positions

        # Simulation Metrics
        self._display_metrics(metrics=env_metrics, pos=p1, title="Simulation")

        # Upstream Metrics
        self._display_metrics(metrics=upstream_metrics, pos=p2, title="Upstream Metrics")

        # Downstream Metrics
        self._display_metrics(metrics=downstream_metrics, pos=p3, title="Downstream Metrics")

        # Global Road Metrics
        self._display_metrics(metrics=global_metrics, pos=p4, title="Road Metrics")

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
