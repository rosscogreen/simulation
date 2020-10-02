import os
from typing import Union, Tuple

import numpy as np
import pygame
from numpy import deg2rad, rad2deg
from pygame import Surface
from pygame.locals import *

from simulation.car import Car
from simulation.config import DISPLAY_METRICS, LANE_WIDTH, NUM_LANES
from simulation.custom_types import CL, NL, SL

MIN_ANGLE = deg2rad(2)

TOP_LEFT = (0, 0)

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"



#ROAD_HEIGHT = (8 + NUM_LANES) * LANE_WIDTH * SCALING
#METRIC_HEIGHT = int((SCREEN_HEIGHT / 2) - (ROAD_HEIGHT / 2))

#ROAD_POS = (0, (2.5 / 8) * SCREEN_HEIGHT)

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
    SCALING = 3.36
    METRIC_HEIGHT = 100
    ROAD_HEIGHT = (8 + NUM_LANES) * LANE_WIDTH * SCALING
    SCREEN_WIDTH = 1680
    SCREEN_HEIGHT = int(ROAD_HEIGHT + (3*METRIC_HEIGHT))



    def __init__(self, env):
        self.env = env
        self.offscreen = env.offscreen

        pygame.init()
        pygame.display.set_caption("Dynamic Lane Reversal Simulation")

        if not self.offscreen:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.screen.fill(WHITE)

        # Draw road and cars
        self.road_surface = WorldSurface((self.SCREEN_WIDTH, self.ROAD_HEIGHT), self.SCALING)
        self.lane_surface = WorldSurface((self.SCREEN_WIDTH, self.ROAD_HEIGHT), self.SCALING)

        self.font = None
        if DISPLAY_METRICS:
            self.upstream_metric_display = WorldSurface((self.SCREEN_WIDTH, self.METRIC_HEIGHT), self.SCALING)
            self.downstream_metric_display = WorldSurface((self.SCREEN_WIDTH, self.METRIC_HEIGHT), self.SCALING)
            self.metrics_display = WorldSurface((self.SCREEN_WIDTH, self.METRIC_HEIGHT), self.SCALING)
            self.font = pygame.font.SysFont('timesnewroman', 20)

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

    def update_metric_displays(self):
        if not DISPLAY_METRICS:
            return

        self.upstream_metric_display.fill(WHITE)
        self.downstream_metric_display.fill(WHITE)
        self.metrics_display.fill(WHITE)

        for x, detector in self.env.road.detectors.upstream_detectors.items():
            if x == 490:
                x=x-25
            self.draw_detector_metrics(detector, self.upstream_metric_display, x)

        for x, detector in self.env.road.detectors.downstream_detectors.items():
            if x == 490:
                x=x-25
            self.draw_detector_metrics(detector, self.downstream_metric_display, x)


        self.metrics_display.blit(self.font.render(f'step: {self.env.current_step}', True, BLACK), (20,20))
        self.metrics_display.blit(self.font.render(f'upstream demand: {self.env.upstream_demand_for_step*4} veh/hr', True, BLACK), (20,40))
        self.metrics_display.blit(self.font.render(f'downstream demand: {self.env.downstream_demand_for_step*4} veh/hr', True, BLACK), (20,60))

        self.screen.blit(self.upstream_metric_display, (0, 0))
        self.screen.blit(self.downstream_metric_display, (0, self.ROAD_HEIGHT + self.METRIC_HEIGHT))
        self.screen.blit(self.metrics_display, (0, self.ROAD_HEIGHT + (2*self.METRIC_HEIGHT)))

    def draw_detector_metrics(self, detector, surface, x):
        obs = detector.obs

        metrics = [f"flow: {round(obs['flow'], 1)} veh/hr", f"speed: {round(obs['speed'], 1)} km/h"]
        for i, metric in enumerate(metrics):
            img = self.font.render(metric, True, BLACK)
            pos = surface.vec2pix((x, (i*5) + 10))
            surface.blit(img, pos)

    def update_screen(self):
        self.screen.blit(self.road_surface, (0, self.METRIC_HEIGHT))
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
        pygame.draw.line(surface, color, surface.vec2pix(start_pos), surface.vec2pix(end_pos), width)

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
