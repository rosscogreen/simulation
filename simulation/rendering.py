from typing import Union, Tuple

import numpy as np
import pygame
from pygame.locals import *
from pygame import Surface
from numpy import deg2rad, rad2deg
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from simulation.config import LANE_WIDTH, NUM_LANES
from simulation.constants import CAR_LENGTH, CAR_WIDTH
from simulation.custom_types import CL, NL, SL

BLACK = (0, 0, 0)
GREY = (100, 100, 100)
GREEN = (0, 100, 0)
YELLOW = (200, 200, 0)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
BLUE = (100, 200, 255)

SCALING = 3.36
METRIC_HEIGHT = 100
ROAD_HEIGHT = (8 + NUM_LANES) * LANE_WIDTH * SCALING
SCREEN_WIDTH = 1680
SCREEN_HEIGHT = int(ROAD_HEIGHT + (3 * METRIC_HEIGHT))


class WorldSurface(pygame.Surface):
    def __init__(self, size: Tuple[int, int], scaling: Union[int, float]):
        flags = DOUBLEBUF
        # flags = 0
        super(WorldSurface, self).__init__(size, flags, Surface(size))
        # self.set_alpha(0)
        self.origin = np.array([0, 0])
        self.scaling = scaling

    def pix(self, length):
        return int(length * self.scaling)

    def pos2pix(self, x, y):
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec):
        return self.pos2pix(vec[0], vec[1])


class Screen(object):

    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, color=WHITE, offscreen=False):
        self.offscreen = offscreen
        pygame.init()
        pygame.display.set_caption("Dynamic Lane Reversal Simulation")
        if not offscreen:
            surface = pygame.display.set_mode((width, height))
            surface.fill(color)

            self.surface = surface

    def display(self, surface, pos=None):
        if pos is None:
            pos = (0, 0)
        if not self.offscreen:
            self.surface.blit(surface, pos)

    def update(self):
        if not self.offscreen:
            pygame.display.flip()


class Cars(pygame.sprite.Group):
    MIN_ANGLE = deg2rad(2)
    HALF_LENGTH = CAR_LENGTH / 2

    image = None

    def _load_image(self, surface, offscreen):
        width = surface.pix(CAR_LENGTH)
        height = surface.pix(CAR_WIDTH)
        position = (0, (width / 2) - (height / 2))
        image = pygame.image.load("/Users/rossgreen/PycharmProjects/simulation/img/car.png")

        if not offscreen:
            image = image.convert()

        scaled_image = pygame.transform.scale(image, (width, height))
        surf = pygame.Surface((width, width))
        surf.blit(scaled_image, position)

        self.image = surf

    def draw(self, surface, offscreen):
        if self.image is None:
            self._load_image(surface, offscreen)

        to_pix = surface.vec2pix
        blit = surface.blit
        half_length = self.HALF_LENGTH

        for car in self.sprites():
            if car.image is None:
                car.image = self.image.copy()
            car.rect = blit(car.image, to_pix(car.position - half_length))

    def draw2(self, surface, offscreen):
        if self.image is None:
            self._load_image(surface, offscreen)

        rotate = pygame.transform.rotate
        to_pix = surface.vec2pix
        blit = surface.blit
        half_length = self.HALF_LENGTH

        for car in self.sprites():
            if car.image is None:
                car.image = self.image.copy()

            position = to_pix(car.position - half_length)

            heading = car.heading
            angle = rad2deg(-heading if abs(heading) > self.MIN_ANGLE else 0)
            rotated = rotate(car.image, angle)
            car.rect = blit(rotated, position)


class Lane(object):
    STRIPE_SPACING = 5.0
    STRIPE_LENGTH = 3.0
    STRIPE_WIDTH = 0.3

    @classmethod
    def draw(cls, surface, lane):
        x1, y1 = lane.start_pos
        x2, y2 = lane.end_pos
        lw = lane.width / 2
        length = lane.length
        starts = (x1, y1 - lw), (x1, y1 + lw)
        ends = (x2, y2 - lw), (x2, y2 + lw)
        rs = (-lw, lw)

        for line_type, start, end, r in zip(lane.line_types, starts, ends, rs):
            if line_type == NL:
                continue
            if line_type == CL:
                draw_line(surface, lane.position(0, r), lane.position(lane.length, r))
            else:
                s_starts = np.arange(length // cls.STRIPE_SPACING) * cls.STRIPE_SPACING
                s_length = cls.STRIPE_LENGTH if line_type == SL else cls.STRIPE_SPACING
                s_ends = s_starts + s_length
                for s1, s2 in zip(s_starts, s_ends):
                    draw_line(surface, lane.position(s1, r), lane.position(s2, r))


class Lanes(object):
    def __init__(self, env, width, height, scaling):
        self.env = env
        self.surface = WorldSurface((width, height), scaling)

    def draw(self):
        if self.env.road.redraw:
            self.surface.fill(GREY)
            for lane in self.env.road.network:
                Lane.draw(self.surface, lane)
            self.draw_detectors()
            self.env.road.redraw = False

    def draw_detectors(self):
        y1 = 0
        y2 = self.surface.get_height()

        for x1, x2, x in self.env.road.detectors.x_ranges:
            draw_striped_line(surface=self.surface,
                              color=YELLOW,
                              start_pos=np.array([x, y1]),
                              end_pos=np.array([x, y2]))


class Road(object):
    height = (8 + NUM_LANES) * LANE_WIDTH * SCALING

    def __init__(self, env, width=SCREEN_WIDTH, height=None, scaling=SCALING):
        self.env = env
        height = height or self.height
        self.surface = WorldSurface((width, height), scaling)
        self.lane_surface = Lanes(env, width, height, scaling)

    def draw(self):
        self.lane_surface.draw()
        self.surface.blit(self.lane_surface.surface, (0, 0))
        self.env.road.cars.draw(self.surface, offscreen=self.env.offscreen)


class Metrics(object):
    height = 100

    def __init__(self, width=SCREEN_WIDTH, height=None, scaling=SCALING, font=None):
        self.height = height or self.height
        self.surface = WorldSurface((width, self.height), scaling)
        self.reset()
        self.font = font or pygame.font.SysFont('timesnewroman', 20)

    def reset(self):
        self.surface.fill(WHITE)

    def draw_metric(self, metric, x, y):
        img = self.font.render(metric, True, BLACK)
        self.surface.blit(img, self.surface.vec2pix((x, y)))


class DetectorMetrics(Metrics):

    def __init__(self, width=SCREEN_WIDTH, height=None, scaling=SCALING, font=None, detectors=None):
        self.height = height or self.height
        super(DetectorMetrics, self).__init__(width, height, scaling, font)
        self.detectors = detectors

    def draw(self, y_start, upstream=True):
        self.reset()
        for x, detector in self.detectors.items():
            self.draw_detector(detector, y_start, upstream=upstream)

    def draw_detector(self, detector, y_start, gap=6, upstream=True):
        metrics = [f"flow: {int(detector.flow)} veh/hr",
                   f"speed: {int(detector.speed)} km/h"]

        x = detector.x

        if upstream:
            if x == 490:
                x -= 25
        else:
            if x == 10:
                x += 25
            x = 500 - x

        for i, metric in enumerate(metrics):
            y = y_start + (i * gap)
            self.draw_metric(metric, x, y)


class Viewer(object):
    DISPLAY_METRICS = False

    def __init__(self, env):
        self.env = env
        self.offscreen = self.env.offscreen
        self.fps = self.env.simulation_frequency
        self.screen = Screen(offscreen=self.offscreen)
        self.road = Road(env)

        font = pygame.font.SysFont('timesnewroman', 20)
        self.upstream_detectors = DetectorMetrics(
                detectors=self.env.road.detectors.upstream_detectors,
                font=font)

        self.downstream_detectors = DetectorMetrics(
                detectors=self.env.road.detectors.downstream_detectors,
                font=font)

        self.metrics = Metrics(font=font)

        self.clock = pygame.time.Clock()

    def display(self):
        self.draw_road()
        self.screen.update()
        self.clock.tick(self.fps)

    def draw_road(self):
        self.road.draw()
        self.screen.display(self.road.surface, (0, self.upstream_detectors.height))

    def update_metrics(self):
        if not self.DISPLAY_METRICS:
            return

        self.upstream_detectors.draw(y_start=14)
        self.screen.display(self.upstream_detectors.surface)

        self.downstream_detectors.draw(y_start=6, upstream=False)
        self.screen.display(self.downstream_detectors.surface, (0, self.upstream_detectors.height + self.road.height))

        self.metrics.reset()
        self.metrics.draw_metric(f'step: {self.env.current_step}', 5, 0)
        self.metrics.draw_metric(f'upstream demand: {int(self.env.upstream_demand_for_step * 4)} veh/hr', 5, 6)
        self.metrics.draw_metric(f'downstream demand: {int(self.env.downstream_demand_for_step * 4)} veh/hr', 5, 12)
        self.screen.display(self.metrics.surface,
                            (0, self.upstream_detectors.height + self.road.height + self.downstream_detectors.height))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.env.close()
            elif event.type == KEYDOWN:
                key = event.key
                if key == K_ESCAPE:
                    self.env.close()
                if key == K_UP:
                    self.env._act(self.env.ADD_UPSTREAM)
                elif key == K_DOWN:
                    self.env._act(self.env.ADD_DOWNSTREAM)

    def close(self):
        pygame.quit()

    def get_image(self):
        data = pygame.surfarray.array3d(self.road.surface)
        return np.moveaxis(data, 0, 1)

    def save(self):
        pygame.image.save(self.road.surface, 'images/test.png')


def draw_line(surface, start_pos, end_pos, width=1, color=WHITE):
    pygame.draw.line(surface, color, surface.vec2pix(start_pos), surface.vec2pix(end_pos), width)


def draw_striped_line(surface, start_pos, end_pos,
                      stripe_spacing=2, stripe_length=1, width=1, color=WHITE):
    x1, y1 = start_pos
    x2, y2 = end_pos
    length = y2 - y1
    starts = np.arange(length // stripe_spacing) * stripe_spacing
    ends = starts + stripe_length
    for y1, y2 in zip(starts, ends):
        draw_line(surface, (x1, y1), (x1, y2), color=color, width=width)
