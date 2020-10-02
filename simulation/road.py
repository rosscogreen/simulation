import logging
from functools import lru_cache
from typing import Optional
import numpy as np

from scipy.stats import hmean
from numpy import mean

from gym.utils import seeding

from simulation.car import Car
from simulation.config import WEST_NODE, EAST_NODE
from simulation.constants import CAR_LENGTH, MIN_INSERTION_GAP
from simulation.custom_types import CL, SL
from simulation.detector import Detectors
from simulation.graphics import Cars
from simulation.idm import IDM
from simulation.lanes import AbstractLane
from simulation.network import RoadNetwork

logger = logging.getLogger(__name__)

VERBOSE = 0


class Road(object):
    east = 'east'
    west = 'west'

    CAR_HALF_LENGTH = CAR_LENGTH / 2

    def __init__(self, np_random=None):
        self.network: "RoadNetwork" = RoadNetwork()
        self.cars = Cars()

        self.np_random = np_random or seeding.np_random()[0]

        # Lane reversal State
        self.is_lane_reversal_in_progress: bool = False
        self.new_active_lane: Optional["AbstractLane"] = None
        self.new_forbidden_lane: Optional["AbstractLane"] = None

        self.detectors = Detectors()

        self.total_lane_count = 8
        self.upstream_lane_count = 4
        self.downstream_lane_count = 4

        # only redraw on changes to lane structure
        self.redraw = True

    def step(self, dt: float):
        self.cars.update(dt)
        self.update_position_state()
        self.lane_reversal_step()

    def get_active_lanes(self, lanes):
        return [lane for lane in lanes if not lane.forbidden]

    def update_position_state(self):
        for car in self.cars:
            self.detectors.update(car)

    def initiate_lane_reversal(self, upstream: bool):
        i = self.upstream_lane_count

        if upstream and i == 7:
            return

        if not upstream and i == 1:
            return

        self.is_lane_reversal_in_progress = True

        n = self.total_lane_count

        get_lane = self.network.get_lane

        if upstream:
            self.new_active_lane = get_lane((WEST_NODE, EAST_NODE, i + 1))
            self.new_forbidden_lane = get_lane((EAST_NODE, WEST_NODE, n - i))
        else:
            self.new_active_lane = get_lane((EAST_NODE, WEST_NODE, n - i + 1))
            self.new_forbidden_lane = get_lane((WEST_NODE, EAST_NODE, i))

        self.new_forbidden_lane.forbidden = True
        for c in self.cars_on_lane(self.new_forbidden_lane):
            c.set_mandatory_merge(to_right=False)

    def lane_reversal_step(self):
        if not self.is_lane_reversal_in_progress:
            return

        if not self.is_lane_empty(self.new_forbidden_lane):
            return

        # Redraw lines
        if self.new_active_lane.upstream:
            o, d, i = self.new_active_lane.index
            self.new_active_lane.line_types[0] = SL
            self.network.get_lane((o, d, i + 1)).line_types[0] = CL
        else:
            # new_forbidden is upstream
            o, d, i = self.new_forbidden_lane.index
            self.new_forbidden_lane.line_types[0] = CL
            self.network.get_lane((o, d, i + 1)).line_types[0] = SL

        self.redraw = True
        self.new_active_lane.forbidden = False
        self.new_active_lane = None
        self.new_forbidden_lane = None
        self.is_lane_reversal_in_progress = False

        self.upstream_lane_count = sum(
                [not lane.forbidden for lane in self.network.get_lanes(WEST_NODE, EAST_NODE)]) + 1
        self.downstream_lane_count = self.total_lane_count - self.upstream_lane_count

    def is_lane_empty(self, lane: "AbstractLane"):
        for car in self.cars:
            if lane.same_lane(car.lane):
                return False
        return True

    def spawn(self, n: int, upstream: bool):
        lanes = self.network.upstream_source_lanes if upstream else self.network.downstream_source_lanes
        active_lanes = self.get_active_lanes(lanes)

        added = 0
        for i in range(n):
            self.np_random.shuffle(active_lanes)
            for lane in active_lanes:
                if self.spawn_car_on_lane(lane):
                    added += 1
                    break


    def spawn_car_on_lane(self, lane: "AbstractLane", destination=None) -> bool:
        fwd = self.first_car_on_lane(lane, merging=True)

        speed = lane.speed_limit

        if fwd:
            gap = fwd.s_path
            if gap < MIN_INSERTION_GAP:
                return False
            speed = IDM.calc_max_initial_speed(gap=gap, fwd_speed=fwd.speed, speed_limit=speed)

        car = Car.make_on_lane(road=self, lane=lane, speed=speed)

        if destination:
            car.plan_route_to(destination)

        return True

    def cars_on_lane(self, lane, merging=False):
        if merging:
            return [c for c in self.cars if lane.same_lane(c.lane) or lane.same_lane(c.target_lane)]
        else:
            return [c for c in self.cars if lane.same_lane(c.lane)]

    def first_car_on_lane(self, lane, merging=False) -> Optional["Car"]:
        cars = self.cars_on_lane(lane, merging)
        if not cars:
            return None
        return min(cars, key=lambda c: c.s_path)

    def get_neighbours(self, car, lane=None):
        lane = lane or car.lane
        #s_me = lane.s(car.position)
        s_me = lane.path_coordinates(car.position)[0]

        cars = self.cars_on_lane(lane)

        s_fwd = 500
        s_bwd = 0
        fwd = None
        bwd = None

        for c in cars:
            s = c.s_path
            if s_me < s <= s_fwd:  # Car in front
                s_fwd = s
                fwd = c
            elif s_bwd < s < s_me:
                s_bwd = s
                bwd = c

        return fwd, bwd

    def get_fwd_car(self, car, lane=None):
        lane = lane or car.lane
        s_me = lane.path_coordinates(car.position)[0]
        cars = [c for c in self.cars_on_lane(lane) if c.s_path >= s_me]
        if not cars:
            return None
        return min(cars, key=lambda c: c.s_path)


    def report(self):
        upstream_highway_cars = []
        upstream_onramp_cars = []
        upstream_offramp_cars = []
        downstream_highway_cars = []
        downstream_onramp_cars = []
        downstream_offramp_cars = []

        for car in self.cars:
            lane = car.lane
            o, d, i = lane.index
            if lane.upstream:
                if lane.is_onramp:
                    upstream_onramp_cars.append(car)
                elif lane.is_offramp:
                    upstream_offramp_cars.append(car)
                elif i != 0:
                    upstream_highway_cars.append(car)
            else:
                if lane.is_onramp:
                    downstream_onramp_cars.append(car)
                elif lane.is_offramp:
                    downstream_offramp_cars.append(car)
                elif i != 0:
                    downstream_highway_cars.append(car)

        return {
            **self._highway_report(upstream_highway_cars, True),
            **self._highway_report(downstream_highway_cars, False)
        }

    def _highway_report(self, cars, upstream):
        updown = 'upstream' if upstream else 'downstream'
        speeds = np.array([c.speed for c in cars]) * 3.6
        target_speeds = np.array([c.target_speed for c in cars]) * 3.6

        # spacings = np.array([c.lane_distance_to(c.fwd) for c in cars if c.fwd is not None])
        # mean_spacing = mean(spacings)

        lanes = self.upstream_lane_count if upstream else self.downstream_lane_count

        total_length_km = (lanes * 500) / 1000
        num_cars = len(cars)
        density = num_cars / total_length_km

        try:
            sms = hmean(speeds)
        except:
            sms = 0

        tms = mean(speeds)

        return {
            f'{updown} lanes':             lanes,
            f'{updown} num cars':          num_cars,
            f'{updown} density':           density,
            f'{updown} time mean speed':   tms,
            f'{updown} space mean speed':  sms,
            f'{updown} mean target speed': mean(target_speeds),
            # f'{updown} spacing m':              mean_spacing,
        }
