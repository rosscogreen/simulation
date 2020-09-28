import logging
from functools import lru_cache
from typing import Optional
from collections import defaultdict

from gym.utils import seeding

from simulation.car import Car
from simulation.constants import MIN_INSERTION_GAP
from simulation.custom_types import LaneIndex, CL, SL
from simulation.graphics.car import Cars
from simulation.idm import IDM
from simulation.lanes import AbstractLane
from simulation.metrics import RoadMetrics
from simulation.network import RoadNetwork
from simulation.road_factory.constants import WEST_NODE, EAST_NODE
from simulation.road_objects import Obstacle
from simulation.detector import Detector

logger = logging.getLogger(__name__)

VERBOSE = 0


class Road(object):

    def __init__(self, np_random=None):
        self.network: "RoadNetwork" = RoadNetwork()
        self.cars = Cars()
        self.obstacles = Cars()
        self.road_objects = Cars()
        self.metrics = RoadMetrics()
        self.np_random = np_random or seeding.np_random()[0]
        self.num_lanes = 8
        self.num_upstream = 4
        self.upstream_wait = 0
        self.downstream_wait = 0
        self.redraw = True
        self.is_lane_reversal_in_progress: bool = False
        self.new_active_lane: Optional["AbstractLane"] = None
        self.new_forbidden_lane: Optional["AbstractLane"] = None
        self.detectors = defaultdict(dict)
        self.detector_bands = set()
        self.steps = 0
        self.time = 0

    def add_detector(self, upstream: str, x: int, agg_period):
        detector = Detector(x=x, agg_period=agg_period)
        self.detectors[upstream][x] = detector

    def step(self, dt: float):
        self.steps += 1
        self.time += dt
        self.cars.update(dt)
        self.lane_reversal_step()
        self.update_detectors_step()

    def update_detectors_step(self):
        for c in self.cars:
            if int(c.x) in self.detectors[c.upstream]:
                self.detectors[c.upstream][int(c.x)].add_car(c)

    def update_detectors_period(self):
        nup = self.num_upstream
        ndown = self.num_lanes - nup
        obs = {}
        for upstream, detectors in self.detectors.items():
            for x, d in detectors.items():
                if upstream == 'upstream':
                    obs[upstream] = d.add_step(nup)
                else:
                    obs[upstream] = d.add_step(ndown)
        return obs


    def initiate_lane_reversal(self, upstream: bool):
        self.is_lane_reversal_in_progress = True
        n = self.num_lanes
        i = self.num_upstream
        get_lane = self.network.get_lane

        if upstream:
            self.new_active_lane = get_lane(LaneIndex(WEST_NODE, EAST_NODE, i + 1))
            self.new_forbidden_lane = get_lane(LaneIndex(EAST_NODE, WEST_NODE, n - i))
        else:
            self.new_active_lane = get_lane(LaneIndex(EAST_NODE, WEST_NODE, n - i + 1))
            self.new_forbidden_lane = get_lane(LaneIndex(WEST_NODE, EAST_NODE, i))

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
            o, d, i = self.new_active_lane.lane_index
            self.new_active_lane.left_line = SL
            self.network.get_lane(LaneIndex(o, d, i + 1)).left_line = CL
        else:
            # new_forbidden is upstream
            o, d, i = self.new_forbidden_lane.lane_index
            self.new_forbidden_lane.left_line = CL
            self.network.get_lane(LaneIndex(o, d, i + 1)).left_line = SL

        self.redraw = True
        self.new_active_lane.forbidden = False
        self.new_active_lane = None
        self.new_forbidden_lane = None
        self.is_lane_reversal_in_progress = False

        self.num_upstream = sum([not lane.forbidden for lane in self.network.get_lanes(WEST_NODE, EAST_NODE)]) + 1

    def is_lane_empty(self, lane: "AbstractLane"):
        for car in self.cars:
            if lane.same_lane(car.lane):
                return False
        return True

    def spawn_car_on_lane(self, lane: "AbstractLane", destination=None) -> bool:
        fwd = self.first_car_on_lane(lane, merging=True)

        speed = lane.speed_limit

        if fwd and not isinstance(fwd, Obstacle):
            gap = lane.local_coordinates(fwd.position)[0]
            if gap < MIN_INSERTION_GAP:
                return False
            speed = IDM.calc_max_initial_speed(gap=gap, fwd_speed=fwd.speed, speed_limit=speed)

        car = Car.make_on_lane(road=self, lane=lane, speed=speed)

        if destination:
            car.plan_route_to(destination)

        return True

    def cars_on_lane(self, lane, merging=False):
        return [c for c in self.cars
                if self.lanes_connected(lane, c.lane)
                and (not merging or self.lanes_connected(lane, c.lane))]

    def first_car_on_lane(self, lane, merging=False) -> Optional["Car"]:
        cars = [(c, c.s if lane.same_lane(c.lane) else lane.s(c.position))
                for c in self.cars
                if self.lanes_connected(lane, c.lane)
                and (not merging or self.lanes_connected(lane, c.lane))]

        s_fwd = 500
        fwd = None

        for c, s in cars:
            if s <= s_fwd:
                s_fwd = s
                fwd = c

        return fwd

    @lru_cache()
    def lanes_connected(self, lane1, lane2):
        if lane1.upstream != lane2.upstream:
            return False

        connected = self.network.is_connected

        l1, l2 = lane1.lane_index, lane2.lane_index
        return connected(l1, l2, same_lane=True, depth=2) or connected(l2, l1, same_lane=True, depth=2)

    def get_neighbours(self, car, lane=None, include_obstacles=False):
        lane = lane or car.lane
        s_me = lane.s(car.position)

        cars = self.road_objects if include_obstacles else self.cars
        cars = [(c, c.s if lane.same_lane(c.lane) else lane.s(c.position))
                for c in cars if self.lanes_connected(lane, c.lane) and c is not car]

        s_fwd = 500
        s_bwd = 0
        fwd = None
        bwd = None

        for c, s in cars:
            if s_me < s <= s_fwd:  # Car in front
                s_fwd = s
                fwd = c
            elif s_bwd < s < s_me:
                s_bwd = s
                bwd = c

        return fwd, bwd

    def get_fwd_car(self, car, lane=None, include_obstacles=False):
        lane = lane or car.lane
        s_me = lane.s(car.position)

        cars = self.road_objects if include_obstacles else self.cars
        cars = [(c, c.s if lane.same_lane(c.lane) else lane.s(c.position))
                for c in cars if self.lanes_connected(lane, c.lane) and c is not car]

        s_fwd = 500
        fwd = None

        for c, s in cars:
            if s_me < s <= s_fwd:  # Car in front
                s_fwd = s
                fwd = c

        return fwd
