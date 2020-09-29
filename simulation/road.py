import logging
from functools import lru_cache
from typing import Optional

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
        self.is_lane_reversal_in_progress = True
        n = self.total_lane_count
        i = self.upstream_lane_count
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
            self.new_forbidden_lane.left_line = CL
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
        for _ in range(n):
            lane = self.np_random.choice(active_lanes)
            self.spawn_car_on_lane(lane)

    def spawn_car_on_lane(self, lane: "AbstractLane", destination=None) -> bool:
        fwd = self.first_car_on_lane(lane, merging=True)

        speed = lane.speed_limit

        if fwd:
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

        l1, l2 = lane1.index, lane2.index
        return connected(l1, l2, same_lane=True, depth=2) or connected(l2, l1, same_lane=True, depth=2)

    def get_neighbours(self, car, lane=None):
        lane = lane or car.lane
        s_me = lane.s(car.position)

        cars = [(c, c.s if lane.same_lane(c.lane) else lane.s(c.position))
                for c in self.cars
                if self.lanes_connected(lane, c.lane)
                and c is not car]

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

    def get_fwd_car(self, car, lane=None):
        lane = lane or car.lane
        s_me = lane.s(car.position)

        cars = [(c, c.s if lane.same_lane(c.lane) else lane.s(c.position))
                for c in self.cars
                if self.lanes_connected(lane, c.lane)
                and c is not car]

        s_fwd = 500
        fwd = None

        for c, s in cars:
            if s_me < s <= s_fwd:  # Car in front
                s_fwd = s
                fwd = c

        return fwd
