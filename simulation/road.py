import logging
from collections import defaultdict
from typing import Optional
import numpy as np
from gym.utils import seeding

from simulation import utils
from simulation.car import Car
from simulation.config import WEST_NODE, EAST_NODE
from simulation.constants import CAR_LENGTH, MIN_INSERTION_GAP
from simulation.custom_types import CL, SL
from simulation.detector import Detectors
from simulation.idm import IDM
from simulation.lanes import AbstractLane
from simulation.network import RoadNetwork
from simulation.rendering import Cars

logger = logging.getLogger(__name__)

VERBOSE = 0

WAIT_LIMIT = 30


class Road(object):
    east = 'east'
    west = 'west'

    CAR_HALF_LENGTH = CAR_LENGTH / 2

    def __init__(self, np_random=None):
        self.network: "RoadNetwork" = RoadNetwork()

        self.cars = Cars()
        self.upstream_cars = Cars()
        self.downstream_cars = Cars()

        self.queues = {'upstream': defaultdict(int), 'downstream': defaultdict(int)}

        self.np_random = np_random or seeding.np_random()[0]

        # Lane reversal State
        self.is_lane_reversal_in_progress: bool = False
        self.new_active_lane: Optional["AbstractLane"] = None
        self.new_forbidden_lane: Optional["AbstractLane"] = None

        self.detectors = Detectors()

        self.total_lane_count = 8
        self.upstream_lane_count = 4
        self.downstream_lane_count = 4

        self.steps_in_current_reversal = 0
        self.total_steps_waiting_for_reversal = 0

        self.outflow = 0
        self.travel_time_log = []
        self.speed_deviations_log = []

        # only redraw on changes to lane structure
        self.redraw = True

    @property
    def average_travel_time(self):
        try:
            return np.mean(self.travel_time_log)
        except:
            return 0

    @property
    def average_deviation_from_free_flow_speed(self):
        try:
            return np.mean(self.speed_deviations_log)
        except:
            return 0

    def step(self, dt: float):
        self.cars.update(dt)
        self.inflow()

        for car in self.cars:
            self.detectors.update_car(car)

        if self.is_lane_reversal_in_progress:
            self.lane_reversal_step()

    def initiate_lane_reversal(self, upstream: bool):
        """ Starts a lane reversal in the given direction """

        i = self.upstream_lane_count

        if (upstream and i == 7) or (not upstream and i == 1):
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

        if self.steps_in_current_reversal >= WAIT_LIMIT:
            self.new_forbidden_lane.forbidden = False
            self.reset_lane_reversal()
            return

        # Are we still waiting for cars to vacate the lane
        if not self.is_lane_empty(self.new_forbidden_lane):
            self.steps_in_current_reversal += 1
            self.total_steps_waiting_for_reversal += 1
            return

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
        # allow cars to enter on new lane
        self.new_active_lane.forbidden = False
        self.reset_lane_reversal()

        # Recount lanes in each direction
        self.upstream_lane_count = sum(
                [not lane.forbidden for lane in self.network.get_lanes(WEST_NODE, EAST_NODE)]) + 1
        self.downstream_lane_count = self.total_lane_count - self.upstream_lane_count

    def reset_lane_reversal(self):
        self.is_lane_reversal_in_progress = False
        self.new_active_lane = None
        self.new_forbidden_lane = None
        self.steps_in_current_reversal = 0

    def is_lane_empty(self, lane: "AbstractLane"):
        cars = self.upstream_cars if lane.upstream else self.downstream_cars
        for car in cars:
            if lane.same_lane(car.lane):
                return False
        return True

    def gap_to_first_car(self, lane):
        first_car = self.first_car_on_lane(lane, merging=False)
        if first_car is None:
            return lane.path_length
        return lane.s_path(first_car.position)

    def spawn(self, num_cars_to_spawn: int, upstream: bool):
        source_lanes = self.network.get_source_lanes_for_direction(upstream)
        self.np_random.shuffle(source_lanes)

        for i in range(num_cars_to_spawn):
            lane = source_lanes[i % len(source_lanes)]
            self.add_car(lane)

    def add_car(self, lane: "AbstractLane"):
        direction = 'upstream' if lane.upstream else 'downstream'
        if self.queues[direction][lane.index] > 0:
            self.queues[direction][lane.index] += 1
        else:
            fwd = self.first_car_on_lane(lane)
            speed = lane.speed_limit

            if fwd:
                gap = fwd.s_path
                if gap < MIN_INSERTION_GAP:
                    self.queues[direction][lane.index] += 1
                    return
                else:
                    speed = IDM.calc_max_initial_speed(gap, fwd.speed, lane.speed_limit)

            car = Car.make_on_lane(road=self, lane=lane, speed=speed)

            self.cars.add(car)

            if lane.upstream:
                self.upstream_cars.add(car)
            else:
                self.downstream_cars.add(car)

    def inflow(self):
        for direction, lane_indexes in self.queues.items():
            for lane_index, queue_count in lane_indexes.items():
                if queue_count > 0:
                    lane = self.network.get_lane(lane_index)
                    fwd = self.first_car_on_lane(lane)
                    speed = lane.speed_limit

                    if fwd is not None:
                        gap = fwd.s_path
                        if gap < MIN_INSERTION_GAP:
                            continue
                        speed = IDM.calc_max_initial_speed(gap, fwd.speed, lane.speed_limit)

                    car = Car.make_on_lane(road=self, lane=lane, speed=speed)
                    self.cars.add(car)
                    if lane.upstream:
                        self.upstream_cars.add(car)
                    else:
                        self.downstream_cars.add(car)

                    self.queues[direction][lane_index] -= 1

    def cars_on_lane(self, lane, merging=False):
        cars = self.upstream_cars if lane.upstream else self.downstream_cars
        if merging:
            return [c for c in cars if lane.same_lane(c.lane) or lane.same_lane(c.target_lane)]
        else:
            return [c for c in cars if lane.same_lane(c.lane)]

    def first_car_on_lane(self, lane, merging=False) -> Optional["Car"]:
        cars = self.cars_on_lane(lane, merging)
        if not cars:
            return None
        return min(cars, key=lambda c: c.s_path)

    def get_neighbours(self, car, lane=None):
        lane = lane if lane is not None else car.lane
        s_me = lane.s_path(car.position)
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
        lane = lane if lane is not None else car.lane
        s_me = lane.s_path(car.position)
        cars = self.cars_on_lane(lane)
        try:
            return utils.first(cars, lambda car: car.s_path >= s_me)
        except StopIteration:
            return None

    def calc_total_queue_length(self):
        total_queue_length = 0
        for direction, lane_indexes in self.queues.items():
            for lane_index, queue_length in lane_indexes.items():
                total_queue_length += queue_length

        return total_queue_length
