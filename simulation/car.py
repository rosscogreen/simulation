from typing import Tuple, List, Optional

import numpy as np
import pygame
from numpy import deg2rad

from simulation import utils
from simulation.constants import *
from simulation.custom_types import Route, Vector, LaneIndex
from simulation.idm import IDM
from simulation.lanes import AbstractLane
from simulation.mobil import Mobil
from simulation.road_objects import Obstacle

MIN_ANGLE = deg2rad(2)

PI_QUARTER = np.pi / 4

LOADED_IMAGE = None

FORCE_CHANGE_DISTANCE = 10.



class Car(pygame.sprite.Sprite):
    # IMAGE_PATH = Path('..') / 'img' / 'car.png'
    IMAGE_PATH = "/Users/rossgreen/PycharmProjects/simulation/img/car.png"

    LENGTH = CAR_LENGTH
    HALF_LENGTH = LENGTH / 2
    WIDTH = CAR_WIDTH
    MIN_SPEED = 2.0
    GAP_THRESHOLD = 0

    def __init__(
            self,
            road: "Road",
            lane: "AbstractLane",
            position: Vector,
            heading: float = 0,
            speed: float = 0,
            target_lane: "AbstractLane" = None,
            target_speed: float = None,
            route: Route = None,
            timer: float = None
    ):
        super(Car, self).__init__()
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.speed = speed
        self.lane = lane
        self.upstream = 'upstream' if self.lane.upstream else 'downstream'
        self.target_lane = target_lane if target_lane else lane
        target_speed = target_speed or self.lane.speed_limit
        target_speed = np.random.uniform(0.9*target_speed, target_speed)
        self.target_speed = max(target_speed, 0)
        self.steering: float = 0.
        self.acceleration: float = 0.
        self.route = route
        self.timer = timer or (np.sum(self.position) * np.pi) % LANE_CHANGE_DELAY
        self.right_bias = 0

        self.fwd = None
        self.bwd = None
        self.priority_car = None

        self.s = 0
        self.r = 0

        self.image = None

    @classmethod
    def make_on_lane(cls,
                     road: "Road",
                     lane: "AbstractLane",
                     longitudinal: float = 0,
                     speed: float = None) -> "Car":
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param speed: initial speed in [m/s]
        :return: A vehicle with at the specified position
        """
        if speed is None:
            speed = lane.speed_limit

        speed = np.random.uniform(speed*0.8, speed)

        car = cls(road=road,
                  lane=lane,
                  position=lane.position(longitudinal, 0),
                  heading=lane.heading_at(longitudinal),
                  speed=speed)

        road.cars.add(car)

        road.road_objects.add(car)

        return car

    @property
    def s_remaining_on_lane(self) -> float:
        return self.lane.length - self.s

    @property
    def should_kill(self) -> bool:
        return self.lane.is_sink and self.lane.after_end(self.position, longitudinal=self.s)

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def destination(self) -> np.ndarray:
        if self.route is not None and len(self.route) >= 1:
            last_lane = self.route[-1]
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def on_lane(self) -> bool:
        """ Is the vehicle on its current lane, or off-road ? """
        return self.lane.on_lane(self.position, longitudinal=self.s, lateral=self.r)

    @property
    def next_lane(self):
        return self.road.network.get_next_lane(self.target_lane, self.route, self.position)

    def update(self, dt: float) -> None:
        """
        :param dt: timestep of integration of the model [s]
        """
        self.timer += dt

        self.fwd, self.bwd = self.get_neighbours(include_obstacles=False)
        # self._priority_car = None

        self.follow_road()
        self.change_lane_policy()
        self.steering = self.steering_control()

        self.acceleration = self.calc_acceleration()
        self.clip_speed()

        beta = np.arctan(1 / 2 * np.tan(self.steering))
        velocity = self.speed * np.array([np.cos(self.heading + beta), np.sin(self.heading + beta)])

        self.position += velocity * dt
        self.x = self.position[0]
        self.s, self.r = self.lane.local_coordinates(self.position)
        self.heading += ((self.speed * np.sin(beta)) / self.HALF_LENGTH) * dt
        self.speed = max(0.01, self.speed + self.acceleration * dt)

        if not self.on_lane:
            self.on_new_lane_state_update()

    def get_neighbours(self, lane=None, include_obstacles=False):
        lane = lane or self.lane
        return self.road.get_neighbours(car=self, lane=lane, include_obstacles=include_obstacles)

    def clip_speed(self):
        if self.speed > MAX_SPEED:  # Speeding
            self.acceleration = min(self.acceleration, MAX_SPEED - self.speed)
        elif self.speed < self.MIN_SPEED:  # Going to slow
            self.acceleration = max(self.acceleration, 1.0 * (self.MIN_SPEED - self.speed))

    def on_new_lane_state_update(self) -> None:
        self.lane = self.road.network.get_closest_lane(car=self)
        self.target_speed = self.lane.speed_limit
        self.right_bias = self.get_right_bias()
        self.s, self.r = self.lane.local_coordinates(self.position)

    def set_mandatory_merge(self, to_right: bool = True):
        self.right_bias = 1 if to_right else -1

    def set_force_merge(self, to_right: bool = True):
        self.right_bias = 100 * (1 if to_right else -1)

    def lane_distance_to(self, car: "Car", lane: "AbstractLane" = None) -> float:
        """
        Compute the signed distance to another vehicle along a lane.
        Longitudinal distance from center to center

        :param car: the other vehicle
        :param lane: a lane
        :return: the distance to the other vehicle [m]
        """
        if not car:
            return np.nan
        if lane is None:
            return self.lane.s(car.position) - self.s
        else:
            return lane.s(car.position) - lane.s(self.position)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.target_lane.after_end(self.position):
            self.target_lane = self.next_lane

    def steering_control(self) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane: lane to follow
        :return: a steering wheel angle command [rad]
        """
        v = utils.not_zero(self.speed)
        heading = self.heading

        target_lane = self.target_lane
        s1, r1 = target_lane.local_coordinates(self.position)
        s2 = s1 + (v * PURSUIT_TAU)

        v_lat = (-KP_LATERAL * r1) / v
        v_lat = np.clip(v_lat, -1, 1)

        heading_long = target_lane.heading_at(s2)
        heading_lat = np.clip(np.arcsin(v_lat), -PI_QUARTER, PI_QUARTER)

        heading_ref = heading_long + heading_lat - heading

        # Heading control
        heading_rate = KP_HEADING * utils.wrap_to_pi(heading_ref)

        # Heading rate to steering angle
        angle = heading_rate * (self.HALF_LENGTH / v)
        angle = np.clip(angle, -1, 1)

        steering_angle = np.arcsin(angle)

        steering_angle = float(np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))

        return steering_angle

    def calc_acceleration(self) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param fwd: the vehicle preceding the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        calc_acc = IDM.calc_acceleration

        # To car in in front on same lane
        acc = calc_acc(self, self.fwd)

        # Cars coming from onramp
        if self.lane.priority_lane_index is not None:
            priority_lane = self.road.network.get_lane(self.lane.priority_lane_index)
            fwd = self.road.get_fwd_car(self, priority_lane)
            if fwd is not None:
                acc_yield = IDM.calc_acceleration(self, fwd)
                acc = min(acc, acc_yield)

        # On on ramp, slow down if car in front
        if self.lane.onramp_merge_to_lane_index is not None:
            priority_lane = self.road.network.get_lane(self.lane.onramp_merge_to_lane_index)
            fwd = self.road.get_fwd_car(self, priority_lane)
            if fwd is not None:# and fwd.speed > 3:
                acc_yield = IDM.calc_acceleration(self, fwd)
                acc = min(acc, acc_yield)

        # From mandatory lane change
        if self.priority_car is not None:
            acc_yield = IDM.calc_acceleration(self, self.priority_car)
            acc = min(acc, acc_yield)

        return acc

        #return float(np.clip(acc, -ACC_MAX, ACC_MAX))

    @property
    def should_abort_lane_change(self):
        if self.lane.forbidden:
            return False

        if self.target_lane.forbidden:
            return True

        if self.lane.same_road(self.target_lane) and self.lane.index != self.target_lane.index:
            for c in self.road.cars:
                if c is not self \
                        and not isinstance(c, Obstacle) \
                        and c.lane != self.target_lane \
                        and c.target_lane == self.target_lane \
                        and 0 < self.lane_distance_to(c) < IDM.d_star(self, c):
                    return True
        return False

    @property
    def time_to_change_lanes(self):
        if self.lane.is_onramp or self.lane.forbidden or self.lane.is_next_to_offramp or self.timer >= LANE_CHANGE_DELAY:
            self.timer = 0
            return True
        else:
            return False

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """

        # Force Change
        if self.lane.is_onramp and self.s_remaining_on_lane < FORCE_CHANGE_DISTANCE:
            self.target_lane = self.road.network.get_lane(self.lane.onramp_merge_to_lane_index)
            return

        if self.should_abort_lane_change:
            self.target_lane = self.lane
            return

        # Only do lane changes at given frequency
        if not self.time_to_change_lanes:
            return

        for potential_next_lane in self.road.network.side_lanes(self.lane, self.position[0]):
            if potential_next_lane and potential_next_lane.is_reachable_from(self.position):
                if potential_next_lane.upstream == self.lane.upstream:

                    fwd_new, bwd_new = self.road.get_neighbours(car=self, lane=potential_next_lane)

                    self.mandatory_merge_required(fwd_new, bwd_new)

                    if not potential_next_lane.forbidden:
                        if Mobil.should_change(lane=potential_next_lane,
                                           me=self,
                                           fwd_old=self.fwd,
                                           bwd_old=self.bwd,
                                           fwd_new=fwd_new,
                                           bwd_new=bwd_new,
                                           right_bias=self.right_bias):
                            self.target_lane = potential_next_lane

    def mandatory_merge_required(self, fwd: "Car", bwd: "Car"):

        if fwd is None:
            if bwd is not None and bwd is not self.priority_car:
                self.priority_car = None
            return

        if self.priority_car is not None:
            if (fwd is not None and fwd is self.priority_car) or (bwd is not None and bwd is self.priority_car):
                return

        is_on_left = 1 if fwd.lane.index < self.lane.index else -1
        bias = fwd.right_bias * is_on_left

        if bias >= 1 and self.lane_distance_to(fwd) >= self.GAP_THRESHOLD:
            self.priority_car = fwd

    def get_right_bias(self) -> float:
        if self.lane.is_onramp:
            return 1.
        elif self.lane.forbidden:
            return -1.
        elif self.lane.is_next_to_offramp is not None:
            return -np.random.uniform(0, 0.2)
        elif self.lane.index == 1:
            return 0.2
        else:
            return 0

    def plan_route_to(self, destination: str) -> "Car":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network

        example: plan_route_to('east')
        """

        if self.lane.destination == destination:
            self.route = [self.lane.lane_index]
            return self

        try:
            path = self.road.network.shortest_path(self.lane.destination, destination)
        except KeyError:
            path = []

        self.route = [self.lane.lane_index]

        for i in range(len(path) - 1):
            origin = path[i]
            destination = path[i + 1]
            lane_index = LaneIndex(origin=origin, destination=destination, index=None)
            self.route.append(lane_index)

            if self.lane.upstream and destination in self.road.network.upstream_sink_destinations:
                break

            if not self.lane.upstream and destination in self.road.network.downstream_sink_destinations:
                break

        return self

    def __str__(self):
        x, y = self.position
        o, d, i = self.lane.lane_index
        return ", ".join([
            f'[{i}] Car #{id(self) % 1000}: at [{round(x, 1)}, {round(y, 1)}], s: {round(self.s)}',
            f'v: {round(self.speed, 1)}, v0: {round(self.target_speed, 1)}, a: {round(self.acceleration, 1)}',
            f'{o} -> {d}'
        ])

    def __repr__(self):
        return self.__str__()
