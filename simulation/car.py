import numpy as np
import pygame
from numpy import deg2rad

from simulation import utils
from simulation.constants import *
from simulation.custom_types import Route, Vector
from simulation.idm import IDM
from simulation.lanes import AbstractLane
from simulation.mobil import Mobil

MIN_ANGLE = deg2rad(2)

PI_QUARTER = np.pi / 4

LOADED_IMAGE = None

FORCE_CHANGE_DISTANCE = 2.


class Car(pygame.sprite.Sprite):
    IMAGE_PATH = "/Users/rossgreen/PycharmProjects/simulation/img/car.png"

    LENGTH = CAR_LENGTH
    HALF_LENGTH = LENGTH / 2
    WIDTH = CAR_WIDTH
    MIN_SPEED = 0.0
    GAP_THRESHOLD = 1.0

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
        target_speed = np.random.uniform(0.9 * target_speed, target_speed)
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

        self.s_path = 0
        self.r_path = 0

        self.prev_heading = None
        self.prev_steering_angle = None

        self.take_exit_flag = False

        self.on_highway = self.lane.index[2] > 0

        self.travel_time = 0

        self.speed_deviations = []

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
        return cls(road=road,
                   lane=lane,
                   position=lane.position(longitudinal, 0),
                   heading=lane.heading_at(longitudinal),
                   speed=lane.speed_limit if speed is None else speed)

    @property
    def s_remaining_on_lane(self) -> float:
        return self.lane.length - self.s - self.HALF_LENGTH

    @property
    def s_remaining_on_path(self) -> float:
        return self.lane.path_length - self.s_path - self.HALF_LENGTH

    @property
    def reached_destination(self) -> bool:
        return self.lane.is_sink and self.lane.after_end(self.position, longitudinal=self.s)

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def destination(self) -> np.ndarray:
        if self.route is not None and len(self.route) >= 1:
            last_lane_index = self.route[-1]
            last_lane = self.road.network.get_lane(last_lane_index)
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
        self.travel_time += dt

        self.fwd, self.bwd = self.get_neighbours(lane=self.lane)
        self.priority_car = self.get_priority_car()
        self.yield_car = self.get_yield_car()

        # self._priority_car = None

        self.follow_road()
        self.change_lane_policy()
        self.steering = self.steering_control()

        self.acceleration = self.calc_acceleration()
        self.clip_speed()

        beta = np.arctan(1 / 2 * np.tan(self.steering))
        velocity = self.speed * np.array([np.cos(self.heading + beta), np.sin(self.heading + beta)])

        self.position += velocity * dt
        self.s, self.r = self.lane.local_coordinates(self.position)
        self.s_path, self.r_path = self.lane.path_coordinates(self.position)

        self.heading += ((self.speed * np.sin(beta)) / self.HALF_LENGTH) * dt
        self.speed = max(self.MIN_SPEED, self.speed + self.acceleration * dt)

        self.speed_deviations.append(self.lane.speed_limit - self.speed)

        if self.reached_destination:
            self.road.outflow += 1
            self.road.travel_time_log.append(self.travel_time)
            self.road.speed_deviations_log.append(np.mean(self.speed_deviations))

            self.kill()

        else:
            if not self.on_lane:
                self.on_new_lane_state_update()

    def get_neighbours(self, lane=None):
        lane = lane if lane is not None else self.lane
        return self.road.get_neighbours(car=self, lane=lane)

    def clip_speed(self):
        if self.speed > MAX_SPEED:  # Speeding
            self.acceleration = min(self.acceleration, MAX_SPEED - self.speed)

    def on_new_lane_state_update(self) -> None:
        self.lane = self.road.network.get_closest_lane(car=self)
        self.target_speed = self.lane.speed_limit
        self.right_bias = self.get_right_bias()
        self.s, self.r = self.lane.local_coordinates(self.position)
        self.s_path, self.r_path = self.lane.path_coordinates(self.position)
        self.on_highway = self.lane.index[2] > 0

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

        lane = lane or self.lane

        return lane.path_coordinates(car.position)[0] - lane.path_coordinates(self.position)[0]

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
        angle = np.clip(heading_rate * (self.HALF_LENGTH / v), -1, 1)

        steering_angle = np.arcsin(angle)

        steering_angle = min(steering_angle, MAX_STEERING_ANGLE)
        steering_angle = max(steering_angle, -MAX_STEERING_ANGLE)

        return steering_angle

    def get_acc_yield_for_lane(self, lane_index):
        if lane_index is None:
            return None

        priority_lane = self.road.network.get_lane(lane_index)

        fwd = self.road.get_fwd_car(self, priority_lane)

        if fwd is None:
            return None

        return IDM.calc_acceleration(self, fwd)

    def get_priority_car(self):
        priority_lane = None

        # Currently next to on ramp
        if self.lane.priority_lane_index is not None:
            priority_lane = self.road.network.get_lane(self.lane.priority_lane_index)

        else:
            # Lane to right needs to merge in due to lane reversal
            o, d, i = self.lane.index
            if i > 0 and i < 7:
                right_lane = self.road.network.get_lane((o, d, i + 1))
                if right_lane is not None and right_lane.forbidden:
                    priority_lane = right_lane

        if priority_lane is not None:
            priority_car = self.road.get_fwd_car(self, priority_lane)
            if self.lane_distance_to(priority_car, self.lane) <= self.lane_distance_to(self.fwd, self.lane):
                return priority_car

        return None

    def get_yield_car(self):
        yield_lane = None

        if self.lane.onramp_merge_to_lane_index is not None:
            yield_lane = self.road.network.get_lane(self.lane.onramp_merge_to_lane_index)

        else:
            o, d, i = self.lane.index
            if i > 1 and self.lane.forbidden and self.right_bias <= -1:
                yield_lane = self.road.network.get_lane((o, d, i - 1))

        if yield_lane is not None:
            yield_car = self.road.get_fwd_car(self, yield_lane)
        else:
            yield_car = None

        return yield_car

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

        if self.priority_car is not None:
            acc_priority = calc_acc(self, self.priority_car)
            acc = min(acc, acc_priority)

        # if self.yield_car is not None:
        #     acc_yield = calc_acc(self, self.yield_car)
        #     acc = min(acc, acc_yield)

        if self.lane.is_onramp and self.target_lane.index != self.lane.onramp_merge_to_lane_index:
            acc_end = IDM.calc_acceleration_to_end(self)
            acc = min(acc, acc_end)

        else:
            if self.yield_car is not None:
                acc_yield = calc_acc(self, self.yield_car)
                acc = min(acc, acc_yield)

        return acc

        # # Cars coming from onramp
        # acc_yield = self.get_acc_yield_for_lane(self.lane.priority_lane_index)
        # if acc_yield is not None:
        #     acc = min(acc, acc_yield)
        #
        # # On on ramp, slow down if car in front
        # acc_yield = self.get_acc_yield_for_lane(self.lane.onramp_merge_to_lane_index)
        # if acc_yield is not None:
        #     acc = min(acc, acc_yield)
        #
        # # From mandatory lane change
        # if self.priority_car is not None:
        #     acc_yield = IDM.calc_acceleration(self, self.priority_car)
        #     acc = min(acc, acc_yield)
        #
        # return min(acc, ACC_MAX)

    def should_abort_lane_change(self):
        if self.lane.forbidden:
            return False

        if self.target_lane.forbidden:
            return True

        if self.lane.same_road(self.target_lane) and self.lane.index[2] != self.target_lane.index[2]:
            cars = self.road.upstream_cars if self.upstream else self.road.downstream_cars
            for c in cars:
                if c is not self \
                        and c.lane != self.target_lane \
                        and (c.target_lane == self.target_lane or c.lane.onramp_merge_to_lane_index == self.target_lane) \
                        and 0 < self.lane_distance_to(c) < IDM.d_star(self, c):
                    return True
        return False

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

        if self.should_abort_lane_change():
            self.target_lane = self.lane
            return

        # if self.target_lane.index != self.lane.index:
        #     return

        # Only do lane changes at given frequency
        if not self.time_to_change_lanes():
            return

        if self.lane.is_next_to_offramp and not self.take_exit_flag:
            self.take_exit_flag = True
            if self.road.np_random.random() < 0.3:
                target_lane = self.road.network.get_lane(self.lane.offramp_lane_index)
                if target_lane is not None:
                    self.target_lane = target_lane
                    return

        potential_lanes = self.road.network.side_lanes(self.lane, self.position[0])

        for potential_next_lane in potential_lanes:

            if self.target_lane.index == potential_next_lane.index:
                continue

            if potential_next_lane is not None \
                    and potential_next_lane.is_reachable_from(self.position) \
                    and potential_next_lane.upstream == self.lane.upstream \
                    and not potential_next_lane.forbidden:

                fwd_new, bwd_new = self.get_neighbours(lane=potential_next_lane)
                if Mobil.should_change(lane=potential_next_lane,
                                       me=self,
                                       fwd_old=self.fwd,
                                       bwd_old=self.bwd,
                                       fwd_new=fwd_new,
                                       bwd_new=bwd_new,
                                       right_bias=self.right_bias,
                                       priority_car=self.priority_car):
                    self.target_lane = potential_next_lane

    def on_right_lane(self, car: "Car"):
        me_i = self.lane.index[2]
        other_i = car.lane.index[2]

        if other_i > me_i:  # right lane
            return 1
        elif other_i == me_i:  # same lane
            return 0
        else:  # left lane
            return -1

    def mandatory_merge_required(self, fwd: "Car", bwd: "Car"):
        if fwd is None:
            if bwd is not None and bwd is not self.priority_car:
                self.priority_car = None
            return

        if self.priority_car is not None:
            if (fwd is not None and fwd is self.priority_car) or (bwd is not None and bwd is self.priority_car):
                return

        bias = fwd.right_bias * -self.on_right_lane(fwd)

        if bias >= 1:  # and self.lane_distance_to(fwd) >= self.GAP_THRESHOLD:
            self.priority_car = fwd

    def get_right_bias(self) -> float:
        o, d, i = self.lane.index

        if self.lane.is_onramp:
            if self.s_remaining_on_path <= 5:
                return 100
            else:
                return 1.
        elif self.lane.forbidden:
            return -1.
        elif self.lane.priority_lane_index is not None:
            return 0.4
        elif i == 1:
            return 0.3
        else:
            return 0

    def plan_route_to(self, destination: str) -> "Car":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network

        example: plan_route_to('east')
        """

        if self.lane.index[2] == destination:
            self.route = [self.lane.index]
            return self

        try:
            path = self.road.network.shortest_path(self.lane.index[2], destination)
        except KeyError:
            path = []

        self.route = [self.lane.index]

        for i in range(len(path) - 1):
            origin = path[i]
            destination = path[i + 1]
            self.route.append((origin, destination, None))

            if self.lane.upstream and destination in self.road.network.upstream_sink_destinations:
                break

            if not self.lane.upstream and destination in self.road.network.downstream_sink_destinations:
                break

        return self

    def to_dict(self):
        x, y = self.position
        return dict(
                id=id(self) % 1000,
                upstream=self.lane.upstream,
                lane=self.lane.path_index,
                x=x,
                y=y,
                s=self.s,
                s_path=self.s_path,
                speed=self.speed,
                acc=self.acceleration,
                target_speed=self.target_speed,
                headway=self.lane_distance_to(self.fwd)
        )

    def __str__(self):
        x, y = self.position
        return ", ".join([
            f'Car #{id(self) % 1000}: at [{round(x, 1)}, {round(y, 1)}], s: {round(self.s)}',
            f'v: {round(self.speed, 1)}, v0: {round(self.target_speed, 1)}, a: {round(self.acceleration, 1)}',
            f'{self.lane.index}'
        ])

    def __repr__(self):
        return self.__str__()
