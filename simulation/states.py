import numpy as np
from gym import spaces

FREE_FLOW_SPEED = 100.
CAPACITY_PER_LANE = 70
ROAD_CAPACITY = 560

calc_mean_speed = lambda cars: np.mean(np.array([car.speed for car in cars]) * 3.6) if cars else 0


def get_imbalance(a, b):
    try:
        return a / (a + b)
    except ZeroDivisionError:
        return 0


class State(object):

    def __init__(self, env: "HighwayEnv"):
        self.env = env
        self.v_total = 0
        self.v_up = 0
        self.v_down = 0
        self.k_total = 0
        self.k_up = 0
        self.k_down = 0
        self.q_total = 0
        self.q_up = 0
        self.q_down = 0

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    @property
    def initial_state(self):
        return np.array([0.5, 0, 0, 0])

    @property
    def keys(self):
        return 'lane_ratio, speed_imbalance, density_imbalance, flow_imbalance'.split(', ')

    def observation(self, period):
        self.env.road.detectors.update(period)

        road = self.env.road
        target = self.env.FREE_FLOW_SPEED

        cars_up = road.upstream_cars
        cars_down = road.downstream_cars

        cars_up_highway = [c for c in cars_up if c.on_highway]
        cars_down_highway = [c for c in cars_down if c.on_highway]

        lane_ratio = road.upstream_lane_count / road.total_lane_count

        v_up = calc_mean_speed(cars_up_highway) / target
        v_down = calc_mean_speed(cars_down_highway) / target
        v_total = v_up + v_down
        v_imbalance = get_imbalance(v_up, v_down)

        k_up = len(cars_up_highway) / (road.upstream_lane_count * CAPACITY_PER_LANE)
        k_down = len(cars_down_highway) / (road.downstream_lane_count * CAPACITY_PER_LANE)
        k_total = k_up + k_down
        k_imbalance = get_imbalance(k_up, k_down)

        q_up = np.mean([d.flow for d in road.detectors.upstream_detectors.values()])
        q_down = np.mean([d.flow for d in road.detectors.downstream_detectors.values()])
        q_total = q_up + q_down
        q_imbalance = get_imbalance(q_up, q_down)

        self.v_total = v_total
        self.v_up = v_up
        self.v_down = v_down
        self.k_total = k_total
        self.k_up = k_up
        self.k_down = k_down
        self.q_total = q_total
        self.q_up = q_up
        self.q_down = q_down

        return np.array([lane_ratio, v_imbalance, k_imbalance, q_imbalance])
