from __future__ import division, print_function

import importlib

import numpy as np

from simulation.custom_types import LaneIndex

EPSILON = 0.01

import json

def format_metric_dictionary(d):
    s = []
    for k, v in d.items():
        s.append(f'{k} = {v}')
    return s


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def not_zero(x):
    if abs(x) > EPSILON:
        return x
    elif x > 0:
        return EPSILON
    else:
        return -EPSILON


def wrap_to_pi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_in_rectangle(point, rect_min, rect_max):
    """
        Check if a point is inside a rectangle
    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point, center, length, width, angle):
    """
        Check if a point is inside a rotated rectangle
    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, [-length / 2, -width / 2], [length / 2, width / 2])


def point_in_ellipse(point, center, angle, length, width):
    """
        Check if a point is inside an ellipse
    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1, rect2):
    """
        Do two rotated rectangles intersect?
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def has_corner_inside(rect1, rect2):
    """
        Check if rect1 has a corner inside rect2
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l1v = np.array([l1 / 2, 0])
    w1v = np.array([0, w1 / 2])
    r1_points = np.array([[0, 0],
                          - l1v, l1v, -w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1 + np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def do_every(duration, timer):
    return duration < timer


def do_on_step(step_number, step_interval):
    return step_number % step_interval == 0


def remap(v, x, y):
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def class_from_path(path):
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def get_active_lanes(lanes):
    return [lane for lane in lanes if not lane.forbidden]


def first(iterable, condition=lambda x: True):
    """
    Returns the first item in the `iterable` that
    satisfies the `condition`.

    If the condition is not given, returns the first item of
    the iterable.

    Raises `StopIteration` if no item satysfing the condition is found.

    >>> first( (1,2,3), condition=lambda x: x % 2 == 0)
    2
    >>> first(range(3, 100))
    3
    >>> first( () )
    Traceback (most recent call last):
    ...
    StopIteration
    """

    return next(x for x in iterable if condition(x))


def first_forbidden_lane(lanes):
    for lane in lanes:
        if lane.forbidden:
            return lane


def is_same_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
    """
    Is lane 1 in the same road as lane 2?

    >>> is_same_road(LaneIndex('a','b',0), LaneIndex('a','b',0), same_lane=True)
    True
    >>> is_same_road(LaneIndex('a','b',0), LaneIndex('a','b',1), same_lane=True)
    False
    >>> is_same_road(LaneIndex('a','b',0), LaneIndex('a','b',1), same_lane=False)
    True
    """
    return lane_index_1[:2] == lane_index_2[:2] \
           and (not same_lane or lane_index_1.index == lane_index_2.index)


def is_leading_to_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
    """
    Is lane 1 leading to of lane 2?
    >>> is_leading_to_road(LaneIndex('a','b',0), LaneIndex('b','c',1), same_lane=False)
    True
    >>> is_leading_to_road(LaneIndex('a','b',0), LaneIndex('b','c',1), same_lane=True)
    False
    >>> is_leading_to_road(LaneIndex('a','b',0), LaneIndex('b','c',0), same_lane=True)
    True
    """
    return lane_index_1.destination == lane_index_2.origin \
           and (not same_lane or lane_index_1.index == lane_index_2.index)


def save_graph(road):
    graph = {}
    for o in road.network.graph:
        graph[o] = {}
        for d in road.network.graph[o]:
            graph[o][d] = {}
            for i, lane in road.network.graph[o][d].items():
                graph[o][d][i] = str(lane)

    import json
    path = '/Users/rossgreen/PycharmProjects/simulation/resources/graph.json'

    with open(path, 'w') as f:
        json.dump(graph, f)


def is_conflict_possible(car1: "Car", car2: "Car", horizon: int = 3, step: float = 0.25) -> bool:
    times = np.arange(step, horizon, step)
    positions_1, headings_1 = car1.predict_trajectory_constant_speed(times)
    positions_2, headings_2 = car2.predict_trajectory_constant_speed(times)

    for position_1, heading_1, position_2, heading_2 in zip(positions_1, headings_1, positions_2, headings_2):
        # Fast spherical pre-check
        if np.linalg.norm(position_2 - position_1) > car1.LENGTH:
            continue

        # Accurate rectangular check
        if rotated_rectangles_intersect((position_1, 1.5 * car1.LENGTH, 0.9 * car1.WIDTH, heading_1),
                                        (position_2, 1.5 * car2.LENGTH, 0.9 * car2.WIDTH, heading_2)):
            return True


def respect_priorities(car1: "Car", car2: "Car") -> "Car":
    """
    Resolve a conflict between two vehicles by determining who should yield

    :param car1: first vehicle
    :param car2: second vehicle
    :return: the yielding vehicle
    """
    if car1.lane.is_onramp:
        return car2
    if car1.lane.priority > car2.lane.priority:
        return car2
    elif car1.lane.priority < car2.lane.priority:
        return car1
    else:  # The vehicle behind should yield
        return car1 if car1.lane_distance_to(car2) > car2.lane_distance_to(car1) else car2

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
