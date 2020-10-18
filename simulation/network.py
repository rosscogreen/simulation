import logging
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Iterable, Tuple, DefaultDict, Dict, Set

import numpy as np

from simulation.car import Car
from simulation.custom_types import Route, Path, LaneIndex
from simulation.lanes import AbstractLane
from simulation.config import WEST_NODE, EAST_NODE
import random

logger = logging.getLogger(__name__)


class RoadNetwork(object):
    graph: DefaultDict[str, DefaultDict[str, Dict[int, AbstractLane]]]

    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(dict))

        self.upstream_source_lanes: List[AbstractLane] = []
        self.downstream_source_lanes: List[AbstractLane] = []

        self.upstream_sink_lanes: List[AbstractLane] = []
        self.downstream_sink_lanes: List[AbstractLane] = []

        self.upstream_sink_destinations: Set[str] = set()
        self.downstream_sink_destinations: Set[str] = set()

        self.upstream_source_origins: Set[str] = set()
        self.downstream_source_origins: Set[str] = set()

        self.upstream_lane_list = []
        self.downstream_lane_list = []

    def add_lane(self, lane: "AbstractLane"):
        o, d, i = lane.index
        self.graph[o][d][i] = lane

        if lane.upstream:
            self.upstream_lane_list.append(lane)
            if lane.is_source:
                self.upstream_source_lanes.append(lane)
                self.upstream_source_origins.add(o)
            elif lane.is_sink:
                self.upstream_sink_lanes.append(lane)
                self.upstream_sink_destinations.add(d)
        else:
            self.downstream_lane_list.append(lane)
            if lane.is_source:
                self.downstream_source_lanes.append(lane)
                self.downstream_source_origins.add(o)
            elif lane.is_sink:
                self.downstream_sink_lanes.append(lane)
                self.downstream_sink_destinations.add(d)

    def add_lanes(self, lanes: List[AbstractLane]) -> None:
        for lane in lanes:
            self.add_lane(lane)

    def get_lanes(self, origin: str, destination: str) -> List[AbstractLane]:
        """ Return all lanes from origin -> destination """
        return list(self.graph[origin][destination].values())

    def get_active_lanes(self, origin, destination) -> List[AbstractLane]:
        return [l for l in self.get_lanes(origin, destination) if not l.forbidden]

    def get_forbidden_lanes(self, origin, destination) -> List[AbstractLane]:
        return [l for l in self.get_lanes(origin, destination) if l.forbidden]

    def get_source_lanes_for_direction(self, upstream, shuffle=False):
        all_source_lanes = self.upstream_source_lanes if upstream else self.downstream_source_lanes
        active_source_lanes = [lane for lane in all_source_lanes if not lane.forbidden]
        if shuffle:
            np.random.shuffle(active_source_lanes)
        return active_source_lanes

    @lru_cache()
    def get_lane(self, index: LaneIndex) -> Optional[AbstractLane]:
        """
            Get the lane geometry corresponding to a given index in the road_network network.
        :return: the corresponding lane geometry.
        """
        o, d, i = index

        try:
            return self.graph[o][d][i]
        except KeyError:
            if i is None and len(self.graph[o][d]) == 1:
                return list(self.graph[o][d].values())[0]

    def get_closest_lane(self, car: Car) -> AbstractLane:
        """ Returns the lane that is closest to the given cars current position """
        pos = car.position
        lanes = self.upstream_lane_list if car.lane.upstream else self.downstream_lane_list
        return min(lanes, key=lambda l: l.distance(pos))

    @lru_cache()
    def get_lanes_for_index(self, upstream: bool, i: int) -> List[AbstractLane]:
        lanes = self.upstream_lane_list if upstream else self.downstream_lane_list
        return [lane for lane in lanes if lane.index[2] == i]

    @lru_cache()
    def get_lane_at_position(self, upstream: bool, i: int, x) -> Optional[AbstractLane]:
        lanes = self.get_lanes_for_index(upstream, i)
        for lane in lanes:
            x1 = lane.start_pos[0] if upstream else lane.end_pos[0]
            x2 = lane.end_pos[0] if upstream else lane.start_pos[0]
            if x1 <= x <= x2:
                return lane

        print(f'lane not found for upstream: {upstream}, index: {i}, x_pos: {x}')
        return None

    @lru_cache()
    def side_lanes(self, lane: "AbstractLane", x_pos=None) -> List[AbstractLane]:
        """ Returns lanes to left and right of given lane """
        if lane.is_offramp:
            return []

        lanes = []

        o, d, i = lane.index
        right_o, right_d, right_i = o, d, i + 1
        left_o, left_d, left_i = o, d, i - 1

        upstream = lane.upstream

        if i == 1:
            if upstream:
                right_o, right_d = WEST_NODE, EAST_NODE
            else:
                right_o, right_d = EAST_NODE, WEST_NODE
            left_o, left_d = None, None

        if i == 2:
            left_lane = self.get_lane_at_position(upstream, i=1, x=x_pos)
            if left_lane is None:
                left_o, left_d = None, None
            else:
                left_o, left_d, left_i = left_lane.index

        try:
            lanes.append(self.graph[right_o][right_d][right_i])
        except KeyError:
            pass

        try:
            lanes.append(self.graph[left_o][left_d][left_i])
        except KeyError:
            pass

        return lanes

    @lru_cache()
    def bfs_paths(self, start: str, goal: str) -> List[Path]:
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node]) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    @lru_cache()
    def shortest_path(self, start: str, goal: str) -> Path:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return next(self.bfs_paths(start, goal), [])

    @lru_cache()
    def is_connected(self, l1: LaneIndex, l2: LaneIndex, same_lane: bool = False, route: Route = None, depth: int = 0):
        graph = self.graph

        if l1[:2] == l2[:2] and (not same_lane or l1[2] == l2[2]):
            return True

        if l1[1] == l2[0] and (not same_lane or l1[2] == l2[2]):
            return True

        if same_lane and l1[2] not in graph[l1[0]][l1[1]]:
            return False

        destinations = self.upstream_sink_destinations | self.downstream_sink_destinations
        if l1[1] in destinations:
            return False

        if depth > 0:
            if route:
                r = route[0]
                if r[:2] == l1[:2]:
                    return self.is_connected(l1, l2, same_lane, route[1:], depth)
                if l1[1] == r[0]:
                    return self.is_connected(r, l2, same_lane, route[1:], depth - 1)
            else:
                o, d, i = l1
                return any([self.is_connected((d, d_, i), l2, same_lane, route, depth - 1)
                            for d_ in graph.get(d, {})])

        return False

    @lru_cache()
    def get_path_from_lane_to_destination(self, lane, destination):
        o, d, i = lane.index
        path = [lane.index]

        if d == destination:
            return path

        if lane.is_sink:
            return []

        for next_destination, lanes in self.graph[d].items():
            if i in lanes:
                lane = lanes[i]
            elif i + 1 in lanes:
                lane = lanes[i + 1]

            return path + self.get_path_from_lane_to_destination(lane, destination)

    def position_heading_along_route(self, route: Route, longitudinal: float, lateral: float) \
            -> Tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :return: position, heading
        """

        while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
            longitudinal -= self.get_lane(route[0]).length
            route = route[1:]

        lane = self.get_lane(route[0])

        position = lane.position(longitudinal, lateral)
        heading = lane.heading_at(longitudinal)

        return position, heading

    def get_next_lane(self, lane, route=None, position=None):
        if lane.is_sink:
            return lane

        if lane.next_lane_index_options:
            if len(lane.next_lane_index_options) == 1:
                next_lane_index = lane.next_lane_index_options[0]
            else:
                o, d, i = lane.index

                found = False
                for o_, d_, i_ in lane.next_lane_index_options:
                    if i == i_:
                        next_lane_index = (o_, d_, i_)
                        found = True
                        break

                if not found:
                    next_lane_index = random.choice(lane.next_lane_index_options)

            next_lane = self.get_lane(next_lane_index)

            if next_lane is None:
                print(f'next lane not found: {lane.index}, {next_lane_index}\n')

            return next_lane

        else:
            o, d, i = lane.index
            options = self.graph[d]
            for option in options:
                try:
                    return self.graph[d][option][i]
                except:
                    continue

        return lane

    def __repr__(self):
        return str(self.graph)

    def __iter__(self) -> Iterable["AbstractLane"]:
        """ Iterate through every lane in the graph """
        for origin in self.graph:
            for destination in self.graph[origin]:
                for index, lane in self.graph[origin][destination].items():
                    yield lane
