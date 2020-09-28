import logging
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Iterable, Tuple, DefaultDict, Dict, Set

import numpy as np

from simulation.car import Car
from simulation.custom_types import Route, LanesList, Origin, Destination, Index, Path, LaneIndex
from simulation.lanes import AbstractLane
from simulation.road_factory.constants import WEST_NODE, EAST_NODE

logger = logging.getLogger(__name__)


class RoadNetwork(object):
    graph: DefaultDict[Origin, DefaultDict[Destination, Dict[Index, AbstractLane]]]

    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(dict))

        self.upstream_source_lanes: LanesList = []
        self.downstream_source_lanes: LanesList = []

        self.upstream_sink_lanes: LanesList = []
        self.downstream_sink_lanes: LanesList = []

        self.upstream_sink_destinations: Set[Destination] = set()
        self.downstream_sink_destinations: Set[Destination] = set()

        self.upstream_source_origins: Set[Origin] = set()
        self.downstream_source_origins: Set[Origin] = set()

        self.upstream_lane_list = []
        self.downstream_lane_list = []

    def add_lane(self, lane: "AbstractLane"):
        self.graph[lane.origin][lane.destination][lane.index] = lane

        if lane.upstream:
            self.upstream_lane_list.append(lane)
            if lane.is_source:
                self.upstream_source_lanes.append(lane)
                self.upstream_source_origins.add(lane.origin)
            elif lane.is_sink:
                self.upstream_sink_lanes.append(lane)
                self.upstream_sink_destinations.add(lane.destination)
        else:
            self.downstream_lane_list.append(lane)
            if lane.is_source:
                self.downstream_source_lanes.append(lane)
                self.downstream_source_origins.add(lane.origin)
            elif lane.is_sink:
                self.downstream_sink_lanes.append(lane)
                self.downstream_sink_destinations.add(lane.destination)

    def add_lanes(self, lanes: LanesList) -> None:
        for lane in lanes:
            self.add_lane(lane)

    def get_lanes(self, origin: Origin, destination: Destination) -> LanesList:
        """ Return all lanes from origin -> destination """
        return list(self.graph[origin][destination].values())

    @lru_cache()
    def get_lane(self, lane_index: LaneIndex) -> Optional["AbstractLane"]:
        """
            Get the lane geometry corresponding to a given index in the road_network network.
        :return: the corresponding lane geometry.
        """
        o, d, i = lane_index

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
    def get_lanes_for_index(self, upstream: bool, index: int) -> List[AbstractLane]:
        lanes = self.upstream_lane_list if upstream else self.downstream_lane_list
        return [lane for lane in lanes if lane.index == index]

    @lru_cache()
    def get_lane_at_position(self, upstream: bool, index: int, x) -> AbstractLane:
        lanes = self.get_lanes_for_index(upstream, index)
        for lane in lanes:
            x1 = lane.start_pos[0] if upstream else lane.end_pos[0]
            x2 = lane.end_pos[0] if upstream else lane.start_pos[0]
            if x1 <= x < x2:
                return lane

    @lru_cache()
    def side_lanes(self, lane: "AbstractLane", x_pos=None) -> LanesList:
        """ Returns lanes to left and right of given lane """
        if lane.is_offramp:
            return []

        lanes = []

        o, d, i = lane.lane_index
        right_o, right_d, right_i = o, d, i + 1
        left_o, left_d, left_i = o, d, i - 1

        upstream = lane.upstream

        if i == 1:
            if lane.upstream:
                right_o, right_d = WEST_NODE, EAST_NODE
            else:
                right_o, right_d = EAST_NODE, WEST_NODE

            if not lane.is_next_to_offramp:
                left_o, left_d = None, None

        if i == 2:
            lane = self.get_lane_at_position(upstream, 1, x_pos)
            left_o, left_d, left_i = lane.lane_index

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
    def bfs_paths(self, start: Destination, goal: Destination) -> List[Path]:
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

        if same_lane and l1.index not in graph[l1[0]][l1[1]]:
            return False

        destinations = self.upstream_sink_destinations | self.downstream_sink_destinations
        if l1[1] in destinations:
            return False

        if depth > 0:
            if route:
                r = route[0]
                if r[:2] == l1[:2]:
                    return self.is_connected(l1, l2, same_lane, route[1:], depth)
                if l1.destination == r.origin:
                    return self.is_connected(r, l2, same_lane, route[1:], depth - 1)
            else:
                o, d, i = l1
                return any([self.is_connected(LaneIndex(d, d_, i), l2, same_lane, route, depth - 1)
                            for d_ in graph.get(d, {})])

        return False

    @lru_cache()
    def get_path_from_lane_to_destination(self, lane, destination):
        lane_index = lane.lane_index
        path = [lane_index]

        if lane.destination == destination:
            return path

        if lane.is_sink:
            return []

        index = lane.index

        for next_destination, lanes in self.graph[lane_index.destination].items():
            if index in lanes:
                lane = lanes[index]
            elif index + 1 in lanes:
                lane = lanes[index + 1]

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

        lane_index = lane.lane_index
        o, d, i = lane_index

        graph = self.graph[d]
        d_list = list(graph)

        d_next = None

        # Finished first leg of route -> remove it
        if route and route[0][:2] == lane_index[:2]:
            route.pop(0)

        if route and d == route.origin:
            d_next = route.destination

        if d_next is None:
            if len(d_list) == 1:
                d_next = d_list[0]
            else:
                dest_options = [d for d in graph if i in graph[d]]
                if dest_options:
                    d_next = np.random.choice(dest_options)
                else:
                    d_next = np.random.choice(d_list)

        try:
            return graph[d_next][i]
        except KeyError:
            pass

        if len(graph[d_next]) == 1:
            i_next = list(graph[d_next])[0]
            return graph[d_next][i_next]

        if position is None:
            position = lane.end_pos

        return min(list(graph[d_next].values()), key=lambda lane: lane.distance(position))

    def __repr__(self):
        return str(self.graph)

    def __iter__(self) -> Iterable["AbstractLane"]:
        """ Iterate through every lane in the graph """
        for origin in self.graph:
            for destination in self.graph[origin]:
                for index, lane in self.graph[origin][destination].items():
                    yield lane
