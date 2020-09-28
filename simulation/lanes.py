from abc import abstractmethod, ABC
from typing import Tuple
import re
import numpy as np

from simulation.config import LANE_WIDTH
from simulation.constants import CAR_LENGTH, HALF_CAR_LENGTH
from simulation.custom_types import Vector, LineType, Node, AngleRadians, LaneIndex, Origin, Destination, Index

onramp_pattern = re.compile(r'^(up|down)stream_onramp[\d]_merge_(start|end)$')
offramp_pattern = re.compile(r'^(up|down)stream_offramp[\d]_(merge|converge)_(start|end)$')


# offramp_pattern = re.compile(r'^(up|down)stream_offramp[\d]_merge_(start|end)$')


class AbstractLane(ABC):
    """A lane on the road, described by its central curve."""

    # Defaults
    start_pos = np.array([0, 0])
    end_pos = np.array([0, 0])
    length: float = 0
    width: float = LANE_WIDTH
    highway_speed_limit: float = 100.0 / 3.6
    ramp_speed_limit: float = 70.0 / 3.6
    left_line: LineType = LineType.NONE
    right_line: LineType = LineType.NONE
    is_source: bool = False
    is_sink: bool = False
    upstream: bool = False
    forbidden: bool = False
    origin: Origin = ''
    destination: Destination = ''
    index: Index = 0
    lane_index: "LaneIndex" = LaneIndex(origin, destination, index)
    speed_limit: float = highway_speed_limit

    is_onramp: bool = False
    is_offramp: bool = False
    is_next_to_offramp: bool = False

    onramp_merge_to_lane_index: "LaneIndex" = None
    offramp_lane_index: "LaneIndex" = None
    priority_lane_index: "LaneIndex" = None

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        """
        Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate, i.e. distance along lane in [m]
        :param lateral: lateral lane coordinate i.e. distance adjacent to lane in [m]
        :return: the corresponding world position (x,y) [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        """
        Convert a world position to local lane coordinates.

        :param position: a world position (x,y) [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    def same_lane(self, other):
        return self.lane_index == other.lane_index

    def same_road(self, other):
        return self.lane_index[:2] == other.lane_index[:2]

    def leading_to_lane(self, other):
        return self.destination == other.origin and self.index == other.index

    def leading_to_road(self, other):
        return self.destination == other.origin

    def on_lane(self,
                position: np.ndarray,
                longitudinal: float = None,
                lateral: float = None,
                margin: float = 0) -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if not longitudinal or not lateral:
            longitudinal, lateral = self.local_coordinates(position)

        return np.abs(lateral) <= self.width / 2 + margin and -CAR_LENGTH <= longitudinal < self.length + CAR_LENGTH

    def is_reachable_from(self, position: np.ndarray) -> bool:
        """
        Whether the lane is reachable from a given world position

        :param position: the world position [m]
        :return: is the lane reachable?
        """
        s, r = self.local_coordinates(position)
        return 0 <= s < self.length + CAR_LENGTH and np.abs(r) <= 2 * self.width

    def s(self, position: Vector) -> float:
        """ Return longitudinal distance to position """
        return self.local_coordinates(position)[0]

    def r(self, position: Vector) -> float:
        """ Return longitudinal distance to position """
        return self.local_coordinates(position)[1]

    def after_end(self, position: np.ndarray, longitudinal: float = None) -> bool:
        """ True is car half way point is past end_pos """
        if not longitudinal:
            longitudinal = self.s(position)
        return longitudinal + HALF_CAR_LENGTH > self.length

    def distance(self, position):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def __repr__(self):
        up_down = 'Upstream' if self.upstream else 'Downstream'
        x1, y1 = self.start_pos.astype(int)
        x2, y2 = self.end_pos.astype(int)
        return f'[{self.index}] {self.origin} -> {self.destination} ({up_down})\n[{x1}, {y1}] -> [{x2}, {y2}]'


class StraightLane(AbstractLane):
    """A lane going in straight line."""

    def __init__(self,
                 start_pos: Vector,
                 end_pos: Vector,
                 origin: Origin,
                 destination: Destination,
                 index: Index = 0,
                 forbidden: bool = False,
                 speed_limit: float = 28,
                 is_source: bool = False,
                 is_sink: bool = False,
                 left_line=LineType.NONE,
                 right_line=LineType.NONE) -> None:

        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)

        delta = self.end_pos - self.start_pos
        x,y = delta

        self.heading: float = np.arctan2(y, x) # radians
        self.length = np.linalg.norm(delta)
        self.direction: np.ndarray = delta / self.length
        self.direction_lateral: np.ndarray = np.flip(self.direction) * [-1, 1]

        self.upstream: bool = x >= 0
        self.forbidden = forbidden

        self.origin = origin
        self.destination = destination
        self.index = index
        self.lane_index = LaneIndex(origin, destination, index)

        self.speed_limit = speed_limit
        self.is_source = is_source
        self.is_sink = is_sink

        self.left_line = left_line
        self.right_line = right_line

        self.set_lane_settings()

    def set_lane_settings(self):
        o, d, i = self.lane_index

        origin_onramp_match = onramp_pattern.match(o)
        destination_onramp_match = onramp_pattern.match(d)
        origin_offramp_match = offramp_pattern.match(o)

        self.is_onramp = origin_onramp_match is not None and i == 0
        self.onramp_merge_to_lane_index = LaneIndex(o, d, 1) if self.is_onramp else None

        self.is_offramp = origin_offramp_match is not None and i == 0
        self.is_next_to_offramp = origin_offramp_match is not None and i == 1
        self.offramp_lane_index = LaneIndex(o, d, 0) if self.is_next_to_offramp else None

        # Before merge lane or next to merge lane
        if destination_onramp_match and i == 1:
            if destination_onramp_match[2] == 'start':
                self.priority_lane_index = LaneIndex(d, d.replace('start', 'end'), 0)
            else:
                self.priority_lane_index = LaneIndex(d.replace('end', 'start'), d, 0)


    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.start_pos + (longitudinal * self.direction) + (lateral * self.direction_lateral)

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def local_coordinates(self, position: np.ndarray) -> Vector:
        delta = position - self.start_pos
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)


class SineLane(StraightLane):
    """A sinusoidal lane."""

    def __init__(self,
                 start_pos: Vector,
                 end_pos: Vector,
                 amplitude: float,
                 pulsation: float,
                 phase: float,
                 origin: Node,
                 destination: Node,
                 index: int = 0,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 is_source: bool = False,
                 is_sink: bool = False,
                 left_line=LineType.NONE,
                 right_line=LineType.NONE) -> None:
        """
        New sinusoidal lane.

        :param start_pos: the lane starting position [m]
        :param end_pos: the lane ending position [m]
        :param amplitude: the lane oscillation amplitude [m]
        :param pulsation: the lane pulsation [rad/m]
        :param phase: the lane initial phase [rad]
        """
        super().__init__(start_pos=start_pos,
                         end_pos=end_pos,
                         origin=origin,
                         destination=destination,
                         index=index,
                         forbidden=forbidden,
                         speed_limit=speed_limit,
                         is_source=is_source,
                         is_sink=is_sink,
                         left_line=left_line,
                         right_line=right_line)
        self.amplitude = amplitude
        self.pulsation = pulsation
        self.phase = phase

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return super().position(longitudinal,
                                lateral + self.amplitude * np.sin(self.pulsation * longitudinal + self.phase))

    def heading_at(self, longitudinal: float) -> float:
        return super().heading_at(longitudinal) + np.arctan(
                self.amplitude * self.pulsation * np.cos(self.pulsation * longitudinal + self.phase))

    def local_coordinates(self, position: np.ndarray) -> Vector:
        longitudinal, lateral = super().local_coordinates(position)
        return longitudinal, lateral - self.amplitude * np.sin(self.pulsation * longitudinal + self.phase)
