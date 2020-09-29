import re
from abc import abstractmethod, ABC
from typing import Tuple, List

import numpy as np

from simulation.config import LANE_WIDTH, HIGHWAY_SPEED_LIMIT
from simulation.constants import CAR_LENGTH
from simulation.custom_types import Vector, LineType, LaneIndex

onramp_pattern = re.compile(r'^(up|down)stream_onramp[\d]_merge_(start|end)$')
offramp_pattern = re.compile(r'^(up|down)stream_offramp[\d]_(merge|converge)_(start|end)$')


class AbstractLane(ABC):
    """A lane on the road, described by its central curve."""

    CL = LineType.CONTINUOUS_LINE
    SL = LineType.STRIPED
    UL = LineType.CONTINUOUS
    NL = LineType.NONE

    # Defaults
    start_pos = np.array([0, 0])
    end_pos = np.array([0, 0])

    length: float = 0
    width: float = LANE_WIDTH

    line_types = [NL, NL]

    is_source: bool = False
    is_sink: bool = False

    upstream: bool = False
    forbidden: bool = False

    speed_limit: float = HIGHWAY_SPEED_LIMIT

    index: LaneIndex = None

    path_index: LaneIndex = index
    path_start_pos = start_pos

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
    def path_position(self, longitudinal: float, lateral: float) -> np.ndarray:
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

    @abstractmethod
    def path_coordinates(self, position: np.ndarray) -> Vector:
        raise NotImplementedError()

    def same_lane(self, other: "AbstractLane"):
        return self.index == other.index

    def same_road(self, other: "AbstractLane"):
        return self.index[:2] == other.index[:2]

    def leading_to_lane(self, other: "AbstractLane"):
        return self.index[1] == other.index[0] and self.index[2] == other.index[2]

    def leading_to_road(self, other: "AbstractLane"):
        return self.index[1] == other.index[0]

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
        return longitudinal + (CAR_LENGTH / 2) > self.length

    def distance(self, position):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def __repr__(self):
        dir = 'U' if self.upstream else 'D'
        x1, y1 = self.start_pos.astype(int)
        x2, y2 = self.end_pos.astype(int)
        return f'{self.index} ({dir})\nstart=[{x1}, {y1}] end=[{x2}, {y2}]'


class StraightLane(AbstractLane):
    """A lane going in straight line."""

    def __init__(self,
                 start_pos: Vector,
                 end_pos: Vector,
                 index: LaneIndex,
                 line_types: List[LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = HIGHWAY_SPEED_LIMIT,
                 is_source: bool = False,
                 is_sink: bool = False,
                 path_index: LaneIndex = None,
                 path_start_pos = None
                 ) -> None:

        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.index = index

        self.path_index = path_index or index
        self.path_start_pos = path_start_pos or start_pos

        self.speed_limit = speed_limit
        self.is_source = is_source
        self.is_sink = is_sink

        if line_types is not None:
            self.line_types = line_types

        delta = self.end_pos - self.start_pos
        x, y = delta

        self.heading: float = np.arctan2(y, x)  # radians
        self.length = np.linalg.norm(delta)
        self.direction: np.ndarray = delta / self.length
        self.direction_lateral: np.ndarray = np.flip(self.direction) * [-1, 1]

        self.upstream: bool = x >= 0
        self.forbidden = forbidden

        self.set_lane_settings()

    def set_lane_settings(self):
        o, d, i = self.index

        origin_onramp_match = onramp_pattern.match(o)
        destination_onramp_match = onramp_pattern.match(d)
        origin_offramp_match = offramp_pattern.match(o)

        self.is_onramp = origin_onramp_match is not None and i == 0
        self.onramp_merge_to_lane_index = (o, d, 1) if self.is_onramp else None

        self.is_offramp = origin_offramp_match is not None and i == 0
        self.is_next_to_offramp = origin_offramp_match is not None and i == 1
        self.offramp_lane_index = (o, d, 0) if self.is_next_to_offramp else None

        # Before merge lane or next to merge lane
        if destination_onramp_match and i == 1:
            if destination_onramp_match[2] == 'start':
                self.priority_lane_index = (d, d.replace('start', 'end'), 0)
            else:
                self.priority_lane_index = (d.replace('end', 'start'), d, 0)

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.start_pos + (longitudinal * self.direction) + (lateral * self.direction_lateral)

    def path_position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.path_start_pos + (longitudinal * self.direction) + (lateral * self.direction_lateral)

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def local_coordinates(self, position: np.ndarray) -> Vector:
        delta = position - self.start_pos
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)

    def path_coordinates(self, position: np.ndarray) -> Vector:
        delta = position - self.path_start_pos
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)


class SineLane(StraightLane):
    """A sinusoidal lane."""

    def __init__(self,
                 start_pos: Vector,
                 end_pos: Vector,
                 index,
                 line_types: List[LineType] = None,
                 amplitude: float = 0,
                 pulsation: float = 0,
                 phase: float = 0,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 is_source: bool = False,
                 is_sink: bool = False) -> None:
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
                         index=index,
                         forbidden=forbidden,
                         speed_limit=speed_limit,
                         is_source=is_source,
                         is_sink=is_sink,
                         line_types=line_types)
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
