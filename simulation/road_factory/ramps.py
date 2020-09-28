import numpy as np

from simulation.road_objects import Obstacle
from simulation.config import *
from simulation.custom_types import CL, UL, NL
from simulation.lanes import StraightLane, SineLane
from simulation.road import Road
from simulation.road_factory.constants import *


def onramp(road: "Road", x_start, upstream: bool, ramp_idx: int):

    x_start = x_start if upstream else ROAD_LENGTH - x_start
    x_end = x_start + RUNWAY_LENGTH if upstream else x_start - RUNWAY_LENGTH
    y = Y_UPSTREAM_RAMP_START if upstream else Y_DOWNSTREAM_RAMP_START

    up_down = 'upstream' if upstream else 'downstream'

    runway_node = f'{up_down}_onramp{ramp_idx}_runway_start'

    converge_node = f'{up_down}_onramp{ramp_idx}_converge_start'

    merge_origin_node = f'{up_down}_onramp{ramp_idx}_merge_start'
    merge_destination_node = f'{up_down}_onramp{ramp_idx}_merge_end'

    barrier_origin_node = f"{up_down}_onramp{ramp_idx}_barrier_start"
    barrier_destination_node = f"{up_down}_onramp{ramp_idx}_barrier_end"

    l1 = CL if upstream else NL
    l2 = NL if upstream else CL

    # Runway Lane
    runway = StraightLane(start_pos=[x_start, y],
                          end_pos=[x_end, y],
                          origin=runway_node,
                          destination=converge_node,
                          index=0,
                          left_line=CL, right_line=CL,
                          is_source=True,
                          speed_limit=RAMP_SPEED_LIMIT)

    # Converge Lane
    amplitude = -LANE_WIDTH / 2
    converge = SineLane(start_pos=runway.position(longitudinal=RUNWAY_LENGTH, lateral=-amplitude),
                        end_pos=runway.position(longitudinal=RUNWAY_LENGTH + CONVERGE_LENGTH, lateral=-amplitude),
                        origin=converge_node,
                        destination=merge_origin_node,
                        index=0,
                        amplitude=amplitude,
                        pulsation=np.pi / CONVERGE_LENGTH,
                        phase=np.pi / 2,
                        left_line=UL, right_line=UL,
                        speed_limit=RAMP_SPEED_LIMIT)

    # Merge Lane
    start = converge.position(CONVERGE_LENGTH, 0)
    end = start + [MERGE_LENGTH, 0] if upstream else start - [MERGE_LENGTH, 0]
    merge = StraightLane(start_pos=start,
                         end_pos=end,
                         origin=merge_origin_node,
                         destination=merge_destination_node,
                         index=0,
                         left_line=UL,
                         forbidden=True)

    Obstacle.make_on_lane(road=road, lane=merge, longitudinal=merge.length)

    # Barrier
    start = merge.position(MERGE_LENGTH, 0)
    end = start + [BARRIER_LENGTH, LANE_WIDTH] if upstream else start - [BARRIER_LENGTH, LANE_WIDTH]
    barrier = StraightLane(start_pos=start,
                           end_pos=end,
                           origin=barrier_origin_node,
                           destination=barrier_destination_node,
                           index=0,
                           left_line=l1,
                           right_line=l2,
                           forbidden=True,
                           speed_limit=0)

    road.network.add_lanes([runway, converge, merge, barrier])


def offramp(road: "Road", x_start: int, upstream: bool, ramp_idx: int):
    x_start = x_start if upstream else ROAD_LENGTH - x_start
    x_end = x_start + BARRIER_LENGTH if upstream else x_start - BARRIER_LENGTH
    y_start = Y_ROAD_START if upstream else Y_ROAD_START + ROAD_WIDTH
    y_end = y_start - LANE_WIDTH if upstream else y_start + LANE_WIDTH

    up_down = 'upstream' if upstream else 'downstream'

    barrier_origin_node = f"{up_down}_offramp{ramp_idx}_barrier_start"
    barrier_destination_node = f"{up_down}_offramp{ramp_idx}_barrier_end"

    merge_origin_node = f'{up_down}_offramp{ramp_idx}_merge_start'
    merge_destination_node = f'{up_down}_offramp{ramp_idx}_merge_end'

    converge_node = f'{up_down}_offramp{ramp_idx}_converge_end'

    runway_node = f'{up_down}_offramp{ramp_idx}_runway_end'

    l1 = CL if upstream else NL
    l2 = NL if upstream else CL

    barrier = StraightLane(start_pos=[x_start, y_start],
                           end_pos=[x_end, y_end],
                           origin=barrier_origin_node,
                           destination=barrier_destination_node,
                           index=0,
                           left_line=l1,
                           right_line=l2,
                           speed_limit=0)

    start = np.array([x_end, y_end])
    end = start + [MERGE_LENGTH, 0] if upstream else start - [MERGE_LENGTH, 0]
    merge = StraightLane(start_pos=start,
                         end_pos=end,
                         origin=merge_origin_node,
                         destination=merge_destination_node,
                         left_line=l1,
                         right_line=l2,
                         speed_limit=RAMP_SPEED_LIMIT)

    amplitude = LANE_WIDTH / 2
    converge = SineLane(start_pos=merge.position(MERGE_LENGTH, -amplitude),
                        end_pos=merge.position(MERGE_LENGTH + CONVERGE_LENGTH, -amplitude),
                        origin=merge_destination_node,
                        destination=converge_node,
                        amplitude=amplitude,
                        pulsation=np.pi / CONVERGE_LENGTH,
                        phase=np.pi / 2,
                        left_line=UL,
                        right_line=UL,
                        speed_limit=RAMP_SPEED_LIMIT)

    start = converge.position(CONVERGE_LENGTH, 0)
    end = start + [RUNWAY_LENGTH, 0] if upstream else start - [RUNWAY_LENGTH, 0]
    runway = StraightLane(start_pos=start,
                          end_pos=end,
                          origin=converge_node,
                          destination=runway_node,
                          left_line=UL,
                          right_line=UL,
                          is_sink=True,
                          speed_limit=RAMP_SPEED_LIMIT)

    road.network.add_lanes([barrier, merge, converge, runway])
