import numpy as np
import itertools

from simulation.config import *
from simulation.custom_types import CL, SL, NL, UL
from simulation.lanes import StraightLane, SineLane
from simulation.road import Road

ON_RAMP_STARTS = [0, 260]
OFF_RAMP_STARTS = [150, 410]

RUNWAY_LENGTH = 5
CONVERGE_LENGTH = 20
MERGE_LENGTH = 60
BARRIER_LENGTH = 5

SECTION_LENGTHS = [RUNWAY_LENGTH, CONVERGE_LENGTH, MERGE_LENGTH, BARRIER_LENGTH]

HIGHWAY_SPEED_LIMIT: float = 100.0 / 3.6
RAMP_SPEED_LIMIT: float = 80.0 / 3.6

NUM_LANES = 8

RAMP_LENGTH = RUNWAY_LENGTH + CONVERGE_LENGTH + MERGE_LENGTH
MERGE_SECTION_LENGTH = MERGE_LENGTH + BARRIER_LENGTH
ON_RAMP_MERGE_STARTS = np.array(ON_RAMP_STARTS) + RUNWAY_LENGTH + CONVERGE_LENGTH
MERGEABLE_SECTION_STARTS = np.array(list(itertools.chain(*zip(ON_RAMP_MERGE_STARTS, OFF_RAMP_STARTS))))
RAMP_WIDTH = 2 * LANE_WIDTH
Y_UPSTREAM_RAMP_START = RAMP_WIDTH
Y_ROAD_START = Y_UPSTREAM_RAMP_START + RAMP_WIDTH
ROAD_WIDTH = (NUM_LANES - 1) * LANE_WIDTH
Y_DOWNSTREAM_RAMP_START = Y_ROAD_START + ROAD_WIDTH + RAMP_WIDTH


def create_road(x_start=0, x_end=ROAD_LENGTH, nlanes=NUM_LANES):
    road = Road()
    nup = nlanes // 2

    merge_lanes(road)

    # Reversable Lanes
    for i in range(2, NUM_LANES):  # 2,3,4,5,6,7
        y = Y_ROAD_START + ((i - 1) * LANE_WIDTH)

        road.network.add_lane(
                StraightLane([x_start, y], [x_end, y], (WEST_NODE, EAST_NODE, i),
                             [CL if i == nlanes // 2 + 1 else SL, NL],
                             forbidden=i > nup, is_source=True, is_sink=True))

        road.network.add_lane(StraightLane([x_end, y], [x_start, y], (EAST_NODE, WEST_NODE, nlanes - i + 1),
                                           forbidden=i <= nup, is_source=True, is_sink=True))

    # Entry Ramps
    for i, x in enumerate(ON_RAMP_STARTS):
        onramp(road=road, x=x, y=Y_UPSTREAM_RAMP_START, upstream=True, idx=i)
        onramp(road=road, x=ROAD_LENGTH - x, y=Y_DOWNSTREAM_RAMP_START, upstream=False, idx=i)

    # Exit Ramps
    for i, x in enumerate(OFF_RAMP_STARTS):
        offramp(road=road, x=x, y=Y_ROAD_START, upstream=True, idx=i)
        offramp(road=road, x=ROAD_LENGTH - x, y=Y_ROAD_START + ROAD_WIDTH, upstream=False, idx=i)

    return road


def onramp(road, x, y, idx, upstream):
    up_down = 'upstream' if upstream else 'downstream'
    runway_node = f'{up_down}_onramp{idx}_runway_start'
    converge_node = f'{up_down}_onramp{idx}_converge_start'
    merge_origin_node = f'{up_down}_onramp{idx}_merge_start'
    merge_destination_node = f'{up_down}_onramp{idx}_merge_end'
    barrier_origin_node = f"{up_down}_onramp{idx}_barrier_start"
    barrier_destination_node = f"{up_down}_onramp{idx}_barrier_end"

    upstream = 1 if upstream else -1

    lengths = np.array([0, RUNWAY_LENGTH, CONVERGE_LENGTH, MERGE_LENGTH, BARRIER_LENGTH])
    xs = x + upstream * np.cumsum(lengths)
    r = LANE_WIDTH / 2
    ys = y + upstream * r * np.arange(0, 4)

    runway = StraightLane([xs[0], ys[0]], [xs[1], ys[0]], (runway_node, converge_node, 0), [CL, CL],
                          is_source=True, speed_limit=RAMP_SPEED_LIMIT)

    converge = SineLane([xs[1], ys[1]], [xs[2], ys[1]], (converge_node, merge_origin_node, 0), [UL, UL],
                        amplitude=-r, pulsation=np.pi / CONVERGE_LENGTH, phase=np.pi / 2,
                        speed_limit=RAMP_SPEED_LIMIT)

    merge = StraightLane([xs[2], ys[2]], [xs[3], ys[2]], (merge_origin_node, merge_destination_node, 0), [UL, NL])

    barrier = SineLane([xs[3], ys[3]], [xs[4], ys[3]], (barrier_origin_node, barrier_destination_node, 0), [UL, NL],
                       amplitude=-r, pulsation=np.pi / BARRIER_LENGTH, phase=np.pi / 2,
                       forbidden=True, speed_limit=0)

    road.network.add_lanes([runway, converge, merge, barrier])


def offramp(road: "Road", x: int, y, upstream: bool, idx: int):
    up_down = 'upstream' if upstream else 'downstream'
    barrier_origin_node = f"{up_down}_offramp{idx}_barrier_start"
    barrier_destination_node = f"{up_down}_offramp{idx}_barrier_end"
    merge_origin_node = f'{up_down}_offramp{idx}_merge_start'
    merge_destination_node = f'{up_down}_offramp{idx}_merge_end'
    converge_node = f'{up_down}_offramp{idx}_converge_end'
    runway_node = f'{up_down}_offramp{idx}_runway_end'

    upstream = 1 if upstream else -1

    lengths = np.array([0, BARRIER_LENGTH, MERGE_LENGTH, CONVERGE_LENGTH, RUNWAY_LENGTH])
    xs = x + upstream * np.cumsum(lengths)
    r = -LANE_WIDTH / 2
    ys = y + upstream * r * np.arange(1, 5)

    barrier = SineLane([xs[0], ys[0]], [xs[1], ys[0]], (barrier_origin_node, barrier_destination_node, 0), [UL, NL],
                       amplitude=r, pulsation=np.pi / BARRIER_LENGTH, phase=-np.pi / 2,
                       forbidden=True, speed_limit=0)

    merge = StraightLane([xs[1], ys[1]], [xs[2], ys[1]], (merge_origin_node, merge_destination_node, 0), [UL, NL])

    converge = SineLane([xs[2], ys[2]], [xs[3], ys[2]], (merge_destination_node, converge_node, 0), [UL, UL],
                        amplitude=r, pulsation=np.pi / CONVERGE_LENGTH, phase=-np.pi / 2,
                        speed_limit=RAMP_SPEED_LIMIT)

    runway = StraightLane([xs[3], ys[3]], [xs[4], ys[3]], (converge_node, runway_node, 0), [CL, CL],
                          is_sink=True, speed_limit=RAMP_SPEED_LIMIT)

    road.network.add_lanes([barrier, merge, converge, runway])


def merge_lanes(road, x_start=0, x_end=ROAD_LENGTH, y=Y_ROAD_START, nlanes=NUM_LANES, width=LANE_WIDTH):
    onramp_index = offramp_index = 0
    l = MERGE_LENGTH + BARRIER_LENGTH

    n_up, n_down, x_up, x_down = WEST_NODE, EAST_NODE, x_start, x_end
    y2 = y + (nlanes-1) * width

    upstream_path_index = (WEST_NODE, EAST_NODE, 1)
    upstream_path_start_pos = (x_start, y)
    downstream_path_index = (EAST_NODE, WEST_NODE, 1)
    downstream_path_start_pos = (x_end, y2)

    for i, x in enumerate(MERGEABLE_SECTION_STARTS):

        if i % 2 == 0:
            ramp_type = 'onramp'
            idx = onramp_index
            onramp_index += 1
        else:
            ramp_type = 'offramp'
            idx = offramp_index
            offramp_index += 1

        n1_up = f'upstream_{ramp_type}{idx}_merge_start'
        n2_up = f'upstream_{ramp_type}{idx}_merge_end'
        n1_down = f'downstream_{ramp_type}{idx}_merge_start'
        n2_down = f'downstream_{ramp_type}{idx}_merge_end'

        up_1 = StraightLane([x_up, y], [x, y], (n_up, n1_up, 1), [CL, NL],
                            is_source=i == 0,
                            path_index=upstream_path_index, path_start_pos=upstream_path_start_pos)

        up_2 = StraightLane([x, y], [x + l, y], (n1_up, n2_up, 1), [SL, NL],
                            path_index=upstream_path_index, path_start_pos=upstream_path_start_pos)

        down_1 = StraightLane([x_down, y2], [x_end - x, y2], (n_down, n1_down, 1), [CL, NL],
                              is_source=i == 0, path_index=downstream_path_index, path_start_pos=downstream_path_start_pos)

        down_2 = StraightLane([x_end - x, y2], [x_end - x - l, y2], (n1_down, n2_down, 1), [SL, NL],
                              path_index=downstream_path_index, path_start_pos=downstream_path_start_pos)

        road.network.add_lanes([up_1, up_2, down_1, down_2])

        x_up = x + l
        x_down = x_end - x - l
        n_up = n2_up
        n_down = n2_down

    up_1 = StraightLane([x_up, y], [x_end, y], (n_up, EAST_NODE, 1), [CL, NL],
                        is_sink=True, path_index=upstream_path_index, path_start_pos=upstream_path_start_pos)

    down_1 = StraightLane([x_end - MERGEABLE_SECTION_STARTS[-1] - l, y2], [x_start, y2], (n_down, WEST_NODE, 1), [CL, NL],
                          is_sink=True, path_index=downstream_path_index, path_start_pos=downstream_path_start_pos)

    _up_1 = StraightLane([x_start, y2], [x_end, y2], (WEST_NODE, EAST_NODE, nlanes), line_types=[SL,NL], forbidden=True)
    _down_1 = StraightLane([x_end, y], [x_start, y], (EAST_NODE, WEST_NODE, nlanes), forbidden=True)

    road.network.add_lanes([up_1, down_1, _up_1, _down_1])
