"""
        striped_length = 55
        [ 35 150 295 410]
        onramp, offramp, onramp, offramp

        i = 0
        [0 -> 35] block
        [35 -> 90] allow from onramp 1

        i = 1
        [90 -> 150] block
        [150 -> 205] allow to offramp 1

        i = 2
        [205 -> 295] block
        [295 -> 350] allow from onramp 2

        i = 3
        [350 -> 410] block
        [410 -> 465] allow to offramp 2
    """

from simulation.config import *
from simulation.custom_types import CL, SL
from simulation.road_factory.constants import *
from simulation.road_factory.lanes import make_lane


def make_upstream_highway_merge_lane(road):
    """ Upstream merge lane """

    y = Y_ROAD_START

    x1 = 0
    n1 = WEST_NODE

    onramp_index = 0
    offramp_index = 0

    for i, x2 in enumerate(MERGEABLE_SECTION_STARTS):
        x3 = x2 + MERGE_SECTION_LENGTH

        if i % 2 == 0:
            # On ramps
            n2 = f'upstream_onramp{onramp_index}_merge_start'
            n3 = f'upstream_onramp{onramp_index}_merge_end'
            onramp_index += 1
        else:
            # Off ramps
            n2 = f'upstream_offramp{offramp_index}_merge_start'
            n3 = f'upstream_offramp{offramp_index}_merge_end'
            offramp_index += 1

        block = make_lane(x1=x1, x2=x2, y1=y,
                          o=n1, d=n2, i=LEFT_LANE_INDEX,
                          l1=CL, road=road)
        if i == 0:
            block.is_source = True

        make_lane(x1=x2, x2=x3, y1=y,
                  o=n2, d=n3, i=LEFT_LANE_INDEX,
                  l1=SL, road=road)

        # start at striped line end on next iteration
        x1 = x3
        n1 = n3


    # End x_starts loop

    make_lane(x1=x1, x2=ROAD_LENGTH, y1=y,
              o=n1, d=EAST_NODE, i=LEFT_LANE_INDEX,
              sink=True, l1=CL,
              road=road)

    # Downstream
    make_lane(x1=ROAD_LENGTH, x2=0, y1=y,
              o=EAST_NODE, d=WEST_NODE, i=RIGHT_LANE_INDEX,
              f=True,
              road=road)


def make_downstream_highway_merge_lane(road):
    """ Downstream merge lane """

    y = Y_ROAD_START + ROAD_WIDTH

    x1 = ROAD_LENGTH
    n1 = EAST_NODE
    n1_up = WEST_NODE

    onramp_index = 0
    offramp_index = 0

    starts = ROAD_LENGTH - MERGEABLE_SECTION_STARTS

    for i, x2 in enumerate(starts):
        x3 = x2 - MERGE_SECTION_LENGTH

        if i % 2 == 0:
            # On ramps
            n2 = f'downstream_onramp{onramp_index}_merge_start'
            n3 = f'downstream_onramp{onramp_index}_merge_end'
            n2_up = f'forbidden_upstream_onramp{onramp_index}_merge_start'
            n3_up = f'forbidden_upstream_onramp{onramp_index}_merge_end'
            onramp_index += 1
        else:
            # Off ramps
            n2 = f'downstream_offramp{offramp_index}_merge_start'
            n3 = f'downstream_offramp{offramp_index}_merge_end'
            n2_up = f'forbidden_upstream_offramp{onramp_index}_merge_start'
            n3_up = f'forbidden_upstream_offramp{onramp_index}_merge_end'
            offramp_index += 1

        # Draw with upstream lanes
        make_lane(x1=x2, x2=x1, y1=y,
                  o=n1_up, d=n2_up, i=RIGHT_LANE_INDEX,
                  f=True, l1=SL, l2=CL, road=road, source=False, sink=False)

        make_lane(x1=x3, x2=x2, y1=y,
                  o=n2_up, d=n3_up, i=RIGHT_LANE_INDEX,
                  f=True, l1=SL, l2=SL, road=road, source=False, sink=False)

        # Downstream Lanes
        down_block = make_lane(x1=x1, x2=x2, y1=y,
                               o=n1, d=n2, i=LEFT_LANE_INDEX,
                               f=False, road=road)

        make_lane(x1=x2, x2=x3, y1=y,
                  o=n2, d=n3, i=LEFT_LANE_INDEX,
                  f=False, road=road)

        if i == 0:
            down_block.is_source = True

        #print(f'{n1_up} -> {n2_up} -> {n3_up}')

        # start at striped line end on next iteration
        x1 = x3
        n1 = n3
        n1_up = n3_up

    #print(f'{n1_up} -> {EAST_NODE}')

    make_lane(x1=0, x2=x1, y1=y,
              o=n1_up, d=EAST_NODE, i=RIGHT_LANE_INDEX,
              f=True, l1=SL, l2=CL, road=road)

    make_lane(x1=x1, x2=0, y1=y,
              o=n1, d=WEST_NODE, i=LEFT_LANE_INDEX,
              sink=True, road=road)
