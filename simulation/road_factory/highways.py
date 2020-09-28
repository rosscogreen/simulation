from simulation.config import *
from simulation.custom_types import CL, SL, NL
from simulation.road import Road
from simulation.road_factory.constants import *
from simulation.road_factory.lanes import make_lane
from simulation.road_factory.merge_lanes import make_upstream_highway_merge_lane, make_downstream_highway_merge_lane


def highway(road: "Road"):
    for index in range(1, NUM_LANES + 1):  # 1,2,3,4,5,6,7,8

        if index == LEFT_LANE_INDEX:  # Upstream Highway Merge Lane
            make_upstream_highway_merge_lane(road)

        elif index == RIGHT_LANE_INDEX:  # Downstream Highway Merge Lane
            make_downstream_highway_merge_lane(road)

        else:
            y = Y_ROAD_START + ((index - 1) * LANE_WIDTH)

            l1 = CL if index == NUM_UPSTREAM + 1 else SL

            up_forbidden = index > NUM_UPSTREAM
            down_forbidden = not up_forbidden

            down_lane_index = NUM_LANES - index + 1

            make_lane(x1=0, x2=ROAD_LENGTH, y1=y,
                      o=WEST_NODE, d=EAST_NODE, i=index,
                      f=up_forbidden,
                      source=True, sink=True,
                      l1=l1, l2=NL,
                      road=road)

            make_lane(x1=ROAD_LENGTH, x2=0, y1=y,
                      o=EAST_NODE, d=WEST_NODE, i=down_lane_index,
                      f=down_forbidden,
                      source=True, sink=True,
                      road=road)
