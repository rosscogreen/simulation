from simulation.lanes import StraightLane, AbstractLane
from simulation.road import Road
from simulation.custom_types import CL, SL, UL, NL
from simulation.road_factory.constants import NUM_LANES, WEST_NODE, EAST_NODE


def make_lane(x1, x2, y1, o, d, i,
              y2=None,
              f=None,
              l1=None, l2=None,
              nlanes=None,
              nup=None,
              source=None,
              sink=None,
              rev=False,
              road: "Road" = None) -> "AbstractLane":
    nlanes = nlanes or NUM_LANES
    nup = nup or nlanes // 2
    y2 = y2 or y1

    lane = StraightLane(start_pos=[x1, y1], end_pos=[x2, y2], origin=o, destination=d, index=i)

    if not lane.upstream:
        lane.left_line = NL
        lane.right_line = NL

    if l1 is not None:
        lane.left_line = l1
    else:
        if lane.upstream:
            if i == 0 or i == nup:
                lane.left_line = CL
            else:
                lane.left_line = SL

    if l2 is not None:
        lane.right_line = l2
    else:
        if lane.upstream and i == nlanes - 1:
            lane.right_line = CL

    if f is not None:
        lane.forbidden = f
    else:
        lane.forbidden = (lane.upstream and i >= nup) or (not lane.upstream and i < nup)

    if source is not None:
        lane.is_source = source
    else:
        lane.is_source = (lane.upstream and lane.origin == WEST_NODE) \
                         or (not lane.upstream and lane.origin == EAST_NODE)

    if sink is not None:
        lane.is_sink = sink
    else:
        lane.is_sink = (lane.upstream and lane.destination == EAST_NODE) \
                       or (not lane.upstream and lane.destination == WEST_NODE)

    if road is not None:
        road.network.add_lane(lane)

    return lane
