# %% imports
import sys
import numpy as np

sys.path.insert(0, '../..')

from simulation.road_factory import constants as c
from simulation.road_factory.roads import create_road
from simulation.custom_types import LaneIndex
from simulation.car import Car
from simulation.mobil import Mobil
from simulation.idm import IDM
from simulation import utils


road = create_road()

#%%

list(road.network.graph['upstream_offramp0_merge_start'])
#%%

def get_path_from_lane(lane, destination = None):
    lane_index = lane.lane_index
    path = [lane_index]

    if lane.is_sink:
        return path

    index = lane.index

    while True:
        for next_destination, lanes in road.network.graph[lane_index.destination].items():
            if index in lanes:
                lane = lanes[index]
            else:
                if index + 1 in lanes:
                    return path + get_path_from_lane(lanes[index+1])

            lane_index = lane.lane_index
            path.append(lane_index)
            if lane.is_sink:
                return path


lane = road.network.get_lane(LaneIndex('upstream_onramp0_runway_start', 'upstream_onramp0_converge_start', 0))
path = get_path_from_lane(lane)
path

#%%

def get_path_from_lane(lane, destination):
    lane_index = lane.lane_index
    print(lane_index)
    path = [lane_index]

    if lane.destination == destination:
        return path

    if lane.is_sink:
        return []

    index = lane.index

    for next_destination, lanes in road.network.graph[lane_index.destination].items():
        if index in lanes:
            lane = lanes[index]
        elif index + 1 in lanes:
            lane = lanes[index+1]

        return path + get_path_from_lane(lane, destination)


lane = road.network.get_lane(LaneIndex('west', 'east', 3))
destination = 'upstream_offramp0_runway_end'
route = get_path_from_lane(lane, destination=destination)
route

#%%

print(lane.lane_index)
#%%
lane = road.network.get_lane(LaneIndex('west', 'east', 3))
destination = 'upstream_offramp0_runway_end'
lane_index = lane.lane_index
route = [lane_index]

while route[-1].destination != destination:
    lane_index = LaneIndex(lane.origin, lane.destination, lane.index - 1)
    lane = road.network.get_lane(lane_index)
    print(lane)
    route = route + get_path_from_lane(lane, destination)
    print(route)

route




#%%
def plan_route_to(lane, destination: str):
    """
    Plan a route to a destination in the road network

    :param destination: a node in the road network

    example: plan_route_to('east')
    """
    lane_index = lane.lane_index
    route = [lane_index]

    if lane.destination == destination:
        return route

    if lane.is_sink:
        return []

    try:
        path = road.network.shortest_path(lane.destination, destination)
    except KeyError:
        path = []

    for i in range(len(path) - 1):
        origin = path[i]
        destination = path[i + 1]
        lane_index = LaneIndex(origin=origin, destination=destination, index=None)
        route.append(lane_index)

    return route

lane = road.network.get_lane(LaneIndex('west', 'east', 3))
destination = 'upstream_offramp0_runway_end'
route = plan_route_to(lane, destination)
route

# %%
lane1 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_start', 'upstream_onramp0_merge_end', 0))
lane2 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_start', 'upstream_onramp0_merge_end', 1))
lane2 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_start', 'upstream_onramp0_merge_end', 1))

lane3 = road.network.get_lane(LaneIndex(c.WEST_NODE, 'upstream_onramp0_merge_start', 1))

lane4 = road.network.get_lane(LaneIndex(c.WEST_NODE, c.EAST_NODE, 2))
lane5 = road.network.get_lane(LaneIndex(c.WEST_NODE, c.EAST_NODE, 4))

lane6 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_start', 'upstream_onramp0_merge_end', 1))
lane7 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_end', 'upstream_offramp0_merge_start', 1))
lane8 = road.network.get_lane(LaneIndex('upstream_offramp0_merge_start', 'upstream_offramp0_merge_end', 1))
lane9 = road.network.get_lane(LaneIndex('upstream_offramp0_merge_end', 'upstream_onramp1_merge_start', 1))
lane10 = road.network.get_lane(LaneIndex('upstream_offramp0_merge_end', 'upstream_offramp0_converge_end', 1))

# %%
car1 = Car.make_on_lane(road, lane1, longitudinal=25)
# car2 = Car.make_on_lane(road, lane2, longitudinal=0)
# car3 = Car.make_on_lane(road, lane2, longitudinal=50)
car4 = Car.make_on_lane(road, lane3, longitudinal=5)
car5 = Car.make_on_lane(road, lane7, longitudinal=40)
car6 = Car.make_on_lane(road, lane3, longitudinal=20)
car7 = Car.make_on_lane(road, lane4, longitudinal=0)

# %%
lane1 = road.network.get_lane(LaneIndex('upstream_offramp0_merge_start', 'upstream_offramp0_merge_end', 1))
road.network.side_lanes(lane1)

# %%
# car1 = Car.make_on_lane(road, lane1, longitudinal=26.)
car2 = Car.make_on_lane(road, lane2, longitudinal=20.)
# car3 = Car.make_on_lane(road, lane3, longitudinal=10.)
# car4 = Car.make_on_lane(road, lane2, longitudinal=30.)
# car5 = Car.make_on_lane(road, lane2, longitudinal=25.)


# %%
lane1 = road.network.get_lane(LaneIndex('west', 'upstream_onramp0_merge_start', 1))
lane2 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_start', 'upstream_onramp0_merge_end', 1))
lane3 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_end', 'upstream_offramp0_merge_start', 1))
lane4 = road.network.get_lane(LaneIndex('upstream_offramp0_merge_start', 'upstream_offramp0_merge_end', 1))
lane5 = road.network.get_lane(LaneIndex('upstream_offramp0_merge_end', 'upstream_onramp1_merge_start', 1))

car1 = Car.make_on_lane(road, lane1, longitudinal=0.)
car2 = Car.make_on_lane(road, lane2, longitudinal=0.)
car3 = Car.make_on_lane(road, lane3, longitudinal=0.)
car4 = Car.make_on_lane(road, lane4, longitudinal=0.)
car5 = Car.make_on_lane(road, lane5, longitudinal=0.)

# %%

car1.lane.destination


