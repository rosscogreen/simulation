# %% imports
import sys

import pygame

sys.path.insert(0, '../..')

from simulation.config import *
from simulation.road_factory.constants import *
from simulation.car import Car
from simulation.graphics.common import WorldSurface
from simulation.graphics.road import RoadGraphics
from simulation.road_factory.roads import create_road
from simulation.custom_types import LaneIndex
from simulation import utils

road = create_road()
# %%
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT // 4))
road_surface = WorldSurface((SCREEN_WIDTH, SCREEN_HEIGHT // 4), SCALING)
road_graphics = RoadGraphics(road=road)
road_graphics.draw(surface=road_surface)
road.cars.draw(road_surface)
screen.blit(road_surface, (0, 0))
pygame.image.save(screen, "image.jpg")

# %% On Ramp
lane1 = road.network.get_lane(LaneIndex('upstream_onramp0_runway_start', 'upstream_onramp0_converge_start', 0))
lane2 = road.network.get_lane(LaneIndex('upstream_onramp0_converge_start', 'upstream_onramp0_merge_start', 0))
lane3 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_start', 'upstream_onramp0_merge_end', 0))
lane4 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_end', 'upstream_offramp0_merge_start', 1))

# %% Merge Lane
m1 = road.network.get_lane(LaneIndex(WEST_NODE, 'upstream_onramp0_merge_start', 1))
m2 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_start', 'upstream_onramp0_merge_end', 1))
m3 = road.network.get_lane(LaneIndex('upstream_onramp0_merge_end', 'upstream_offramp0_merge_start', 1))
m4 = road.network.get_lane(LaneIndex('upstream_offramp0_merge_start', 'upstream_offramp0_merge_end', 1))

# %% Cars
car1 = Car.make_on_lane(road, lane1, longitudinal=0)
car2 = Car.make_on_lane(road, lane2, longitudinal=0)
car3 = Car.make_on_lane(road, lane3, longitudinal=0)
car4 = Car.make_on_lane(road, lane3, longitudinal=lane3.length)

road.cars.add(car1)
road.cars.add(car2)
road.cars.add(car3)
road.cars.add(car4)

print(f'{car1.position} {lane1}')
print(f'{car2.position} {lane2}')
print(f'{car3.position} {lane3}')
print(f'{car4.position} {lane3}')
# %%
car5 = Car.make_on_lane(road, m1, longitudinal=3)
road.cars.add(car5)
print(f'{car5.position} {car5.lane}')

# %%
car6 = Car.make_on_lane(road, m4, longitudinal=10)
road.cars.add(car6)

print(f'{car6.position} {car6.lane}')

# %%


# %%
car5.distance_to_end_of_lane

# %%
car5.plan_route_to(EAST_NODE)
car5.route

# %%
car5.plan_route_to('upstream_offramp1_converge_end')
car5.route

# %%

first_car = road.first_car_on_lane(m2)
s = m2.local_coordinates(first_car.position)[0]
s

# %%
origin_lane = m1

cars_on_origin = []

for c in road.cars:
    target_lane = c.lane
    if road.is_same_lane(origin_lane, target_lane):
        cars_on_origin.append(c)

for c in cars_on_origin:
    print(c.position)
# %%
utils.is_leading_to_road(lane1.lane_index, lane2.lane_index, same_lane=False)

# %%
road.network.is_connected_road(lane_index_1=car1.lane.lane_index,
                               lane_index_2=car2.lane.lane_index,
                               route=car1.route,
                               same_lane=True,
                               depth=4)

# %%


# %%
route = [lane1.lane_index, lane2.lane_index, lane3.lane_index]
road.network.position_heading_along_route(route=route,
                                          longitudinal=0,
                                          lateral=0)

# %%


# %%
utils.is_same_road(lane1.lane_index, lane2.lane_index)

# %%
utils.is_leading_to_road(lane1.lane_index, lane2.lane_index)

# %%
print(lane2.lane_index)
print(lane3.lane_index)

# %%
same_lane = False
same_road = lane2.lane_index[:2] == lane3.lane_index[:2]
print(same_road)

# %%
same_lane = same_road and (not same_lane or lane1.lane_index.index == lane2.lane_index.index)
same_lane
# %%


# %%

road.network.next_lane(current_lane=lane1)

# %%
road_graphics = RoadGraphics(road=road)
road_graphics.draw(surface=road_surface)
road.cars.draw(road_surface)
screen.blit(road_surface, (0, 0))
pygame.image.save(screen, "image.jpg")

# %%
car4.plan_route_to(EAST_NODE)


# %%

def bfs(start, goal):
    queue = [(start, [start])]

    while queue:

        (node, path) = queue.pop(0)

        if node not in road.network.graph:
            yield []

        for _next in set(road.network.graph[node].keys()) - set(path):

            if _next == goal:
                yield path + [_next]

            elif _next in road.network.graph:
                queue.append((_next, path + [_next]))


def shortest_path(start, goal):
    return next(bfs(start, goal), [])


# %%
start = lane4.destination
print(start)

goal = 'east'
print(goal)

# %%
path = shortest_path(start, goal)
path

# %%
start_lane = lane4
index = start_lane.index

route_lanes = [start_lane]
for i in range(len(path) - 1):
    print(f'{path[i]} -> {path[i + 1]}')
    next_lane = road.network.get_lane(origin=path[i], destination=path[i + 1], index=index)
    print(next_lane)
    print()
    route_lanes.append(next_lane)

route_lanes

# %%
lane = lane4
route = [lane] + [road.network.get_lane(path[i], path[i + 1], lane.index)
                  for i in range(len(path) - 1)]

route

# %%
t = {
    '1': 1,
    '2': 2,
    '3': 3
}

len(t)

# %%


# %%


# %%


# %%
from gym.utils import seeding

seed = 42
np_random, seed = seeding.np_random(seed)

np_random.rand()

# %%
# e.g. [0,2]
np_random.choice(range(4), size=2, replace=False)

# %%
from simulation.lane_change import Mobil

Mobil.GAIN_THRESHOLD

# %%
node = 'upstream_onramp1_merge_start'

# %%

'onramp' in node and 'merge' in node
