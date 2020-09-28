#%% imports
import sys
sys.path.insert(0, '../..')

from simulation.road_factory import constants as c
from simulation.road_factory.roads import create_road
from simulation.custom_types import LaneIndex

#%%
road = create_road()

#%%
lane1 = road.network.get_lane(LaneIndex(c.WEST_NODE, c.EAST_NODE, 2))
lane2 = road.network.get_lane(LaneIndex(c.EAST_NODE, c.WEST_NODE, 6))

#%%
