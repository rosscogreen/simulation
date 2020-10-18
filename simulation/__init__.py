# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# Import the envs module so that envs register themselves
import simulation.envs

from simulation.car import Car
from simulation.lanes import AbstractLane, SineLane, StraightLane
from simulation.road import Road
from simulation.network import RoadNetwork
from simulation.idm import IDM
from simulation.mobil import Mobil

