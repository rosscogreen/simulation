from simulation.road import Road
from simulation.road_factory.constants import *
from simulation.road_factory.highways import highway
from simulation.road_factory.ramps import onramp, offramp


def create_road():
    road = Road()
    highway(road)

    # Entry Ramps
    for onramp_idx, onramp_start in enumerate(ON_RAMP_STARTS):
        onramp(road=road, x_start=onramp_start, upstream=True, ramp_idx=onramp_idx)
        onramp(road=road, x_start=onramp_start, upstream=False, ramp_idx=onramp_idx)

    # Exit Ramps
    for offramp_idx, offramp_start in enumerate(OFF_RAMP_STARTS):
        offramp(road=road, x_start=offramp_start, upstream=True, ramp_idx=offramp_idx)
        offramp(road=road, x_start=offramp_start, upstream=False, ramp_idx=offramp_idx)

    return road