from simulation.graphics.common import GREY
from simulation.graphics.common import WorldSurface
from simulation.graphics.lane import LaneGraphics


class RoadGraphics(object):

    def __init__(self, road):
        self.road = road

    def draw(self, surface: "WorldSurface"):
        if self.road.redraw:
            surface.fill(GREY)
            for lane in self.road.network:
                lane_graphics = LaneGraphics(lane)
                lane_graphics.draw(surface)
            self.road.redraw = False
