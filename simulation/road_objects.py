import numpy as np
import pygame

from simulation.custom_types import Vector
from simulation.lanes import AbstractLane


class RoadObject(pygame.sprite.Sprite):
    """
    Common interface for objects that appear on the road, beside vehicles.

    For now we assume all objects are rectangular.
    """

    LENGTH = 2.0  # Object length [m]
    HALF_LENGTH = LENGTH / 2
    WIDTH = 10.0  # Object width [m]

    def __init__(self,
                 road: "Road",
                 lane: "AbstractLane",
                 position: Vector):
        super(RoadObject, self).__init__()
        """
        :param road: the road instance where the object is placed in
        :param position: cartesian position of object in the surface
        :param speed: cartesian speed of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        """
        self.road = road
        self.lane = lane
        self.target_lane = lane
        self.position = np.array(position, dtype=np.float)
        self.s = self.lane.s(self.position)
        self.speed = 0
        self.heading = 0
        self.hit = False
        self.image = None

    @classmethod
    def make_on_lane(cls, road: "Road", lane: "AbstractLane", longitudinal: float = 0):
        """
        Create an object on a given lane at a longitudinal position.

        :param road: the road instance where the object is placed in
        :param lane_index: a tuple (origin node, destination node, lane id on the road).
        :param longitudinal: longitudinal position along the lane
        :return: An object with at the specified position
        """
        obstacle = cls(road=road, lane=lane, position=lane.position(longitudinal, 0))
        road.obstacles.add(obstacle)
        road.road_objects.add(obstacle)

    def __str__(self):
        x, y = self.position
        u = 'Up' if self.lane.upstream else 'Down'
        return f"{self.__class__.__name__} #{id(self) % 1000}: at ({round(x, 1)},{round(y, 1)}) on {u}{self.lane.index} {self.lane.origin}->{self.lane.destination}"

    def __repr__(self):
        return self.__str__()


class Obstacle(RoadObject):
    """Obstacles on the road."""

    pass
