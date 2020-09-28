import pygame
from .common import WorldSurface
from typing import List
from simulation.car import Car
from numpy import deg2rad, rad2deg
from simulation.graphics.common import YELLOW

MIN_ANGLE = deg2rad(2)

LOADED_IMAGE = None

def load_image(sprite: "Car", surface: "WorldSurface"):
    global LOADED_IMAGE

    width = surface.pix(sprite.LENGTH)
    height = surface.pix(sprite.WIDTH)

    path = getattr(sprite, 'IMAGE_PATH', None)

    if LOADED_IMAGE is None and path is not None:
        LOADED_IMAGE = pygame.image.load(str(path)).convert()

    if LOADED_IMAGE is not None:
        vehicle_surface = pygame.Surface((width, width), pygame.SRCALPHA)
        vehicle_surface.blit(
                source=pygame.transform.scale(LOADED_IMAGE, (width, height)),
                dest=(0, (width / 2) - (height / 2))
        )
    else:
        vehicle_surface = pygame.Surface((sprite.WIDTH, sprite.WIDTH), pygame.SRCALPHA)
        vehicle_surface.fill(YELLOW)

    sprite.image = vehicle_surface
    sprite.rect = vehicle_surface.get_rect()

class Cars2(list):

    def add(self, car):
        self.append(car)

    def kill(self, car):
        self.remove(car)

    def update(self, dt: float):
        for c in self:
            c.update(dt)

    def draw(self, surface):
        for c in self:
            if c.image is None:
                load_image(c, surface)

            x, y = c.position
            rect = surface.pos2pix(x - c.HALF_LENGTH, y - c.HALF_LENGTH)

            # Rotate surface to match vehicle rotation
            rotate_angle = rad2deg(-c.heading if abs(c.heading) > MIN_ANGLE else 0)
            rotated = pygame.transform.rotate(c.image, rotate_angle)

            surface.blit(rotated, rect)

class Cars(pygame.sprite.RenderUpdates):

    def draw(self, surface: WorldSurface):
        dirty = self.lostsprites
        self.lostsprites = []

        for sprite in self.sprites():

            if sprite.image is None:
                load_image(sprite, surface)

            sprite_rect = self.spritedict[sprite]

            x, y = sprite.position
            rect = surface.pos2pix(x - sprite.LENGTH / 2, y - sprite.LENGTH / 2)

            # Rotate surface to match vehicle rotation
            rotate_angle = rad2deg(-sprite.heading if abs(sprite.heading) > MIN_ANGLE else 0)
            rotated = pygame.transform.rotate(sprite.image, rotate_angle)

            newrect = surface.blit(rotated, rect)
            sprite.rect = newrect

            if sprite_rect:
                if newrect.colliderect(sprite_rect):
                    dirty.append(newrect.union(sprite_rect))
                else:
                    dirty.append(newrect)
                    dirty.append(sprite_rect)
            else:
                dirty.append(newrect)

            self.spritedict[sprite] = newrect

        return dirty