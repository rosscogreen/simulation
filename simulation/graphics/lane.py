from simulation.custom_types import UL, CL, NL, SL
import numpy as np
from simulation.lanes import AbstractLane
import pygame
from simulation.graphics.common import WHITE


class LaneGraphics(pygame.sprite.Sprite):
    STRIPE_SPACING = 5.0
    STRIPE_LENGTH = 3.0
    STRIPE_WIDTH = 0.3

    def __init__(self, lane: "AbstractLane"):
        super(LaneGraphics, self).__init__()
        self.lane = lane
        self.surface = None
        self.line_width = 1

    def draw(self, surface):
        self.surface = surface
        self.line_width = max(surface.pix(self.STRIPE_WIDTH), 1)
        self._draw(self.lane.left_line, 0)
        self._draw(self.lane.right_line, 1)

    def _draw(self, line_type, side_idx):

        if line_type == NL:
            return

        r = (side_idx - 0.5) * self.lane.width  # -2 or +2

        if line_type == CL:

            x1, y1 = self.lane.start_pos
            x2, y2 = self.lane.end_pos

            start_pos = [x1, y1 + r]
            end_pos = [x2, y2 + r]

            self.draw_line(start_pos=start_pos, end_pos=end_pos)

        else:
            s_start = self.lane.local_coordinates(self.lane.start_pos)[0]
            s_end = self.lane.local_coordinates(self.lane.end_pos)[0]
            stripe_count = abs(s_end - s_start) // self.STRIPE_SPACING
            stripe_starts = s_start + np.arange(stripe_count) * self.STRIPE_SPACING
            stripe_length = self.STRIPE_LENGTH if line_type == SL else self.STRIPE_SPACING
            starts = [self.lane.position(s, r) for s in stripe_starts]
            ends = [self.lane.position(s + stripe_length, r) for s in stripe_starts]
            for start_pos, end_pos in zip(starts, ends):
                self.draw_line(start_pos=start_pos, end_pos=end_pos)

    def draw_line(self, start_pos, end_pos, color=WHITE):
        pygame.draw.line(self.surface,
                         color,
                         self.surface.vec2pix(start_pos),
                         self.surface.vec2pix(end_pos),
                         self.line_width)
