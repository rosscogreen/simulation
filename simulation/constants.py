from numpy import pi

# Car
CAR_LENGTH = 3
CAR_WIDTH = 3

HALF_CAR_LENGTH = CAR_LENGTH / 2

# Kinematics

DEFAULT_SPEEDS = [23, 25]
MAX_SPEED = 40.
#MIN_SPEED = 0

MIN_INSERTION_GAP = 2 + CAR_LENGTH

LANE_CHANGE_DELAY = 1.0

TAU_A = 0.6  # [s]
TAU_DS = 0.1  # [s]
PURSUIT_TAU = 0.8 * TAU_DS  # [s]
KP_A = 1 / TAU_A
KP_HEADING = 1 / TAU_DS
KP_LATERAL = 1 / 3 * KP_HEADING  # [1/s]
MAX_STEERING_ANGLE = pi / 3  # [rad]

"""Maximum acceleration."""
ACC_MAX = 8.0  # [m/s2]
