import numpy as np

SECONDS_PER_HOUR = 3600


class Experiment(object):
    """ Experiment Configuration """

    """  Number of steps to perform """
    DURATION = 100

    """ Number of steps per second [Hz] """
    # e.g. 20 steps per seconds -> dt = 0.05 sec per step
    SIMULATION_FREQUENCY = 20  #

    """ Number of actions to perform per second """
    POLICY_FREQUENCY = 0.2

    """ What does 1 second represent """
    TIME_WARP = 300 # 5 min

    """ Number of periods  """
    NUM_PERIODS = 2 * np.pi

    """ Requested inflow in vehicles per hour """
    TOTAL_DEMAND = 20000

    """ Minimum demand level per direction """
    MIN_DEMAND = 2000
