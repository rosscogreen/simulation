import numpy as np

from simulation import utils
from simulation.constants import CAR_LENGTH


class IDM(object):
    s0 = 2.0 + CAR_LENGTH
    """ Minimum distance [m] """

    T = 1.0
    """ Safe time headway [s] """

    a = 2.0
    """ Max acceleration [m/s^2] """

    b = 4.0
    """ Comfortable Deceleration [m/s^2] """

    b_max = 8.0

    b_emergency = 18.0

    DELTA = 4.0
    """ Acceleration exponent """

    AB_TERM = 2 * np.sqrt(a * b)

    TAU_A = 0.6  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 0.6 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 3 * KP_HEADING  # [1/s]

    ROUND_VEL = 1e-2

    @classmethod
    def d_star(cls, bwd: "Car", fwd: "Car"):
        v0 = max(bwd.speed, 0)
        v1 = max(fwd.speed, 0)
        dv = v0 - v1
        vdv = v0 * dv

        vT = v0 * cls.T
        v_fwd = vdv / cls.AB_TERM

        s = max(0, vT * v_fwd)

        return cls.s0 + s

    @classmethod
    def calc_acceleration(cls, bwd: "Car", fwd: "Car" = None) -> float:
        if bwd is None:
            return 0

        v0 = max(bwd.speed, 0)
        v_star = utils.not_zero(getattr(bwd, "target_speed", 0))

        acc = cls.a * (1 - np.power(v0 / v_star, cls.DELTA)) if v0 <= v_star else cls.a * (1 - v0/v_star)

        b = cls.b

        noise = 0

        if fwd:
            same_lane = bwd.lane.index == fwd.lane.index
            s_gap = bwd.lane_distance_to(fwd)

            if not same_lane:
                is_on_left = 1 if fwd.lane.index[2] < bwd.lane.index[2] else -1
                bias = fwd.right_bias * is_on_left

                if bias >= 1:
                    b = b/2
                else:
                    # if bwd.lane.is_onramp and bwd.s_remaining_on_lane < 15:
                    #     return -cls.b_max
                    # elif fwd.lane.is_onramp and fwd.s_remaining_on_lane < 15:
                    #     return -cls.b_max

                    if bwd.lane.is_onramp and bwd.s_remaining_on_lane < 15:
                        b = b*2
                    elif fwd.lane.is_onramp and fwd.s_remaining_on_lane < 15:
                        b = b*2

                if s_gap < cls.s0:
                    noise = 0.3

            else:
                if s_gap < cls.s0:
                    return -cls.b_max

            s_gap = utils.not_zero(s_gap)
            s_star = cls.d_star(bwd, fwd)

            interactive_term = s_star ** 2 / s_gap ** 2
            acc -= cls.a * interactive_term

            acc_rnd = noise * (np.random.random() - 0.5)
            acc += acc_rnd

        return max(-b, acc)

    @classmethod
    def calc_max_initial_speed(cls, gap: float, fwd_speed: float, speed_limit: float = None) -> float:
        """
        Compute the maximum allowed speed to avoid Inevitable Collision States.

        Assume the front vehicle is going to brake at full deceleration and that
        it will be noticed after a given delay, and compute the maximum speed
        which allows the ego-vehicle to brake enough to avoid the collision.

        :param gap: the distance to the next car
        :param speed: the speed of the next car
        :return: the maximum allowed speed
        """

        a = cls.a
        a2 = a ** 2

        d = gap

        delta = 4 * (a2 * cls.T) ** 2 + 8 * a * (a2) * d + 4 * a2 * fwd_speed ** 2

        v_max = -a * cls.T + np.sqrt(delta) / (2 * a)

        if speed_limit is not None:
            v_max = min(v_max, speed_limit)

        return v_max

    @classmethod
    def speed_control_acceleration(cls, actual_speed, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return cls.KP_A * (target_speed - actual_speed)
