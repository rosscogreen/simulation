import numpy as np

from simulation.constants import CAR_LENGTH
from simulation.idm import IDM
from simulation.lanes import AbstractLane


class Mobil(object):
    B_SAFE = 5.0
    """ Max Safe Braking Deceleration """

    B_SAFE_MAX = 20.0
    """ Max Emergency Braking Deceleration """

    P = 0.1
    """ Politeness Factor """

    MIN_INCENTIVE_THRESHOLD = 0.3
    """ Min Acceleration Gain For Lane Change """

    MIN_GAP = 1.0 + CAR_LENGTH
    """ Minimum Gap Needed For Lane Change """

    @classmethod
    def should_change(cls,
                      lane: "AbstractLane",
                      me: "Car",
                      fwd_old: "Car",
                      bwd_old: "Car",
                      fwd_new: "Car",
                      bwd_new: "Car",
                      right_bias: float = 0.0,
                      priority_car=None) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :param fwd_old: car in front on cars current lane
        :param bwd_old: car behind on cars current lanes
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?

        calc_acc = IDM.calc_acceleration2

        is_right = 1 if lane.index[2] > me.lane.index[2] else -1
        bias = is_right * right_bias

        # Is Enough Gap

        if fwd_new is not None:
            fwd_gap = me.lane_distance_to(fwd_new, lane)
            if fwd_gap < cls.MIN_GAP:
                return False

        if bwd_new is not None:
            bwd_gap = bwd_new.lane_distance_to(me, lane)
            if bwd_gap < cls.MIN_GAP:
                return False

        # Safety Criterion

        me_new_acc = calc_acc(bwd=me, fwd=fwd_new)
        if me_new_acc < -cls.B_SAFE:
            return False

        new_bwd_new_acc = calc_acc(bwd=bwd_new, fwd=me)
        if new_bwd_new_acc < -cls.B_SAFE:
            return False

        if bias >= 1:
            return True

        me_old_acc = calc_acc(bwd=me, fwd=fwd_old)
        if priority_car is not None:
            me_old_acc_priority = calc_acc(bwd=me, fwd=priority_car)
            me_old_acc = min(me_old_acc, me_old_acc_priority)

        new_bwd_old_acc = calc_acc(bwd=bwd_new, fwd=fwd_new)
        old_bwd_new_acc = calc_acc(bwd=bwd_old, fwd=fwd_old)
        old_bwd_old_acc = calc_acc(bwd=bwd_old, fwd=me)

        self_gain = me_new_acc - me_old_acc
        other_gain = cls.P * ((new_bwd_new_acc - new_bwd_old_acc) + (old_bwd_new_acc - old_bwd_old_acc))
        incentive = self_gain + other_gain + bias

        if incentive < cls.MIN_INCENTIVE_THRESHOLD:
            return False

        # All clear to change lanes
        return True
