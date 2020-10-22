from simulation.agents.basic import Agent
from typing import Tuple

def try_divide(a,b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0

class HeuristicAgent(Agent):

    def __init__(self, env, min_gap=0.1):
        self.env = env
        self.min_gap = min_gap

    def predict(self, state, deterministic=True) -> Tuple:
        env = self.env
        demand_up, demand_down = env._demand.demand_for_step
        nlanes_up = env.road.upstream_lane_count
        nlanes_down = env.road.downstream_lane_count

        avg_demand_up = try_divide(demand_up, nlanes_up)
        avg_demand_down = try_divide(demand_down, nlanes_down)
        gap = try_divide((avg_demand_down - avg_demand_up), (avg_demand_up + avg_demand_down))

        if gap > self.min_gap and nlanes_down < 7:
            action = env.ADD_DOWNSTREAM
        elif gap < -self.min_gap and nlanes_up < 7:
            action = env.ADD_UPSTREAM
        else:
            action = env.HOLD

        #print(gap, avg_demand_up, avg_demand_down, action)

        return action, None
