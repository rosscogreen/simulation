class Metrics(object):
    pass

class Detector(object):

    def __init__(self, x: float, agg_period: int):
        self.x = x
        self.agg_period = agg_period
        self.steps = 0
        self.time = 0
        self.car_count_history = [0]
        self.car_speed_history = [0]
        self.lane_count_history = [4]

        self.car_count = 0
        self.speed_sum = 0

        self.checked = set()

    def add_car(self, car):
        if id(car) not in self.checked:
            self.car_count += 1
            self.speed_sum += car.speed
            self.checked.add(id(car))

    def add_step(self, lane_count):
        self.steps += 1

        self.car_count_history.append(self.car_count)
        self.car_speed_history.append(self.speed_sum)
        self.lane_count_history.append(lane_count)

        obs = {
            'v': self.speed_sum,
            'n': self.car_count,
            'l': lane_count
        }

        self.speed_sum = 0
        self.car_count = 0

        self.checked.clear()

        return obs



