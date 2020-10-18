class Detector:

    def __init__(self, x):
        self.x = x
        self._n_cars = 0
        self._total_speed = 0
        self.checked = set()

        self.n_cars = 0
        self.total_speed = 0
        self.flow = 0
        self.speed = 0

    def update(self, period):
        self.n_cars = self._n_cars
        self.total_speed = self._total_speed
        self.flow = self._n_cars / period if period else 0
        self.speed = (self._total_speed / self._n_cars) * 3.6 if self.n_cars else 0
        self.reset()

    def update_car(self, car):
        _id = id(car)
        if _id not in self.checked:
            self._n_cars += 1
            self._total_speed += car.speed
            self.checked.add(_id)

    def reset(self):
        self._total_speed = 0
        self._n_cars = 0
        self.checked.clear()


class Detectors(object):
    detection_points = [10, 120, 250, 380, 490]

    def __init__(self, margin=1):
        self.upstream_detectors = {x: Detector(x) for x in self.detection_points}
        self.downstream_detectors = {x: Detector(x) for x in self.detection_points}
        self.x_ranges = [(x - margin, x + margin, x) for x in self.detection_points]
        self.n = len(self.detection_points)

    def update(self, period):
        for detector in self.upstream_detectors.values():
            detector.update(period)
        for detector in self.downstream_detectors.values():
            detector.update(period)

    def update_car(self, car):
        s = car.s_path
        upstream = car.lane.upstream

        for x1, x2, x in self.x_ranges:
            if x1 <= s <= x2:
                if upstream:
                    self.upstream_detectors[x].update_car(car)
                else:
                    self.downstream_detectors[x].update_car(car)
