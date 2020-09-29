class Detector(object):

    def __init__(self):
        self.car_count = 0
        self.speed_sum = 0
        self.checked = set()

    def update(self, car):
        if id(car) not in self.checked:
            self.car_count += 1
            self.speed_sum += car.speed
            self.checked.add(id(car))

    def report(self, period):
        flow = self.car_count * (3600 / period) if period else 0
        speed = (self.speed_sum / self.car_count) * 3.6 if self.car_count else 0
        obs = {
            'v':     self.speed_sum,
            'n':     self.car_count,
            'flow':  flow,
            'speed': speed
        }

        self.speed_sum = 0
        self.car_count = 0

        self.checked.clear()

        return obs


class Detectors(object):
    detection_points = [10, 120, 250, 380, 490]

    def __init__(self, margin=1):
        self.upstream_detectors = {x: Detector() for x in self.detection_points}
        self.downstream_detectors = {x: Detector() for x in self.detection_points}
        self.x_ranges = [(x - margin, x + margin, x) for x in self.detection_points]

    def update(self, car):
        s = car.lane.path_coordinates(car.position)[0]
        upstream = car.lane.upstream
        for x1, x2, x in self.x_ranges:
            if x1 <= s <= x2:
                if upstream:
                    self.upstream_detectors[x].update(car)
                else:
                    self.downstream_detectors[x].update(car)

    def report(self, period):
        return {
            **self._report(period, self.upstream_detectors, 'upstream'),
            **self._report(period, self.downstream_detectors, 'downstream')
        }

    def _report(self, period, detectors, prefix):
        result = {}
        for x, detector in detectors.items():
            for k, v in detector.report(period).items():
                result[f'{prefix}_{x}_{k}'] = v
        return result
