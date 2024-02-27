from abc import ABC, abstractmethod
import math

class TemperatureScheduler(ABC):
    @abstractmethod
    def get(self, iter):
        pass


class DefaultScheduler(TemperatureScheduler):
    def __init__(self, start, delta) -> None:
        self.start = start
        self.delta = delta

    def get(self, iter):
        return self.start + iter * self.delta        


class CosignScheduler(ABC):
    def __init__(self, minimum, maximum, frequency) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.frequency = frequency

    def get(self, iter):
        return (self.maximum + self.minimum)/2 - (self.maximum - self.minimum)/2 * math.cos(2 * math.pi * iter / self.frequency)