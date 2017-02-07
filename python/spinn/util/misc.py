import numpy as np


class Accumulator(object):
    """Accumulator. Makes it easy to keep a list of metrics."""

    cache = dict()

    def __init__(self, trail=100):
        self.trail = trail

    def add(self, key, val):
        self.cache.setdefault(key, deque(maxlen=self.trail)).append(val)

    def get(self, key):
        ret = self.cache.get(key, [])
        try:
            del self.cache[key]
        except:
            pass
        return ret

    def get_avg(self, key):
        return np.array(self.get(key)).mean()
