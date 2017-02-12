import numpy as np
from collections import deque
import os


class GenericClass():

    def __repr__(self):
        s = "{}"
        return s.format(self.__dict__)


class Args(GenericClass): pass


class EncodeArgs(GenericClass): pass


class Vocab(GenericClass): pass


class Example(GenericClass): pass


def time_per_token(num_tokens, total_time):
    return sum(total_time) / float(sum(num_tokens))


class Accumulator(object):
    """Accumulator. Makes it easy to keep a trailing list of statistics."""

    def __init__(self, maxlen=100):
        self.maxlen = maxlen
        self.cache = dict()

    def add(self, key, val):
        self.cache.setdefault(key, deque(maxlen=self.maxlen)).append(val)

    def get(self, key, clear=True):
        ret = self.cache.get(key, [])
        if clear:
            try:
                del self.cache[key]
            except:
                pass
        return ret

    def get_avg(self, key, clear=True):
        return np.array(self.get(key, clear)).mean()


class MetricsLogger(object):
    """MetricsLogger."""

    def __init__(self, metrics_path):
        self.metrics_path = metrics_path

    def Log(self, key, val, step):
        log_path = os.path.join(self.metrics_path, key) + ".metrics"
        with open(log_path, 'a') as f:
            f.write("{} {}\n".format(step, val))
