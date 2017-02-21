import numpy as np
from collections import deque
import os


class GenericClass(object):
    def __init__(self, **kwargs):
        super(GenericClass, self).__init__()
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

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

    def __init__(self, maxlen=None):
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


class EvalReporter(object):
    def __init__(self):
        self.batches = []

    def save_batch(self, preds, target, example_ids, output, sent1_preds=None, sent2_preds=None):
        sent1_preds = sent1_preds if sent1_preds is not None else [None] * len(example_ids)
        sent2_preds = sent2_preds if sent2_preds is not None else [None] * len(example_ids)
        batch = [preds.view(-1), target.view(-1), example_ids, output, sent1_preds, sent2_preds]
        self.batches.append(batch)

    def write_report(self, filename):
        with open(filename, 'w') as f:
            for b in self.batches:
                for bb in zip(*b):
                    pred, truth, eid, output, sent1_preds, sent2_preds = bb
                    report_str = "{eid} {correct} {truth} {pred} {output}"
                    if sent1_preds is not None:
                        report_str += " {sent1_preds}"
                    if sent2_preds is not None:
                        report_str += " {sent2_preds}"
                    report_str += "\n"
                    report_dict = {
                        "eid": eid,
                        "correct": truth == pred,
                        "truth": truth,
                        "pred": pred,
                        "output": " ".join([str(o) for o in output]),
                        "sent1_preds": '{}'.format("".join(str(t) for t in sent1_preds)) if sent1_preds is not None else None,
                        "sent2_preds": '{}'.format("".join(str(t) for t in sent2_preds)) if sent2_preds is not None else None,
                    }
                    f.write(report_str.format(**report_dict))
