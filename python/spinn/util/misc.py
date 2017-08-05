import numpy as np
from collections import deque
import jsonl
import os
import logging_pb2 as pb


class GenericClass(object):
    def __init__(self, **kwargs):
        super(GenericClass, self).__init__()
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def __repr__(self):
        s = "{}"
        return s.format(self.__dict__)


class Args(GenericClass):
    pass


class EncodeArgs(GenericClass):
    pass


class Vocab(GenericClass):
    pass


class Example(GenericClass):
    pass


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
            except BaseException:
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
        super(EvalReporter, self).__init__()
        self.report = []

    def save_batch(self,
                   preds,
                   target,
                   example_ids,
                   output,
                   sent1_transitions=None,
                   sent2_transitions=None):
        '''Saves a batch. Transforms the batch from column-centric information
        (information split by columns) to row-centric (by EvalSentence).'''

        b = [preds.view(-1), target.view(-1), example_ids, output]
        for i, (pred, truth, eid, output) in enumerate(zip(*b)):
            sent = {}
            sent['sentence_id'] = eid
            sent['prediction'] = pred
            sent['truth'] = truth
            sent['output'] = list(output)
            if sent1_transitions is not None:
                sent['sent1_transitions'] = list(sent1_transitions[i])
            if sent2_transitions is not None:
                sent['sent2_transitions'] = list(sent2_transitions[i])
            self.report.append(sent)

    def write_report(self, filename):
        '''Commits the report to a file. Can be written as textproto or binary.
        Binary is more space-efficient but can't be inspected manually from
        a text editor.'''
        else:
            with open(filename, 'w') as f:
                f.write(str(self.report))


def PrintParamStatistics(name, param):
    data = param.data.cpu().numpy()
    print name,
    print "Mean:", np.mean(data),
    print "Std:", np.std(data),
    print "Min:", np.min(data),
    print "Max:", np.max(data)


def recursively_set_device(inp, gpu):
    if hasattr(inp, 'keys'):
        for k in inp.keys():
            inp[k] = recursively_set_device(inp[k], gpu)
    elif isinstance(inp, list):
        return [recursively_set_device(ii, gpu) for ii in inp]
    elif isinstance(inp, tuple):
        return (recursively_set_device(ii, gpu) for ii in inp)
    elif hasattr(inp, 'cpu'):
        if gpu >= 0:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp
