import numpy as np
from collections import deque
import json


def debug_gradient(model, losses):
    model.zero_grad()

    for name, loss in losses:
        print(name)
        loss.backward(retain_variables=True)
        stats = [
            (p.grad.norm().data[0],
             p.grad.max().data[0],
             p.grad.min().data[0],
             p.size()) for p in model.parameters()]
        for s in stats:
            print(s)
        print()

        model.zero_grad()


class GenericClass(object):
    def __init__(self, **kwargs):
        super(GenericClass, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = "{}"
        return s.format(self.__dict__)


class Args(GenericClass):
    pass


class Vocab(GenericClass):
    pass


class Example(GenericClass):
    pass


def time_per_token(num_tokens, total_time):
    return sum(total_time) / float(sum(num_tokens))


def count_parse2(parse, index, const_parsed=[]):
    """
    To-do: this is more efficient than  count_parse() but is currently broken, fic it!

    Compute Constituents Parsed metric for ListOps style examples.
    """    
    after = parse[index:]
    before = parse[:index]
    between = after[: after.index("]")]

    o_b = between.count("(") # open, between
    c_b = between.count(")") # close, between

    end = after.index("]")
    cafter = after[end+1:]
    stop = None
    stop_list = []
    for item in cafter:
        stop_list.append(")" == item)
        if stop_list[-1] == False:
            break

    if False in stop_list:
        stop = stop_list.index(False)
    else:
        stop = None
    cafter = cafter[: stop]
    c_a = cafter.count(")")

    stop = None
    stop_list = []
    for item in before[::-1] :
        stop_list.append("(" == item)
        if stop_list[-1] == False:
            break

    if False in stop_list:
        stop = len(before) - stop_list.index(False) - 1
    else:
        stop = None
    cbefore = before[stop:]
    o_a = cbefore.count("(")

    ints = sum(c.isdigit() for c in between) + between.count("-")
    op = o_a + o_b
    cl = c_a + c_b

    if op >= ints and cl >= ints:
        if op == ints+1 or cl == ints+1:
            const_parsed.append(1)

    parse[index - o_a : index + len(between) + 1 + c_a] = '-'
            
    mathops = ["[MAX", "[MIN", "[MED", "[SM"]
    some_ops = list(set(mathops) & set(parse))
    ops_i = [parse[::-1].index(m) for m in some_ops]
    if len(ops_i) != 0:
        op_i = min([i for i in ops_i if i >= 0])
        index = len(parse) - op_i - 1
        count_parse2(parse, index, const_parsed)
        
    return sum(const_parsed)

def count_parse(parse, index, const_parsed=[]):
    """
    Compute Constituents Parsed metric for ListOps style examples.
    """
    mathops = ["[MAX", "[MIN", "[MED", "[SM"]
    if "]" in parse:
        after = parse[index:]
        before = parse[:index]
        between = after[: after.index("]")]
        
        nest_check = [m in between[1:] for m in mathops]
        if True in nest_check:
            op_i = nest_check.index(True)
            nested_i = after[1:].index(mathops[op_i]) + 1
            nested = after[nested_i : ]
            c = count_parse(parse, index+nested_i, const_parsed)
            cc = count_parse(parse, index, const_parsed)
        else:
            o_b = between.count("(") # open, between
            c_b = between.count(")") # close, between
                
            end = after.index("]")
            cafter = after[end+1:]
            stop = None
            stop_list = []
            for item in cafter:
                stop_list.append(")" == item)
                if stop_list[-1] == False:
                    break

            if False in stop_list:
                stop = stop_list.index(False)
            else:
                stop = None
            cafter = cafter[: stop]
            c_a = cafter.count(")")

            stop = None
            stop_list = []
            for item in before[::-1] :
                stop_list.append("(" == item)
                if stop_list[-1] == False:
                    break

            if False in stop_list:
                stop = len(before) - stop_list.index(False) - 1
            else:
                stop = None
            cbefore = before[stop:]
            o_a = cbefore.count("(")

            ints = sum(c.isdigit() for c in between) + between.count("-")
            op = o_a + o_b
            cl = c_a + c_b

            if op >= ints and cl >= ints:
                if op == ints+1 or cl == ints+1:
                    const_parsed.append(1)
            parse[index - o_a : index + len(between) + 1 + c_a] = '-'
    return sum(const_parsed)

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
                   sent2_transitions=None,
                   sent1_trees=None,
                   sent2_trees=None,
                   cp_metric=False):
        '''Saves a batch. Transforms the batch from column-centric information
        (information split by columns) to row-centric (by EvalSentence).'''
        cp = 0
        cp_max = 0
        mathops = ["[MAX", "[MIN", "[MED", "[SM"]

        b = [preds.view(-1), target.view(-1), example_ids, output]
        for i, (pred, truth, eid, output) in enumerate(zip(*b)):
            sent = {}
            sent['example_id'] = eid
            sent['prediction'] = pred
            sent['truth'] = truth
            sent['output'] = [str(output_val) for output_val in output]
            if sent1_transitions is not None:
                sent['sent1_transitions'] = sent1_transitions[i].tolist()
            if sent2_transitions is not None:
                sent['sent2_transitions'] = sent2_transitions[i].tolist()
            if sent1_trees is not None:
                sent['sent1_tree'] = sent1_trees[i]
                if cp_metric:
                    parse =  sent1_trees[i].split()
                    some_ops = list(set(mathops) & set(parse))
                    ops_i = [parse.index(m) for m in some_ops]
                    op_i = min([i for i in ops_i if i >= 0])
                    cp_max += parse.count("]")
                    cp += count_parse(parse, op_i, const_parsed=[])
            if sent2_trees is not None:
                sent['sent2_tree'] = sent2_trees[i]

            self.report.append(sent)

        if cp_metric:
            return cp, cp_max

    def write_report(self, filename):
        '''Commits the report to a file.'''
        with open(filename, 'w') as f:
            for example in self.report:
                json.dump(example, f, sort_keys=True)
                f.write('\n')


def PrintParamStatistics(name, param):
    data = param.data.cpu().numpy()
    print(name, end=' ')
    print("Mean:", np.mean(data), end=' ')
    print("Std:", np.std(data), end=' ')
    print("Min:", np.min(data), end=' ')
    print("Max:", np.max(data))


def recursively_set_device(inp, gpu):
    if hasattr(inp, 'keys'):
        for k in list(inp.keys()):
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
