"""
Utility for plotting the various costs and accuracies vs training iteration no. Reads these values from
a log file. Can also be used to compare multiple logs by supplying multiple paths.

Example Log:
17-02-11 23:06:46 [1] Step: 100 Acc: 0.38344  0.00000 Cost: 1.00246 0.99532 0.00000 0.00714 Time: 0.00024
17-02-11 23:06:47 [1] Step: 100 Eval acc: 0.486055   0.000000 ../../spinn/snli_1.0/snli_1.0_dev.jsonl Time: 0.000007
"""

import gflags
import matplotlib.pyplot as plt
import sys

FLAGS = gflags.FLAGS

class TrainLine(object):
    def __init__(self, line):
        tokens = line.split()
        parts = [
        'date', 'time', '_',
        '_','step', '_', 'acc', 'transition_acc',
        '_', 'total_cost', 'xent_cost', 'transition_cost', 'l2_cost',
        '_', 'time_per_example',
        ]

        assert len(tokens) == len(parts)

        for t, k in zip(tokens, parts):
            if k == '_': continue
            setattr(self, k, t)

class EvalLine(object):
    def __init__(self, line):
        tokens = line.split()
        parts = [
        'date', 'time', '_',
        '_','step', '_', '_', 'acc', 'transition_acc',
        '_', '_', 'time_per_example',
        ]

        assert len(tokens) == len(parts)

        for t, k in zip(tokens, parts):
            if k == '_': continue
            setattr(self, k, t)


def is_train(line):
    return 'Acc' in line


def is_eval(line):
    return 'Eval' in line


def read(log_path):
    l_train = []
    l_eval = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()

            if is_train(line):
                l_train.append(TrainLine(line))
            elif is_eval(line):
                l_eval.append(EvalLine(line))
    return l_train, l_eval


def Analyze():
    log_paths = FLAGS.path.split(',')
    for lp in log_paths:
        l_train, l_eval = read(lp)
        max_eval = max([e.acc for e in l_eval]) if len(l_eval) > 0 else 0.0
        print(lp, "max eval:", max_eval)


if __name__ == '__main__':
  
    gflags.DEFINE_string("path", None, "Path to log file")
    gflags.DEFINE_string("index", "1", "csv list of corpus indices. 0 for train, 1 for eval set 1 etc.")
    gflags.DEFINE_boolean("pred_acc", True, "Prediction accuracy")
    gflags.DEFINE_boolean("parse_acc", False, "Parsing accuracy")
    gflags.DEFINE_boolean("total_cost", False, "Total cost, valid only if index == 0")
    gflags.DEFINE_boolean("xent_cost", False, "Cross entropy cost, valid only if index == 0")
    gflags.DEFINE_boolean("l2_cost", False, "L2 regularization cost, valid only if index == 0")
    gflags.DEFINE_boolean("action_cost", False, "Action cost, valid only if index == 0")
    gflags.DEFINE_boolean("legend", False, "Show legend in plot")
    gflags.DEFINE_boolean("subplot", False, "Separate plots for each log")
    gflags.DEFINE_string("ylabel", "", "Label for y-axis of plot")
    gflags.DEFINE_integer("iters", 10000, "Iters to limit plot to")

    FLAGS(sys.argv)
    assert FLAGS.path is not None, "Must provide a log path"
    Analyze()
  
