"""

Investigate common correct examples to estimate potential
upside of using an ensemble.

"""
import gflags
import numpy as np
import pandas as pd
import sys

FLAGS = gflags.FLAGS


def Read(fn):
    return pd.read_csv(fn, delimiter=' ', names=['example_id', 'correct', 'truth', 'pred', 'sent1_parse', 'sent2_parse'])


def Compare(reports):
    assert len(reports) > 1
    assert all(r1.shape == r2.shape for r1, r2 in zip(reports[:-1], reports[1:]))
    filters = []
    for r1 in reports:
        filters.append(r1[r1.correct == True])
    intersect = np.zeros(reports[0].shape[0])
    for f1 in filters:
        for i in f1.index:
            intersect[i] += 1
    common_correct = (intersect == len(reports)).sum()
    upside = (intersect > 0).sum() - common_correct

    return common_correct, upside


def Analyze():
    report_files = FLAGS.path.split(',')

    # Read CSV reports
    reports = []
    for i, fn in enumerate(report_files):
        rep = Read(fn)
        rep.id = i
        reports.append(rep)

    # Compare Every Pair
    common = []
    for i1, rep1 in enumerate(reports):
        for i2, rep2 in enumerate(reports):
            if i2 > i1:
                common_correct, upside = Compare([rep1, rep2])
                common.append((i1, i2, common_correct, upside))

    # Compare Whole
    all_common_correct, all_upside = Compare(reports)

    # Print Report
    for i, (fn, report) in enumerate(zip(report_files, reports)):
        print("{:3} {:6}: {}".format(i, report.correct.sum(), fn))
    print
    total = reports[0].shape[0]
    best = max(reports, key=lambda x: x.correct.sum())
    for i1, i2, c, u in common:
        print("{:3} {:3}: {:6} {:6} {:6} {:6.5f}".format(i1, i2, c, u, c + u, (c+u)/float(total)))
    print
    print("    all: {:6} {:6} {:6} {:6.5f}".format(all_common_correct, all_upside, all_common_correct + all_upside, (all_common_correct + all_upside)/float(total)))
    print
    print("Best:\t{:6} {:6.5f} {}".format(best.correct.sum(), best.correct.sum()/float(total), best.id))
    print("Total:\t{:6}".format(total))

if __name__ == '__main__':

    gflags.DEFINE_string("path", None, "Path to report file")

    FLAGS(sys.argv)
    assert FLAGS.path is not None, "Must provide a report path"
    Analyze()

