"""

The old version of this script is here:
https://github.com/mrdrozdov/spinn/blob/2a06fc489faf44796dcb88a57afd699ba9937cfa/scripts/analyze_trees.py

Usage:
$ mkdir example_trees
$ python scripts/analyze_report_trees.py \
    --data_path ./snli_1.0/snli_1.0_dev.jsonl \
    --report_path ./checkpoints/example-snli.report \
    --out_path ./example_trees

"""
from __future__ import print_function

import gflags
import numpy as np
import pandas as pd
import sys
import pydot
import argparse
from tqdm import tqdm

from spinn.data.listops import load_listops_data


FLAGS = gflags.FLAGS


def create_tree_ascii(words, transitions):
    buf = list(reversed(words))
    stack = []
    for i, t in enumerate(transitions):
        if t == 0:
            stack.append([buf.pop()])
        elif t == 1:
            r = stack.pop()
            l = stack.pop()
            stack.append(['('] + l + r + [')'])
    return stack[-1]


def newline(depth):
    print('\n' + ' '*depth, end='')


def ascii_tree_prettyprint(tree):
    print()
    ind = 0
    for i, sym in enumerate(tree):
        if sym == '(':
            if tree[i - 1] != '(':
                print(' ', end='')
            print(sym, end='')
            ind += 1
        elif sym == ')':
            print(sym, end='')
            ind -= 1
            newline(ind)
        else:
            print(sym, end='')
            if tree[i + 1] != ')':
                newline(ind)



def print_tree(words, transitions):
    ascii_tree_prettyprint(create_tree_ascii(words, transitions))


def read_report(fn):
    return pd.read_csv(fn, delimiter=' ', names=['example_id', 'correct', 'truth', 'pred', '', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'sent1_parse'])


def build_data_dict(data):
    table = dict()
    for ex in data:
        table[str(ex['example_id'])] = ex
    return table


def run():
    print("Loading data...")
    data, _ = load_listops_data.load_data(FLAGS.data_path)
    
    print("Building data table...")
    data_table = build_data_dict(data)

    print("Loading report...")
    report = read_report(FLAGS.report_path)
    report = report.to_records()

    for rec in tqdm(report):
        if str(rec.example_id) not in data_table:
            continue

        ex = data_table[str(rec.example_id)]
        transitions1 = [int(t) for t in rec.sent1_parse]
        tokens1 = ex['tokens']

        # Writes tree to file.
        graph1 = print_tree(tokens1, transitions1)

if __name__ == '__main__':
    gflags.DEFINE_string("report_path", "./checkpoints/example.report", "")
    gflags.DEFINE_string("data_path", "./spinn/data/listops/test_d20a.tsv", "")
    gflags.DEFINE_string("out_path", "./example_trees", "")

    FLAGS(sys.argv)

    run()
