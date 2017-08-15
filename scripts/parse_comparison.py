"""
Reads a parsed corpus (data_path) and a model report (report_path) from a model
that produces latent tree structures and computes the unlabeled F1 score between
the model's latent trees and:
- The ground-truth trees in the parsed corpus
- Strictly left-branching trees for the sentences in the parsed corpus
- Strictly right-branching trees for the sentences in the parsed corpus

Note that for binary-branching trees like these, precision, recall, and F1 are
equal by definition, so only one number is shown.

Usage:
$ python scripts/parse_comparison.py \
    --data_path ./snli_1.0/snli_1.0_dev.jsonl \
    --report_path ./logs/example-nli.report \
"""

import gflags
import sys
import codecs
import json

LABEL_MAP = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

FLAGS = gflags.FLAGS


def tokenize_parse(parse):
    return [token for token in parse.split() if token not in ['(', ')']]


def to_string(parse):
    if type(parse) is not list:
        return parse
    if len(parse) == 1:
        return parse[0]
    else:
        return '( ' + to_string(parse[0]) + ' ' + to_string(parse[1]) + ' )'


def tokens_to_rb(tree):
    if type(tree) is not list:
        return tree
    if len(tree) == 1:
        return tree[0]
    else:
        return [tree[0], tokens_to_rb(tree[1:])]


def to_rb(gt_table):
    new_data = {}
    for key in gt_table:
        parse = gt_table[key]
        tokens = tokenize_parse(parse)
        new_data[key] = to_string(tokens_to_rb(tokens))
    return new_data


def tokens_to_lb(tree):
    if type(tree) is not list:
        return tree
    if len(tree) == 1:
        return tree[0]
    else:
        return [tokens_to_lb(tree[:-1]), tree[-1]]


def to_lb(gt_table):
    new_data = {}
    for key in gt_table:
        parse = gt_table[key]
        tokens = tokenize_parse(parse)
        new_data[key] = to_string(tokens_to_lb(tokens))
    return new_data


def corpus_f1(corpus_1, corpus_2):
    """ 
    Note: If a few examples in one dataset are missing from the other (i.e., some examples from the source corpus were not included 
      in a model corpus), the shorter dataset must be supplied as corpus_1.
    """

    accum = 0.0
    count = 0.0
    for key in corpus_1:     
        accum += example_f1(corpus_1[key], corpus_2[key])
        count += 1
    return accum / count


def to_indexed_contituents(parse):
    sp = parse.split()
    if len(sp) == 1:
        return set([(0, 1)])

    backpointers = []
    indexed_constituents = set()
    word_index = 0
    for index, token in enumerate(sp):
        if token == '(':
            backpointers.append(word_index)
        elif token == ')':
            start = backpointers.pop()
            end = word_index
            if not "_PAD" in sp[start:end]: 
                constituent = (start, end)
                indexed_constituents.add(constituent)
        else:
            if token != "_PAD":
                word_index += 1
    return indexed_constituents


def example_f1(e1, e2):
    c1 = to_indexed_contituents(e1)
    c2 = to_indexed_contituents(e2)

    prec = float(len(c1.intersection(c2))) / len(c2)  # TODO: More efficient.
    return prec  # For strictly binary trees, P = R = F1


def run():
    gt = {}
    with codecs.open(FLAGS.main_data_path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            gt[loaded_example['pairID'] + "_1"] = loaded_example['sentence1_binary_parse']
            gt[loaded_example['pairID'] + "_2"] = loaded_example['sentence2_binary_parse']

    lb = to_lb(gt)
    rb = to_rb(gt)

    report = {}
    with codecs.open(FLAGS.main_report_path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            report[loaded_example['example_id'] + "_1"] = loaded_example['sent1_tree']
            report[loaded_example['example_id'] + "_2"] = loaded_example['sent2_tree']

    ptb = {}
    with codecs.open(FLAGS.ptb_data_path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            ptb[loaded_example['pairID']] = loaded_example['sentence1_binary_parse']

    ptb_report = {}
    with codecs.open(FLAGS.ptb_report_path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            ptb_report[loaded_example['example_id']] = loaded_example['sent1_tree']

    print FLAGS.main_report_path + '\t' + str(corpus_f1(report, lb)) + '\t' + str(corpus_f1(report, rb)) + '\t' + str(corpus_f1(report, gt)) + '\t' + str(corpus_f1(ptb_report, ptb))


if __name__ == '__main__':
    gflags.DEFINE_string("main_report_path", "./checkpoints/example-nli.report", "")
    gflags.DEFINE_string("main_data_path", "./snli_1.0/snli_1.0_dev.jsonl", "")
    gflags.DEFINE_string("ptb_report_path", "./snli_1.0/snli_1.0_dev.jsonl", "")
    gflags.DEFINE_string("ptb_data_path", "./ptb.jsonl", "")

    FLAGS(sys.argv)

    run()
