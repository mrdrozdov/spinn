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
import random
import re
import glob
from collections import Counter

LABEL_MAP = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

FLAGS = gflags.FLAGS

def spaceify(parse):
    return parse #.replace("(", "( ").replace(")", " )")

def tokenize_parse(parse):
    parse = spaceify(parse)
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


def average_depth(parse):
    depths = []
    current_depth = 0
    for token in parse.split():
        if token == '(':
            current_depth += 1
        elif token == ')':
            current_depth -= 1
        else:
            depths.append(current_depth)
    if len(depths) == 0:
        pass
    else:
        return float(sum(depths)) / len(depths)


def corpus_average_depth(corpus):
    local_averages = []
    for key in corpus:
        s = corpus[key]
        if average_depth(s) is not None:
            local_averages.append(average_depth(s))
        else:
            pass
    return float(sum(local_averages)) / len(local_averages)


def average_length(parse):
    parse = spaceify(parse)
    return len(parse.split())


def corpus_average_length(corpus):
    local_averages = []
    for key in corpus:
        if average_length(s) is not None:
            local_averages.append(average_length(s))
        else:
            pass
    return float(sum(local_averages)) / len(local_averages)


def corpus_stats(corpus_1, corpus_2, first_two=False, neg_pair=False):
    """ 
    Note: If a few examples in one dataset are missing from the other (i.e., some examples from the source corpus were not included 
      in a model corpus), the shorter dataset must be supplied as corpus_1.
    """

    f1_accum = 0.0
    count = 0.0
    first_two_count = 0.0
    last_two_count = 0.0
    three_count = 0.0
    neg_pair_count = 0.0
    neg_count = 0.0
    for key in corpus_1:     
        c1 = to_indexed_contituents(corpus_1[key])
        c2 = to_indexed_contituents(corpus_2[key])

        f1_accum += example_f1(c1, c2)
        count += 1

        if first_two and len(c1) > 1:
            if (0, 2) in c1:
                first_two_count += 1
            num_words = len(c1) + 1
            if (num_words - 2, num_words) in c1:
                last_two_count += 1
            three_count += 1 
        if neg_pair:
            word_index = 0
            s = spaceify(corpus_1[key])
            tokens = s.split()
            for token_index, token in enumerate(tokens):
                if token in ['(', ')']:
                    continue
                if token in ["n't", "not", "never", "no", "none", "Not", "Never", "No", "None"]:
                    if tokens[token_index + 1] not in ['(', ')']:
                        neg_pair_count += 1
                    neg_count += 1
                word_index += 1
    stats = f1_accum / count
    if first_two:
        stats = str(stats) + '\t' + str(first_two_count / three_count) + '\t' + str(last_two_count / three_count)
    if neg_pair:
        stats = str(stats) + '\t' + str(neg_pair_count / neg_count)
    return stats

def corpus_stats_labeled(corpus_unlabeled, corpus_labeled):
    """ 
    Note: If a few examples in one dataset are missing from the other (i.e., some examples from the source corpus were not included 
      in a model corpus), the shorter dataset must be supplied as corpus_1.
    """

    correct = Counter()
    total = Counter()

    for key in corpus_labeled:     
        c1 = to_indexed_contituents(corpus_unlabeled[key])
        c2 = to_indexed_contituents_labeled(corpus_labeled[key])
        if len(c2) == 0:
            continue

        ex_correct, ex_total = example_labeled_acc(c1, c2)
        correct.update(ex_correct)
        total.update(ex_total)
    return correct, total


def to_indexed_contituents(parse):
    if parse.count("(") != parse.count(")"):
        print parse
    parse = spaceify(parse)
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
            #if len(backpointers) == 0:
            #    pass
            #else:
            start = backpointers.pop()
            end = word_index
            constituent = (start, end)
            indexed_constituents.add(constituent)
        else:
            word_index += 1
    return indexed_constituents

def to_indexed_contituents_labeled(parse):
    sp = re.findall(r'\([^ ]+| [^\(\) ]+|\)', parse)
    if len(sp) == 1:
        return set([(0, 1)])

    backpointers = []
    indexed_constituents = set()
    word_index = 0
    for index, token in enumerate(sp):
        if token[0] == '(':
            backpointers.append((word_index, token[1:]))
        elif token == ')':
            start, typ = backpointers.pop()
            end = word_index
            constituent = (start, end, typ)
            if end - start > 1:
                indexed_constituents.add(constituent)
        else:
            word_index += 1
    return indexed_constituents

def example_f1(c1, c2):
    prec = float(len(c1.intersection(c2))) / len(c2)  # TODO: More efficient.
    return prec  # For strictly binary trees, P = R = F1

def example_labeled_acc(c1, c2):
    '''Compute the number of non-unary constituents of each type in the labeled (non-binirized) parse appear in the model output.'''
    correct = Counter()
    total = Counter()
    for constituent in c2:
        if (constituent[0], constituent[1]) in c1:
            correct[constituent[2]] += 1
        total[constituent[2]] += 1
    return correct, total

def randomize(parse):
    tokens = tokenize_parse(parse)
    while len(tokens) > 1:
        merge = random.choice(range(len(tokens) - 1))
        tokens[merge] = "( " + tokens[merge] + " " + tokens[merge + 1] + " )"
        del tokens[merge + 1]
    return tokens[0]

def to_latex(parse):
    return ("\\Tree " + parse).replace('(', '[').replace(')', ']').replace(' . ', ' $.$ ')

def read_nli_report(path):
    report = {}
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            report[loaded_example['example_id'] + "_1"] = unpad(loaded_example['sent1_tree'])
            report[loaded_example['example_id'] + "_2"] = unpad(loaded_example['sent2_tree'])
    return report

def read_nli_report_padded(path):
    report = {}
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            report[loaded_example['example_id'] + "_1"] = loaded_example['sent1_tree']
            report[loaded_example['example_id'] + "_2"] = loaded_example['sent2_tree']
    return report

def read_ptb_report(path):
    report = {}
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            report[loaded_example['example_id']] = unpad(loaded_example['sent1_tree'])
    return report

def unpad(parse):
    ok = ["(", ")", "_PAD"]
    unpadded = []
    tokens = parse.split()
    cur = [i for i in range(len(tokens)) if tokens[i] == "_PAD"]
    
    if len(cur) != 0:
        if tokens[cur[0]-1] in ok:
            unpad = tokens[:cur[0]-1]
        else: 
            unpad = tokens[:cur[0]]
    else:
        unpad = tokens
    
    sent = " ".join(unpad)
    while sent.count("(") != sent.count(")"):
        sent += " )"
    
    return sent


def run():
    gt = {}
    gt_labeled = {}
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

            gt_labeled[loaded_example['pairID'] + "_1"] = loaded_example['sentence1_parse']
            gt_labeled[loaded_example['pairID'] + "_2"] = loaded_example['sentence2_parse']

    lb = to_lb(gt)
    rb = to_rb(gt)

    print "GT average depth", corpus_average_depth(gt)

    ptb = {}
    ptb_labeled = {}
    if FLAGS.ptb_data_path != "_":
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
                ptb_labeled[loaded_example['pairID']] = loaded_example['sentence1_parse']

    reports = []
    ptb_reports = []
    if FLAGS.use_random_parses:
        print "Creating five sets of random parses for the main data."
        report_paths = range(5)
        for _ in report_paths:
            report = {}
            for sentence in gt:
                report[sentence] = randomize(gt[sentence])
            reports.append(report)  

        print "Creating five sets of random parses for the PTB data."
        ptb_report_paths = range(5)
        for _ in report_paths:
            report = {}
            for sentence in ptb:
                report[sentence] = randomize(ptb[sentence])
            ptb_reports.append(report)
    else:
        report_paths = glob.glob(FLAGS.main_report_path_template)
        for path in report_paths:
            print "Loading", path
            reports.append(read_nli_report(path))

        if FLAGS.main_report_path_template != "_":
            ptb_report_paths = glob.glob(FLAGS.ptb_report_path_template)
            for path in ptb_report_paths:
                print "Loading", path
                ptb_reports.append(read_ptb_report(path))

    if len(reports) > 1 and FLAGS.compute_self_f1:
        f1s = []
        for i in range(len(report_paths) - 1):
            for j in range(i + 1, len(report_paths)):
                path_1 = report_paths[i]
                path_2 = report_paths[j]
                f1 = corpus_stats(reports[i], reports[j])
                f1s.append(f1)
        print "Mean Self F1:\t" + str(sum(f1s) / len(f1s))

    correct = Counter()
    total = Counter()

    for i, report in enumerate(reports):
        print report_paths[i]
        if FLAGS.print_latex > 0:
            for index, sentence in enumerate(gt):
                if index == FLAGS.print_latex:
                    break
                print to_latex(gt[sentence])
                print to_latex(report[sentence])
                print
        print "left-branching:", str(corpus_stats(report, lb)) + '\t' + "right-branching:",  str(corpus_stats(report, rb)) + '\t' + "ground truth trees:", str(corpus_stats(report, gt, first_two=FLAGS.first_two, neg_pair=FLAGS.neg_pair)) + '\t' + "average tree depth:", str(corpus_average_depth(report))

        set_correct, set_total = corpus_stats_labeled(report, gt_labeled)
        correct.update(set_correct)
        total.update(set_total)

    for key in sorted(total):
        print key + '\t' + str(correct[key] * 1. / total[key]) #+ '\t' + str(total[key])

    correct_ptb = Counter()
    total_ptb = Counter()
    for i, report in enumerate(ptb_reports):
        print ptb_report_paths[i]
        if FLAGS.print_latex > 0:
            for index, sentence in enumerate(ptb):
                if index == FLAGS.print_latex:
                    break
                print to_latex(ptb[sentence])
                print to_latex(report[sentence])
                print
        print "GT average depth", corpus_average_depth(ptb)
        print  "F1 w/ ground truth trees:", str(corpus_stats(report, ptb)) + '\t' + "average tree depth", str(corpus_average_depth(report))
        set_correct_ptb, set_total_ptb = corpus_stats_labeled(report, ptb_labeled)
        correct_ptb.update(set_correct_ptb)
        total_ptb.update(set_total_ptb)

    for key in sorted(total_ptb):
        print key + '_ptb' + '\t' + str(correct_ptb[key] * 1. / total_ptb[key]), '\t', total_ptb[key]

    report_paths = glob.glob(FLAGS.main_report_path_template)
    for path in report_paths:
        reports.append(read_nli_report_padded(path))

    for report in reports:
        ok = ["(", ")", "_PAD"]
        meh = ["(", ")", "_PAD", "."]
        count = 0
        bads = []
        v_bads = []
        for key in report:
            parse = report[key]
            tokens = parse.split()
            for i in range(len(tokens)):
                if tokens[i] == "_PAD":
                    if tokens[i-1] not in ok:
                        count += 1
                        bads.append(parse)
                    if tokens[i-1] not in meh:
                        v_bads.append(parse)
        print "(. _PAD):", float(len(bads))/len(report), '\t',  "(word _PAD):", float(len(v_bads))/len(report)


if __name__ == '__main__':
    gflags.DEFINE_string("main_report_path_template", "./checkpoints/*.report", "A template (with wildcards input as \*) for the paths to the main reports.")
    gflags.DEFINE_string("main_data_path", "./snli_1.0/snli_1.0_dev.jsonl", "A template (with wildcards input as \*) for the paths to the main reports.")
    gflags.DEFINE_string("ptb_report_path_template", "_", "A template (with wildcards input as \*) for the paths to the PTB reports, or '_' if not available.")
    gflags.DEFINE_string("ptb_data_path", "_", "The path to the PTB data in SNLI format, or '_' if not available.")
    gflags.DEFINE_boolean("compute_self_f1", True, "Compute self F1 over all reports matching main_report_path_template.")
    gflags.DEFINE_boolean("use_random_parses", False, "Replace all report trees with randomly generated trees. Report path template flags are not used when this is set.")
    gflags.DEFINE_boolean("first_two", False, "Show 'first two' and 'last two' metrics.")
    gflags.DEFINE_boolean("neg_pair", False, "Show 'neg_pair' metric.")
    gflags.DEFINE_integer("print_latex", 0, "Print this many trees in LaTeX format for each report.")

    FLAGS(sys.argv)

    run()
