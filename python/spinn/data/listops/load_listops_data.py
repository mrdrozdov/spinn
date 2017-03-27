from spinn import util

from spinn.data.listops.base import NUMBERS, FIXED_VOCABULARY

SENTENCE_PAIR_DATA = False
OUTPUTS = range(10)
LABEL_MAP = {str(x): i for i, x in enumerate(OUTPUTS)}


def spans(tokens, transitions):
    distinct_spans = []
    structure_spans = []

    n = len(tokens)
    stack = []
    buf = list(reversed([(l, r) for l, r in zip(range(n), range(1, n+1))]))

    def SHIFT(item):
        distinct_spans.append(item)
        return item

    def REDUCE(l, r):
        new_stack_item = (l[0], r[1])
        distinct_spans.append(new_stack_item)
        return new_stack_item

    for t in transitions:
        if t == 0:
            stack.append(SHIFT(buf.pop()))
        elif t == 1:
            r, l = stack.pop(), stack.pop()
            stack.append(REDUCE(l, r))

    return distinct_spans, structure_spans


def load_data(path, lowercase=None):
    examples = []
    with open(path) as f:
        for example_id, line in enumerate(f):
            line = line.strip()
            label, seq = line.split('\t')
            if len(seq) <= 1:
                continue

            tokens, transitions = util.convert_binary_bracketed_seq(seq.split(' '))

            example = {}
            example["label"] = label
            example["sentence"] = seq
            example["tokens"] = tokens
            example["transitions"] = transitions
            example["example_id"] = str(example_id)

            examples.append(example)
    return examples, FIXED_VOCABULARY
