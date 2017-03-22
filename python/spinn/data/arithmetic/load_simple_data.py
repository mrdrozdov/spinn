from spinn import util

from spinn.data import T_SHIFT, T_REDUCE, T_SKIP, T_STRUCT
from spinn.data.arithmetic.base import NUMBERS, FIXED_VOCABULARY

SENTENCE_PAIR_DATA = False
OUTPUTS = range(-10, 11)
LABEL_MAP = {str(x): i for i, x in enumerate(OUTPUTS)}


def structure_transitions(tokens, transitions):

    OP = 0
    LEFT = 1
    RIGHT = 2
    COMPLETE = 3

    def is_op(x):
        return x == '-' or x == '+'

    def SHIFT(x):
        return OP if is_op(x) else COMPLETE

    def REDUCE(left, right):
        if left == LEFT and right == COMPLETE:
            return COMPLETE
        elif left == OP and right == RIGHT:
            return COMPLETE
        elif left == OP and right == COMPLETE:
            return LEFT
        elif left == COMPLETE and right == COMPLETE:
            return RIGHT
        else:
            raise Exception

    buf = list(reversed(tokens))
    stack = []
    ret = []

    for t in transitions:
        if t == T_SHIFT:
            x = buf.pop()
            stack.append(SHIFT(x))
            ret.append(T_SHIFT)
        elif t == T_REDUCE:
            right, left = stack.pop(), stack.pop()
            new_stack_item = REDUCE(left, right)
            stack.append(new_stack_item)

            if new_stack_item == COMPLETE:
                ret.append(T_STRUCT)
            else:
                ret.append(T_REDUCE)
        elif t == T_SKIP:
            ret.append(T_SKIP)
        else:
            raise Exception
    return ret


def load_data(path, lowercase=None):
    examples = []
    with open(path) as f:
        for example_id, line in enumerate(f):
            line = line.strip()
            label, seq = line.split('\t')
            tokens, transitions = util.convert_binary_bracketed_seq(seq.split(' '))

            example = {}
            example["label"] = label
            example["sentence"] = seq
            example["tokens"] = tokens
            example["transitions"] = transitions
            example["structure_transitions"] = structure_transitions(tokens[:], transitions[:])
            example["example_id"] = str(example_id)

            examples.append(example)
    return examples, FIXED_VOCABULARY
