import gflags

import itertools

from spinn import util
from spinn.data.util.arithmetic import ArithmeticData

import sys


NUMBERS = range(-10, 11)

FIXED_VOCABULARY = {str(x): i + 1 for i, x in enumerate(NUMBERS)}
FIXED_VOCABULARY.update({
    util.PADDING_TOKEN: 0,
    "+": len(FIXED_VOCABULARY) + 1,
    "-": len(FIXED_VOCABULARY) + 2
})
assert len(set(FIXED_VOCABULARY.values())) == len(FIXED_VOCABULARY.values())


class ArithmeticDataType(object):

    @property
    def LABELS(self):
        raise NotImplementedError

    def is_label(self, x):
        raise NotImplementedError


class SignData(ArithmeticDataType):

    LABELS = ["-", "+", "0"]

    def is_label(self, x):
        if x < 0:
            return 0
        if x > 0:
            return 1
        if x == 0:
            return 2


class SimpleData(ArithmeticDataType):

    LABELS = NUMBERS

    def is_label(self, x):
        try:
            return self.LABELS.index(x)
        except:
            return -1


if __name__ == '__main__':
    FLAGS = gflags.FLAGS
    gflags.DEFINE_integer("length", 5, "")
    gflags.DEFINE_integer("limit", 100, "")
    # gflags.DEFINE_string("exclude", None, "If not None, exclude any example that appears in this file.")
    gflags.DEFINE_enum("data_type", "simple", ["simple", "sign"], "")
    FLAGS(sys.argv)

    limit = FLAGS.limit
    length = FLAGS.length

    if FLAGS.data_type == "simple":
        data_type = SimpleData()
    elif FLAGS.data_type == "sign":
        data_type = SignData()

    label_size = limit // len(data_type.LABELS)

    dataset = ArithmeticData(NUMBERS)
    generator = dataset.generate_prefix_seqs(length)

    for idx in range(limit):

        label = min(idx // label_size, len(data_type.LABELS) - 1)

        for ii, _ in enumerate(itertools.repeat(None)):
            result, seq = next(generator)

            if data_type.is_label(result) == label:
                print "{}\t{}".format(data_type.LABELS[label],
                    " ".join(dataset.convert_to_sexpr(seq)),
                    )
                break
