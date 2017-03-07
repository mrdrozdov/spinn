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


class EqData(ArithmeticDataType):

    LABELS = [True, False]

    def is_label(self, x):
        r1, r2 = x
        for i, b in enumerate(self.LABELS):
            if (r1 == r2) == b:
                return i


class RelationalData(ArithmeticDataType):

    LABELS = ["<", ">", "="]

    def is_label(self, x):
        r1, r2 = x
        if r1 < r2:
            return 0
        if r1 > r2:
            return 1
        if r1 == r2:
            return 2


if __name__ == '__main__':
    FLAGS = gflags.FLAGS
    gflags.DEFINE_integer("length", 5, "")
    gflags.DEFINE_integer("limit", 10, "")
    gflags.DEFINE_enum("data_type", "eq", ["eq", "relational"], "")
    FLAGS(sys.argv)

    limit = FLAGS.limit
    length = FLAGS.length

    if FLAGS.data_type == "eq":
        data_type = EqData()
    elif FLAGS.data_type == "relational":
        data_type = RelationalData()

    label_size = limit // len(data_type.LABELS)

    dataset = ArithmeticData(NUMBERS)
    generator = dataset.generate_prefix_seqs(length)

    for idx in range(limit):

        label = min(idx // label_size, len(data_type.LABELS) - 1)

        for ii, _ in enumerate(itertools.repeat(None)):
            if ii % 100 == 0:
                result1, seq1 = next(generator)
            result2, seq2 = next(generator)

            # TODO: Can do equal to, greater than, or equal.
            if data_type.is_label((result1, result2)) == label:
                print "{}\t{}\t{}".format(data_type.LABELS[label],
                    " ".join(dataset.convert_to_sexpr(seq1)),
                    " ".join(dataset.convert_to_sexpr(seq2)),
                    )
                break
