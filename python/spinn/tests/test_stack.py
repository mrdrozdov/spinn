import unittest
import numpy as np

from spinn import util
from spinn.fat_stack import SPINN, SentenceModel, SentencePairModel

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.test import MockModel, default_args, get_batch


class SPINNTestCase(unittest.TestCase):

    def test_basic_stack(self):
        model = MockModel(SentenceModel, default_args())

        train = False

        X, transitions = get_batch()

        class Projection(nn.Module):
            def forward(self, x):
                return x[:, :default_args()['model_dim']]

        class Reduce(nn.Module):
            def forward(self, lefts, rights, tracking):
                batch_size = len(lefts)
                return torch.chunk(torch.cat(lefts, 0) - torch.cat(rights, 0), batch_size, 0)

        model.embed.projection = Projection()
        model.spinn.reduce = Reduce()

        model(X, transitions)
        outputs = model.spinn_outp[0]

        assert outputs[0][0].data[0] == (3 - (1 - (2 - 1)))
        assert outputs[1][0].data[0] == ((3 - 2) - (4 - 5))


    def test_validate_transitions_cantskip(self):
        model = MockModel(SentenceModel, default_args())

        train = False

        # To Test:
        # 1. Cant SKIP
        # 2. Cant SHIFT
        # 3. Cant REDUCE
        # 4. No change SHIFT
        # 5. No change REDUCE

        bufs = [
            [None],
            [],
            [None],
            [None],
            [None],
            [],
        ]

        stacks = [
            [None],
            [None],
            [None],
            [],
            [],
            [None, None],
        ]

        transitions = [
            2, 1, 0, 0, 0, 1
            ]
        preds = np.array([
            0, 0, 1, 1, 0, 1
            ]).astype(np.int32)


        ret = model.spinn.validate(transitions, preds, stacks, bufs, zero_padded=False)
        expected = np.array([
            2, 1, 0, 0, 0, 1
        ], dtype=np.int32)

        assert all(p == e for p, e in zip(preds.ravel().tolist(), expected.ravel().tolist()))


if __name__ == '__main__':
    unittest.main()
