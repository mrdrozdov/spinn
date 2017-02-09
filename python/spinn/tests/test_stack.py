import unittest
import argparse

from nose.plugins.attrib import attr
import numpy as np

import pytest

from spinn import util
from spinn.fat_stack import SPINN
from spinn.fat_stack import SentenceModel

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


def default_args():
    args = {}

    # Required Args
    args['model_dim'] = 10
    args['word_embedding_dim'] = 12
    args['vocab_size'] = 14
    args['num_classes'] = 3
    args['mlp_dim'] = 16
    args['embedding_keep_rate'] = 1.0
    args['classifier_keep_rate'] = 1.0

    initial_embeddings = np.zeros((args['vocab_size'], args['word_embedding_dim'])).astype(np.float32)

    args['initial_embeddings'] = initial_embeddings

    # Tracker Args
    args['tracking_lstm_hidden_dim'] = 4
    args['transition_weight'] = None
    args['use_tracking_lstm'] = False

    # Data Args
    args['use_skips'] = False

    return args


class MockSentenceModel(SentenceModel):

    def __init__(self, **kwargs):
        _kwargs = default_args()

        for k, v in kwargs.iteritems():
            _kwargs[k] = v

        super(MockSentenceModel, self).__init__(**_kwargs)


# def default_embeddings(vocab_size, embedding_dim):
#     initial_embeddings = np.arange(vocab_size).reshape(
#             (vocab_size, 1)).repeat(embedding_dim, axis=1).astype(np.float32)
#     vocab = {
#         'size': initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size,
#         'vectors': initial_embeddings,
#     }
#     vocab = argparse.Namespace(**vocab)
#     return vocab, initial_embeddings


class SPINNTestCase(unittest.TestCase):

    def xtest_basic_stack(self):
        model = MockSentenceModel()

        train = False

        X = np.array([
            [3, 1,  2, 1],
            [3, 2,  4, 5]
        ], dtype=np.int32)

        transitions = np.array([
            # First input: push a bunch onto the stack
            [0, 0, 0, 0, 1, 1, 1],
            # Second input: push, then merge, then push more. (Leaves one item
            # on the buffer.)
            [0, 0, 1, 0, 0, 1, 1]
        ], dtype=np.int32)

        expected_stack_lens = [4, 2]

        example = model.build_example(X, transitions, train)
        _ = model.spinn(example)
        stack_lens = [len(s) for s in model.spinn.stacks]

        np.testing.assert_equal(stack_lens, expected_stack_lens)

    def test_validate_transitions_cantskip(self):
        model = MockSentenceModel()

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
