import unittest

from spinn.plain_rnn import BaseModel

# PyTorch

from spinn.util.test import MockModel, default_args, get_batch, get_batch_pair


class RNNTestCase(unittest.TestCase):

    def test_single_rnn(self):
        model = MockModel(BaseModel, default_args())
        X, transitions = get_batch()
        outputs = model(X, transitions)
        assert outputs.size() == (2, 3)

    def test_pair_rnn(self):
        model = MockModel(BaseModel, default_args(use_sentence_pair=True))
        X, transitions = get_batch_pair()
        assert X.shape == (2, 4, 2)
        assert transitions.shape == (2, 7, 2)
        outputs = model(X, transitions)
        assert outputs.size() == (2, 3)


if __name__ == '__main__':
    unittest.main()
