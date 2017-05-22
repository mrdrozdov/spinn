import unittest


# PyTorch

from spinn.util.misc import Accumulator


class MiscTestCase(unittest.TestCase):

    def test_accumulator(self):
        A = Accumulator()

        A.add('key', 0)
        A.add('key', 0)

        assert len(A.get('key')) == 2
        assert len(A.get('key')) == 0


if __name__ == '__main__':
    unittest.main()
