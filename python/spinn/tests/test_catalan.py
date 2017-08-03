import unittest


from spinn.util.catalan import Catalan, CatalanPyramid


target_3 = [
    [1.0, 1.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
]

target_12 = [
    [1.0, 1.0, 0.714285714286, 0.6, 0.526315789474, 0.466666666667, 0.411764705882, 0.357142857143, 0.3,
        0.238095238095, 0.169230769231, 0.0909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.710526315789, 0.592592592593, 0.514705882353, 0.45, 0.388888888889,
        0.326530612245, 0.259615384615, 0.185185185185, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.705882352941, 0.583333333333, 0.5, 0.428571428571, 0.358974358974,
        0.285714285714, 0.204545454545, 0.111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.7, 0.571428571429, 0.480769230769, 0.4,
        0.318181818182, 0.228571428571, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.692307692308, 0.555555555556,
        0.454545454545, 0.36, 0.259259259259, 0.142857142857, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.681818181818,
        0.533333333333, 0.416666666667, 0.3, 0.166666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.666666666667, 0.5, 0.357142857143, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.642857142857, 0.444444444444, 0.25, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.333333333333, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
]


def almost_equal(a, b, threshold=1e-6):
    return abs(a - b) < threshold


class CatalanTestCase(unittest.TestCase):

    def test_catalan_numbers(self):
        expected = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]
        cat = Catalan()

        for i in range(len(expected)):
            actual = cat.catalan(i)
            assert actual == expected[i], "{} != {}".format(actual, expected)


    def test_lookup_table_3(self):
        n_tokens = 3
        catalan_pyramid = CatalanPyramid()

        depth = n_tokens - 1
        width = n_tokens * 2 - 1

        for i in range(width):
            for n_reduces in range(depth):
                actual = catalan_pyramid.access(n_reduces, i, n_tokens)
                expected = target_3[n_reduces][i]
                assert almost_equal(actual, expected), "\nRet: {}\nExp: {}".format(
                    actual, expected)

    def test_lookup_table_3_with_twist(self):
        n_tokens = 3
        catalan_pyramid = CatalanPyramid()

        catalan_pyramid.access(0, 0, 10)  # populate extra rows.

        depth = n_tokens - 1
        width = n_tokens * 2 - 1

        for i in range(width):
            for n_reduces in range(depth):
                actual = catalan_pyramid.access(n_reduces, i, n_tokens)
                expected = target_3[n_reduces][i]
                assert almost_equal(actual, expected), "\nRet: {}\nExp: {}".format(
                    actual, expected)

    def test_lookup_table_12(self):
        n_tokens = 12
        catalan_pyramid = CatalanPyramid()

        depth = n_tokens - 1
        width = n_tokens * 2 - 1

        for i in range(width):
            for n_reduces in range(depth):
                actual = catalan_pyramid.access(n_reduces, i, n_tokens)
                expected = target_12[n_reduces][i]
                assert almost_equal(actual, expected), "\nRet: {}\nExp: {}".format(
                    actual, expected)

    def test_lookup_table_12_with_twist(self):
        n_tokens = 12
        catalan_pyramid = CatalanPyramid()

        catalan_pyramid.access(0, 0, 20)  # populate extra rows.

        depth = n_tokens - 1
        width = n_tokens * 2 - 1

        for i in range(width):
            for n_reduces in range(depth):
                actual = catalan_pyramid.access(n_reduces, i, n_tokens)
                expected = target_12[n_reduces][i]
                assert almost_equal(actual, expected), "\nRet: {}\nExp: {}".format(
                    actual, expected)

    def test_lookup_table_12_random_access(self):
        n_tokens = 12
        n_reduces = 3
        i = 10
        expected = 0.480769230769

        catalan_pyramid = CatalanPyramid()
        actual = catalan_pyramid.access(n_reduces, i, n_tokens)

        assert almost_equal(actual, expected), "\nRet: {}\nExp: {}".format(
            actual, expected)

    def test_large(self):
        n_tokens = 1000
        n_reduces = 3
        i = 10
        # expected = ?

        catalan_pyramid = CatalanPyramid()
        # causes OverflowError when dividing long ints
        actual = catalan_pyramid.access(n_reduces, i, n_tokens)


if __name__ == '__main__':
    unittest.main()
