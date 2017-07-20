class Catalan(object):
    def __init__(self):
        """
        Returns catalan numbers: 1 1 2 5 14 42 132...

        Memoized. Should run in amortized O(1) time.

        """
        self.sofar = 0
        self.cache = [1]

    def catalan_it(self, start=0, end=0, c=1):
        for n in range(start + 1, end + 2):
            yield c
            c = c * 2 * (2 * n - 1) / (n + 1)
            if n > self.sofar:
                self.sofar = self.sofar + 1
                self.cache.append(c)

    def catalan(self, n):
        if n <= self.sofar:
            return self.cache[n]
        else:
            for c in self.catalan_it(start=self.sofar, end=n, c=self.cache[self.sofar]):
                pass
            return c


class CatalanPyramid(object):
    def __init__(self):
        self.cat = Catalan()
        self.fraction_rows = [
            [(1, 2)],
        ]
        self.decimal_rows = [
            [0.5],
        ]

    def fill_rows(self, depth):
        populated_rows = len(self.decimal_rows)
        if populated_rows < depth:
            for i in range(populated_rows, depth):
                n_i = i + 1
                fraction_row = [[None, None] for _ in range(n_i)]

                # Rule 1: Right Diagonal Numerator
                fraction_row[0][0] = 1

                # Rule 2: Right Diagonal Denominator
                fraction_row[0][1] = i + 2

                # Rule 3: Left Diagonal Denominator
                fraction_row[n_i - 1][1] = self.cat.catalan(i + 2)

                # Rule 4: Left Diagonal Numerator
                fraction_row[n_i - 1][0] = fraction_row[n_i - 1][1] - self.fraction_rows[-1][n_i - 2][1]

                for j in range(1, n_i - 1):
                    # Rule 5: Rest of Numerators
                    fraction_row[j][0] = fraction_row[j - 1][1]

                    # Rule 6: Rest of Denominators
                    fraction_row[j][1] = fraction_row[j][0] + self.fraction_rows[-1][j][1]

                # Normalize the row
                decimal_row = [numerator / float(denominator) for numerator, denominator in fraction_row]

                self.fraction_rows.append(fraction_row)
                self.decimal_rows.append(decimal_row)

    def access(self, n_reduces, i, n_tokens):
        n_rows = n_tokens - 2
        depth = n_tokens - 1
        width = depth * 2 + 1
        n_shifts = i - n_reduces
        n_stack = n_shifts - n_reduces

        self.fill_rows(n_tokens-2)

        if n_reduces > 0 and n_reduces > n_shifts - 1: # unreachable state.
            return 0.0
        elif n_stack <= 1: # insufficient items on stack. must shift.
            return 1.0
        elif n_reduces == n_tokens - 1: # no reduces left. must shift.
            return 1.0
        elif n_shifts == n_tokens: # no shifts left. must reduce.
            return 0.0
        elif i >= n_tokens + n_reduces: # nothing on buffer. must reduce.
            return 0.0
        else:
            offset = n_reduces * 2 + 2
            ii = i - offset

            relevant_rows = list(reversed(self.decimal_rows[0:n_tokens-2]))
            decimal_row = list(reversed(relevant_rows[n_reduces]))

            return decimal_row[ii]


class ShiftProbabilities(object):
    def __init__(self):
        self.cache = dict()
        self.catalan_pyramid = CatalanPyramid()

    def prob(self, n_reduces, i, n_tokens):
        return catalan_pyramid.access(n_reduces, i, n_tokens)
