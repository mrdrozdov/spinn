import random
import numpy as np

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
PROD_MOD = "[PM"
FL_SUM_MOD = "[FLSUM"
END = "]"

#OPERATORS = [MIN, MAX, MED, SUM_MOD, FIRST, LAST, PROD_MOD, FL_SUM_MOD]
OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = list(range(10))

VALUE_P = 0.25
MAX_ARGS = 6
MAX_DEPTH = 5
MAX_SEQ_LEN = 35

DATA_POINTS = 100000


def generate_tree(depth):
    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1))

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t


def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'


def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return (np.sum(l[1]) % 10)
        elif l[0] == PROD_MOD:
            return (np.prod(l[1]) % 10)
        elif l[0] == FL_SUM_MOD:
            return (l[1][0] + l[1][1]) % 10
    # We've hit an unsaturated function and an argument.
    elif isinstance(l, tuple):
        return (l[0], l[1] + [r])


data = set()
while len(data) < DATA_POINTS:
    #data.add(generate_tree(1))
    cur = generate_tree(1)
    if type(cur) == tuple:
        testl = str(cur[0]).replace("(", "").replace(")", "").replace(",", "").split()
        if (len(testl) + 1) > MAX_SEQ_LEN:
            pass
        else:
            data.add(cur)
    else:
        pass

for example in data:
    print(str(to_value(example)) + '\t' + to_string(example))
