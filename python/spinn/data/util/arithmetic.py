from collections import deque
import random
import copy


class ArithmeticData(object):

    def __init__(self, NUMBERS):
        self.NUMBERS = NUMBERS
        self.ops = ['+', '-']
        self.numbers = [str(x) for x in NUMBERS]
        self.all_tokens = self.ops + self.numbers


    def eval_prefix_seq(self, seq):
        token = seq.popleft()
        if token == '+':
            return self.eval_prefix_seq(seq) + self.eval_prefix_seq(seq)
        elif token == '-':
            return self.eval_prefix_seq(seq) - self.eval_prefix_seq(seq)
        else:
            return int(token)


    def gen_prefix_seq(self, max_len):
        length = random.randint(3, max_len)

        seq = [random.choice(self.ops)]
        depth = 2
        for _ in range(length - 1):
            choice = None
            if depth >= 1:
                if random.random() < 0.4:
                    choice = random.choice(self.ops)
                else:
                    choice = random.choice(self.all_tokens)

            if choice is None:
                break

            if choice in self.ops:
                depth += 1
            else:
                depth -= 1
            seq.append(choice)

        return deque(seq)


    def generate_prefix_seqs(self, max_len, min=None, max=None):
        min = self.NUMBERS[0] if min is None else min
        max = self.NUMBERS[-1] if max is None else max
        while True:
            try:
                seq = self.gen_prefix_seq(max_len)
                result = self.eval_prefix_seq(copy.copy(seq))
            except: pass
            else:
                if result >= min and result <= max:
                    yield result, seq


    def convert_to_sexpr(self, prefix_seq):
        ret = []

        depth = 0
        right_branch = False
        for i in range(len(prefix_seq)):
            token = prefix_seq[i]
            if token in self.ops:
                ret.extend(["(", token, "("])

                depth += 2
                right_branch = False
            else:
                ret.append(token)
                if right_branch:
                    ret.extend([")", ")"])
                    depth -= 2
                else:
                    right_branch = True

        ret.extend([")"] * depth)

        return ret
