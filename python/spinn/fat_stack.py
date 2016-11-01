"""
A naive Theano implementation of a stack whose elements are symbolic vector
values. This "fat stack" powers the "fat classifier" model, which supports
training and inference in all model configurations.

Of course, we sacrifice speed for this flexibility. Theano's symbolic
differentiation algorithm isn't friendly to this implementation, and
so it suffers from poor backpropagation runtime efficiency. It is also
relatively slow to compile.
"""

from functools import partial

import numpy as np
from spinn import util

# Chainer imports
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
from chainer.functions.connection import embed_id
from chainer.functions.normalization.batch_normalization import batch_normalization
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer.training import extensions

from chainer.functions.activation import slstm
from chainer.utils import type_check

import spinn.util.chainer_blocks as CB

from spinn.util.chainer_blocks import LSTMChain, RNNChain, EmbedChain
from spinn.util.chainer_blocks import MLP
from spinn.util.chainer_blocks import CrossEntropyClassifier

"""
Documentation Symbols:

B: Batch Size
B*: Dynamic Batch Size
S: Sequence Length
S*: Dynamic Sequence Length
E: Embedding Size
H: Output Size of Current Module

Style Guide:

1. Each __call__() or forward() should be documented with its
   input and output types/dimensions.
2. Every ChainList/Chain/Link needs to have assigned a __gpu and __mod.
3. Each __call__() or forward() should have `train` as a parameter,
   and Variables need to be set to Volatile=True during evaluation.
4. Each __call__() or forward() should have an accompanying `check_type_forward`
   called along the lines of:

   ```
   in_data = tuple([x.data for x in [input_1, input_2]])
   in_types = type_check.get_types(in_data, 'in_types', False)
   self.check_type_forward(in_types)
   ```

   This is mimicing the behavior seen in Chainer Functions.
5. Each __call__() or forward() should have a chainer.Variable as input.
   There may be slight exceptions to this rule, since at a times
   especially in this model a list is preferred, but try to stick to
   this as close as possible.

TODO:

- [x] Compute embeddings for initial sequences.
- [x] Convert embeddings into list of lists of Chainer Variables.
- [x] Loop over transitions, modifying buffer and stack as
      necessary using ``PseudoReduce''.
      NOTE: In this implementation, we pad the transitions
      with `-1` to indicate ``skip''.
- [x] Add projection layer to convert embeddings into proper
      dimensions for the TreeLSTM.
- [x] Use TreeLSTM reduce in place of PseudoReduce.
- [x] Debug NoneType that is coming out of gradient. You probably
      have to pad the sentences. SOLVED: The gradient was not
      being generated for the projection layer because of a
      redundant called to Variable().
- [x] Use the right C and H units for the TreeLSTM.
- [x] Enable evaluation. Currently crashing.
- [ ] Confirm that volatile is working correctly during eval time.
      Time the eval with and without volatile being set. Full eval
      takes about 2m to complete on AD Mac.

Other Tasks:

- [x] Run CBOW. 
- [ ] Enable Cropping and use longer sequences. Currently will
      not work as expected.
- [ ] Enable "transition validation".
- [ ] Enable TreeGRU as alternative option to TreeLSTM.
- [ ] Add TrackingLSTM.
- [ ] Run RNN for comparison.

Questions:

- [ ] Is the Projection layer implemented correctly? Efficiently?
- [ ] Is the composition with TreeLSTM implemented correctly? Efficiently?
- [ ] What should the types of Transitions and Y labels be? np.int64?

"""

def tensor_to_lists(inp, reverse=True):
    b, l = inp.shape[0], inp.shape[1]
    out = [F.split_axis(x, l, axis=0, force_tuple=True) for x in inp]

    if reverse:
        out = [list(reversed(x)) for x in out]
    else:
        out = [list(x) for x in out]

    return out


class TreeLSTMChain(Chain):
    def __init__(self, hidden_dim, prefix="TreeLSTMChain", gpu=-1):
        super(TreeLSTMChain, self).__init__(
            W_l=L.Linear(hidden_dim, hidden_dim*5, nobias=True),
            W_r=L.Linear(hidden_dim, hidden_dim*5, nobias=True),
            b=L.Bias(axis=1, shape=(hidden_dim*5,)),
            )
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def __call__(self, c_l, h_l, c_r, h_r, train=True, keep_hs=False):
        # TODO: Figure out bias. In this case, both left and right
        # weights have intrinsic bias, but this was not the strategy
        # in the previous code base. I think the trick is to use 
        # add_param, and then F.broadcast when doing the addition.
        gates = self.b(self.W_l(h_l) + self.W_r(h_r))

        # Compute and slice gate values
        i_gate, fl_gate, fr_gate, o_gate, cell_inp = \
            F.split_axis(gates, 5, axis=1)

        # Apply nonlinearities
        i_gate = F.sigmoid(i_gate)
        fl_gate = F.sigmoid(fl_gate)
        fr_gate = F.sigmoid(fr_gate)
        o_gate = F.sigmoid(o_gate)
        cell_inp = F.tanh(cell_inp)

        # Compute new cell and hidden value
        c_t = fl_gate * c_l + fr_gate * c_r + i_gate * cell_inp
        h_t = o_gate * F.tanh(c_t)

        return F.concat([h_t, c_t], axis=1)


class ReduceChain(Chain):
    def __init__(self, hidden_dim, prefix="ReduceChain", gpu=-1):
        super(ReduceChain, self).__init__(
            treelstm=TreeLSTMChain(hidden_dim / 2),
        )
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        left_type, right_type = in_types

        type_check.expect(
            left_type.dtype == 'f',
            left_type.ndim >= 1,
            right_type.dtype == 'f',
            right_type.ndim >= 1,
        )

    def __call__(self, left_x, right_x, train=True, keep_hs=False):
        """
        Args:
            left_x:  B* x H
            right_x: B* x H
        Returns:
            final_state: B* x H
        """

        # BEGIN: Type Check
        for l, r in zip(left_x, right_x):
            in_data = tuple([x.data for x in [l, r]])
            in_types = type_check.get_types(in_data, 'in_types', False)
            self.check_type_forward(in_types)
        # END: Type Check

        assert len(left_x) == len(right_x)
        batch_size = len(left_x)

        # Concatenate the list of states.
        left_x = F.stack(left_x, axis=0)
        right_x = F.stack(right_x, axis=0)
        assert left_x.shape == right_x.shape, "Left and Right must match in dimensions."

        # Split each state into its c/h representations.
        c_l, h_l = F.split_axis(left_x, 2, axis=1)
        c_r, h_r = F.split_axis(right_x, 2, axis=1)

        lstm_state = self.treelstm(c_l, h_l, c_r, h_r)
        return lstm_state


class SPINN(Chain):
    def __init__(self, hidden_dim, keep_rate, prefix="SPINN", gpu=-1):
        super(SPINN, self).__init__(
            reduce=ReduceChain(hidden_dim, gpu=gpu),
        )
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        buff_type = in_types[0]

        type_check.expect(
            buff_type.dtype == 'f',
            buff_type.ndim >= 1,
        )

    def __call__(self, buffers, transitions, train=True, keep_hs=False, use_sum=False):
        """
        Pass over batches of transitions, modifying their associated
        buffers at each iteration.

        Args:
            buffers: List of B x S* x E
            transitions: List of B x S
        Returns:
            final_state: List of B x E
        """

        # BEGIN: Type Check
        in_data = tuple([x.data for x in [buffers]])
        in_types = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_types)
        # END: Type Check

        batch_size, seq_length, hidden_dim = buffers.shape[0], buffers.shape[1], buffers.shape[2]
        transitions = transitions.T
        assert len(transitions) == seq_length
        buffers = [b for b in buffers]
        buffers_t = [seq_length-1 for _ in buffers]

        # Initialize stack with at least one item, otherwise gradient might
        # not propogate.
        stacks = [[] for b in buffers]

        def pseudo_reduce(lefts, rights):
            for l, r in zip(lefts, rights):
                yield l + r

        def better_reduce(lefts, rights):
            lstm_state = self.reduce(lefts, rights, train=train)
            for state in lstm_state:
                yield state

        for ii, ts in enumerate(transitions):
            assert len(ts) == batch_size
            assert len(ts) == len(buffers)
            assert len(ts) == len(stacks)

            lefts = []
            rights = []
            for i, (t, buf, stack) in enumerate(zip(ts, buffers, stacks)):
                if t == -1: # skip
                    # Because sentences are padded, we still need to pop here.
                    assert buffers_t[i] >= 0
                    buffers_t[i] -= 1
                elif t == 0: # shift
                    new_stack_item = buf[buffers_t[i]]
                    stack.append(new_stack_item)
                    assert buffers_t[i] >= 0
                    buffers_t[i] -= 1
                elif t == 1: # reduce
                    for lr in [rights, lefts]:
                        if len(stack) > 0:
                            lr.append(stack.pop())
                        else:
                            lr.append(Variable(
                                self.__mod.zeros((hidden_dim,), dtype=self.__mod.float32),
                                volatile=not train))
                else:
                    raise Exception("Action not implemented: {}".format(t))

            assert len(lefts) == len(rights)
            if len(rights) > 0:
                reduced = iter(better_reduce(lefts, rights))
                for i, (t, buf, stack) in enumerate(zip(ts, buffers, stacks)):
                    if t == -1 or t == 0:
                        continue
                    elif t == 1:
                        composition = next(reduced)
                        stack.append(composition)
                    else:
                        raise Exception("Action not implemented: {}".format(t))

        ret = F.stack([s.pop() for s in stacks], axis=0)
        assert ret.shape == (batch_size, hidden_dim)

        return ret


class SentencePairModel(Chain):
    def __init__(self, model_dim, word_embedding_dim,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
                 keep_rate,
                 gpu=-1,
                ):
        super(SentencePairModel, self).__init__(
            projection=L.Linear(word_embedding_dim, model_dim, nobias=True),
            x2h=SPINN(model_dim, gpu=gpu, keep_rate=keep_rate),
            batch_norm_0=L.BatchNormalization(model_dim*2, model_dim*2),
            batch_norm_1=L.BatchNormalization(mlp_dim, mlp_dim),
            l0=L.Linear(model_dim*2, mlp_dim),
            l1=L.Linear(mlp_dim, num_classes)
        )
        self.classifier = CrossEntropyClassifier(gpu)
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.initial_embeddings = initial_embeddings
        self.keep_rate = keep_rate
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim

    def __call__(self, sentences, transitions, y_batch=None, train=True):
        ratio = 1 - self.keep_rate

        # Get Embeddings
        sentences = self.initial_embeddings.take(sentences, axis=0
            ).astype(np.float32)

        # Reshape sentences
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)
        x = Variable(x, volatile=not train)

        batch_size, seq_length = x.shape[0], x.shape[1]

        x = F.dropout(x, ratio=ratio, train=train)

        # Pass embeddings through projection layer, so that they match
        # the dimensions in the output of the compose/reduce function.
        x = F.reshape(x, (batch_size * seq_length, self.word_embedding_dim))
        x = self.projection(x)
        x = F.reshape(x, (batch_size, seq_length, self.model_dim))

        # Extract Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        # Pass through Sentence Encoders.
        h_both = self.x2h(x, t, train=train)
        h_premise, h_hypothesis = F.split_axis(h_both, 2, axis=0)
        
        # Pass through MLP Classifier.
        h = F.concat([h_premise, h_hypothesis], axis=1)
        h = self.batch_norm_0(h, test=not train)
        h = F.dropout(h, ratio, train)
        h = F.relu(h)
        h = self.l0(h)
        h = self.batch_norm_1(h, test=not train)
        h = F.dropout(h, ratio, train)
        h = F.relu(h)
        h = self.l1(h)
        y = h

        # Calculate Loss & Accuracy.
        accum_loss = self.classifier(y, Variable(y_batch, volatile=not train), train)
        self.accuracy = self.accFun(y, self.__mod.array(y_batch))

        return y, accum_loss
