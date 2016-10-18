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

from spinn.util.chainer_blocks import LSTM, LSTMChain, RNNChain, EmbedChain
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
- [x] Enable evaluation. Currently crashing.
- [x] Use the right C and H units for the TreeLSTM.
- [ ] Confirm that volatile is working correctly during eval time.
      Time the eval with and without volatile being set. Full eval
      takes about 2m to complete on AD Mac.

Other Tasks:

- [ ] Enable "transition validation".
- [ ] Enable TreeGRU as alternative option to TreeLSTM.
- [ ] Add TrackingLSTM.
- [ ] Run CBOW and RNN models for comparison.

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

class SLSTMChain(Chain):
    def __init__(self, input_dim, hidden_dim, seq_length, prefix="SLSTMChain", gpu=-1):
        super(SLSTMChain, self).__init__(
            i_l_fwd=L.Linear(hidden_dim, 4 * hidden_dim, nobias=True),
            h_l_fwd=L.Linear(hidden_dim, 4 * hidden_dim),
            i_r_fwd=L.Linear(hidden_dim, 4 * hidden_dim, nobias=True),
            h_r_fwd=L.Linear(hidden_dim, 4 * hidden_dim),
        )
        self.seq_length = seq_length
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

        # TODO: Check that we have all zeros or None for c/h
        # on the very first step. In other words, ensure that
        # states aren't being carried over between batches.
        for el in left_x + right_x:
            if not hasattr(el, 'c'):
                el.c = Variable(self.__mod.zeros((1, self.hidden_dim), dtype=self.__mod.float32))
            if not hasattr(el, 'h'):
                el.h = Variable(self.__mod.zeros((1, self.hidden_dim), dtype=self.__mod.float32))

        # TODO: Keep states between batches. Need to use some crazy pointer
        # system in order to have dynamic batch sizes.
        c_l = F.concat([el.c for el in left_x], axis=0)
        c_r = F.concat([el.c for el in right_x], axis=0)
        h_l = F.concat([el.h for el in left_x], axis=0)
        h_r = F.concat([el.h for el in right_x], axis=0)

        left = F.reshape(F.concat(left_x, axis=0), (-1, self.hidden_dim))
        right = F.reshape(F.concat(right_x, axis=0), (-1, self.hidden_dim))

        assert left.shape == right.shape, "Left and Right must match in dimensions."

        il = self.i_l_fwd(left)
        ir = self.i_r_fwd(right)
        hl = self.h_l_fwd(h_l)
        hr = self.h_r_fwd(h_r)
        ihl = il + hl
        ihr = ir + hr
        c, h = F.slstm(c_l, c_r, ihl, ihr)

        return c, h

class LSTM_TI(Chain):
    def __init__(self, input_dim, hidden_dim, seq_length, prefix="LSTM_TI", gpu=-1):
        super(LSTM_TI, self).__init__(
            reduce=SLSTMChain(input_dim, hidden_dim, seq_length, gpu=gpu),
        )
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        buff_type, trans_type = in_types

        type_check.expect(
            buff_type.dtype == 'f',
            buff_type.ndim >= 1,
            trans_type.dtype == 'i',
            trans_type.ndim >= 1,
        )

    def __call__(self, buffers, transitions, train=True, keep_hs=False):
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
        in_data = tuple([x.data for x in [buffers, transitions]])
        in_types = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_types)
        # END: Type Check

        batch_size, seq_length = buffers.shape[0], buffers.shape[1]
        # c_prev_l, c_prev_r, h_prev_l, h_prev_r = [Variable(self.__mod.zeros(
        #     (batch_size, self.hidden_dim), dtype=self.__mod.float32))
        #     for _ in range(4)]

        transitions = transitions.data.T

        # MAYBE: Initialize stack with None, None in case of initial reduces.
        stacks = [[] for _ in range(batch_size)]
        buffers = [list(b) for b in buffers]

        def pseudo_reduce(lefts, rights):
            for l, r in zip(lefts, rights):
                yield l + r

        def better_reduce(lefts, rights):
            c, h = self.reduce(lefts, rights, train=train)
            for hh in h:
                yield hh

        for ii, ts in enumerate(transitions):
            lefts = []
            rights = []
            for i, (t, buf, stack) in enumerate(zip(ts, buffers, stacks)):
                if t == -1:
                    continue
                elif t == 0: # shift
                    stack.append(buf.pop())
                elif t == 1: # reduce
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
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

        # TODO: It would be good to check that the buffer has been
        # fully consumed.
        # for stack in stacks:
        #     assert len(stack) == 1

        # Flatten List of Lists.
        # Goes from 3-D:`B x 1 x H` to 1-D:`(B * H)`
        stacks = F.concat(zip(*stacks)[0], axis=0)

        # Goes from 1-D:`(B * H)` to 2-D:`B x H`
        stacks = F.reshape(stacks, (batch_size, self.hidden_dim))

        c = None
        # h = self.__mod.array(stacks)
        h = stacks
        hs = None

        return c, h, hs

class SPINN(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 seq_length,
                 initial_embeddings,
                 keep_rate,
                 prefix="SPINN",
                 gpu=-1
                 ):
        super(SPINN, self).__init__(
            spinn=LSTM_TI(word_embedding_dim, model_dim, seq_length, gpu=gpu),
        )
            # batch_norm=L.BatchNormalization(model_dim, model_dim)

        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.keep_rate = keep_rate
        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim

    def __call__(self, buffers, transitions, train=True):
        ratio = 1 - self.keep_rate

        # One of our goals or invariants is to maintain lists of lists
        # for sentences rather than a 3D tensor of batch x sent x emb.
        # buffers = [list(Variable(
        #     self.__mod.array(xx, dtype=self.__mod.float32))) for xx in x]

        # gamma = Variable(self.__mod.array(1.0, dtype=self.__mod.float32), volatile=not train, name='gamma')
        # beta = Variable(self.__mod.array(0.0, dtype=self.__mod.float32),volatile=not train, name='beta')
        # x = batch_normalization(x, gamma, beta, eps=2e-5, running_mean=None,running_var=None, decay=0.9, use_cudnn=False)
        # x = self.batch_norm(x)
        # x = F.dropout(x, ratio, train)

        c, h, hs = self.spinn(buffers, transitions, train)
        return h

class SentencePairModel(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length, initial_embeddings, num_classes,
                 mlp_dim,
                 keep_rate,
                 gpu=-1,
                 ):
        super(SentencePairModel, self).__init__(
            embed=EmbedChain(word_embedding_dim, vocab_size, initial_embeddings, model_dim, gpu=gpu),
            projection=L.Linear(word_embedding_dim, model_dim),
            x2h_premise=SPINN(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings, gpu=gpu, keep_rate=keep_rate),
            x2h_hypothesis=SPINN(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings, gpu=gpu, keep_rate=keep_rate),
            h2y=MLP(dimensions=[model_dim*2, mlp_dim, mlp_dim/2, num_classes],
                    keep_rate=keep_rate, gpu=gpu),
        )
        self.classifier = CrossEntropyClassifier(gpu)
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.keep_rate = keep_rate
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim

    def __call__(self, sentences, transitions, y_batch=None, train=True):
        ratio = 1 - self.keep_rate

        # Get Embeddings
        x_prem = self.embed(Variable(sentences[0]))
        x_hyp = self.embed(Variable(sentences[1]))

        batch_size, seq_length = x_prem.shape[0], x_prem.shape[1]

        # Pass embeddings through projection layer, so that they match
        # the dimensions in the output of the compose/reduce function.
        x_prem = F.reshape(x_prem, (batch_size * seq_length, self.word_embedding_dim))
        x_prem = self.projection(x_prem)
        x_prem = F.reshape(x_prem, (batch_size, seq_length, self.model_dim))
        x_hyp = F.reshape(x_hyp, (batch_size * seq_length, self.word_embedding_dim))
        x_hyp = self.projection(x_hyp)
        x_hyp = F.reshape(x_hyp, (batch_size, seq_length, self.model_dim))

        # Extract Transitions
        t_prem = Variable(transitions[0])
        t_hyp = Variable(transitions[1])

        # Pass through Sentence Encoders.
        h_premise = self.x2h_premise(x_prem, t_prem, train=train)
        h_hypothesis = self.x2h_hypothesis(x_hyp, t_hyp, train=train)
        
        # Pass through Classifier.
        h = F.concat([h_premise, h_hypothesis], axis=1)
        h = F.dropout(h, ratio, train)
        y = self.h2y(h, train)

        # Calculate Loss & Accuracy.
        y = F.dropout(y, ratio, train)
        accum_loss = self.classifier(y, Variable(y_batch), train)
        self.accuracy = self.accFun(y, self.__mod.array(y_batch))

        return y, accum_loss
