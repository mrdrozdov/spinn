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

from spinn.util.chainer_blocks import BaseSentencePairTrainer

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


def HeKaimingInit(shape, real_shape=None):
    # Calculate fan-in / fan-out using real shape if given as override
    fan = real_shape or shape

    return np.random.normal(scale=np.sqrt(4.0/(fan[0] + fan[1])),
                            size=shape)

def get_c(state, hidden_dim):
    return state[:, hidden_dim:]

def get_h(state, hidden_dim):
    return state[:, :hidden_dim]

def get_state(c, h):
    return F.concat([h, c], axis=1)


class SentencePairTrainer(BaseSentencePairTrainer):
    def init_params(self, **kwargs):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            if len(data.shape) >= 2:
                data[:] = HeKaimingInit(data.shape)
            else:
                data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, lr=0.01, **kwargs):
        # self.optimizer = optimizers.Adam(alpha=0.0003, beta1=0.9, beta2=0.999, eps=1e-08)
        self.optimizer = optimizers.SGD(lr=0.001)
        self.optimizer.setup(self.model)
        # self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))
        # self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.00003))


class TreeLSTMChain(Chain):
    def __init__(self, hidden_dim, tracking_lstm_hidden_dim, use_external=False, prefix="TreeLSTMChain", gpu=-1):
        super(TreeLSTMChain, self).__init__(
            W_l=L.Linear(hidden_dim / 2, hidden_dim / 2 * 5, nobias=True),
            W_r=L.Linear(hidden_dim / 2, hidden_dim / 2 * 5, nobias=True),
            b=L.Bias(axis=1, shape=(hidden_dim / 2 * 5,)),
            )
        assert hidden_dim % 2 == 0, "The hidden_dim must be even because contains c and h."
        self.hidden_dim = hidden_dim / 2
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.use_external = use_external

        if use_external:
            self.add_link('W_external', L.Linear(tracking_lstm_hidden_dim / 2, hidden_dim / 2 * 5))

    def __call__(self, l_prev, r_prev, external=None, train=True, keep_hs=False):
        hidden_dim = self.hidden_dim

        l_h_prev = get_h(l_prev, hidden_dim)
        l_c_prev = get_c(l_prev, hidden_dim)
        r_h_prev = get_h(r_prev, hidden_dim)
        r_c_prev = get_c(r_prev, hidden_dim)

        gates = self.b(self.W_l(l_h_prev) + self.W_r(r_h_prev))

        if external is not None:
            gates += self.W_external(external)

        def slice_gate(gate_data, i):
            return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

        # Compute and slice gate values
        i_gate, fl_gate, fr_gate, o_gate, cell_inp = \
            [slice_gate(gates, i) for i in range(5)]

        # Apply nonlinearities
        i_gate = F.sigmoid(i_gate)
        fl_gate = F.sigmoid(fl_gate)
        fr_gate = F.sigmoid(fr_gate)
        o_gate = F.sigmoid(o_gate)
        cell_inp = F.tanh(cell_inp)

        # Compute new cell and hidden value
        c_t = fl_gate * l_c_prev + fr_gate * r_c_prev + i_gate * cell_inp
        h_t = o_gate * F.tanh(c_t)

        return get_state(c_t, h_t)


class ReduceChain(Chain):
    def __init__(self, hidden_dim, tracking_lstm_hidden_dim, use_external=False, prefix="ReduceChain", gpu=-1):
        super(ReduceChain, self).__init__(
            treelstm=TreeLSTMChain(hidden_dim, tracking_lstm_hidden_dim, use_external=use_external),
        )
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def __call__(self, left_x, right_x, external=None, train=True, keep_hs=False):
        """
        Args:
            left_x:  B* x H
            right_x: B* x H
        Returns:
            final_state: B* x H
        """

        assert len(left_x) == len(right_x)
        batch_size = len(left_x)

        # Concatenate the list of states.
        left_x = F.concat(left_x, axis=0)
        right_x = F.concat(right_x, axis=0)
        assert left_x.shape == right_x.shape, "Left and Right must match in dimensions."

        assert left_x.shape[1] % 2 == 0, "Unit dim needs to be even because is concatenated [c,h]"
        unit_dim = left_x.shape[1]
        h_dim = unit_dim / 2

        # Split each state into its c/h representations.
        lstm_state = self.treelstm(left_x, right_x, external)
        return lstm_state


class TrackingLSTM(Chain):
    def __init__(self, hidden_dim, tracking_lstm_hidden_dim, num_actions=2, make_logits=False):
        super(TrackingLSTM, self).__init__(
            W_x=L.Linear(tracking_lstm_hidden_dim / 2, tracking_lstm_hidden_dim / 2 * 4),
            W_h=L.Linear(tracking_lstm_hidden_dim / 2, tracking_lstm_hidden_dim / 2 * 4),
            bias=L.Bias(axis=1, shape=(tracking_lstm_hidden_dim / 2 * 4,)),
        )
        # TODO: TrackingLSTM should be able to be a size different from hidden_dim.
        if make_logits:
            self.add_link('logits', L.Linear(tracking_lstm_hidden_dim / 2, num_actions))

    def __call__(self, c, h, x, train=True):
        gates = self.bias(self.W_x(x) + self.W_h(h))
        c, h = F.lstm(c, gates)
        if hasattr(self, 'logits'):
            logits = self.logits(h)
        else:
            logits = 0.0
        return c, h, logits


class TrackingInput(Chain):
    def __init__(self, hidden_dim, tracking_lstm_hidden_dim, num_inputs=3):
        super(TrackingInput, self).__init__(
            W_stack_0=L.Linear(hidden_dim / 2, tracking_lstm_hidden_dim / 2),
            W_stack_1=L.Linear(hidden_dim / 2, tracking_lstm_hidden_dim / 2),
            W_buffer=L.Linear(hidden_dim / 2, tracking_lstm_hidden_dim / 2),
        )
        assert hidden_dim % 2 == 0, "The hidden_dim must be even because contains c and h."
        self.hidden_dim = hidden_dim / 2

    def __call__(self, stacks, buffers, buffers_t, train=True):
        zeros = Variable(self.xp.zeros((1, self.hidden_dim,), dtype=self.xp.float32),
                                volatile=not train)
        state = self.W_stack_0(F.concat([get_h(s[0], self.hidden_dim) if 0 < len(s) else zeros for s in stacks], axis=0))
        state += self.W_stack_1(F.concat([get_h(s[1], self.hidden_dim) if 1 < len(s) else zeros for s in stacks], axis=0))
        state += self.W_buffer(F.concat([get_h(b[i], self.hidden_dim) if i < len(b) else zeros for (b, i) in zip(buffers, buffers_t)], axis=0))
        return state


class SPINN(Chain):
    def __init__(self, hidden_dim, keep_rate, prefix="SPINN", gpu=-1,
                 tracking_lstm_hidden_dim=4, use_tracking_lstm=True, make_logits=False,
                 use_shift_composition=True):
        super(SPINN, self).__init__(
            reduce=ReduceChain(hidden_dim, tracking_lstm_hidden_dim, use_external=use_tracking_lstm, gpu=gpu),
        )
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.tracking_lstm_hidden_dim = tracking_lstm_hidden_dim
        self.use_tracking_lstm = use_tracking_lstm
        self.use_shift_composition = use_shift_composition

        if use_tracking_lstm:
            self.add_link('tracking_input', TrackingInput(hidden_dim, tracking_lstm_hidden_dim))
            self.add_link('tracking_lstm', TrackingLSTM(hidden_dim, tracking_lstm_hidden_dim, make_logits=make_logits))

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        buff_type = in_types[0]

        type_check.expect(
            buff_type.dtype == 'f',
            buff_type.ndim >= 1,
        )

    def reset_state(self, batch_size, train):
        zeros = Variable(self.__mod.zeros((1, self.tracking_lstm_hidden_dim/2,), dtype=self.__mod.float32),
                                volatile=not train)
        self.c = [zeros for _ in range(batch_size)]
        self.h = [zeros for _ in range(batch_size)]

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
        in_data = tuple([x.data for x in [buffers]])
        in_types = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_types)
        # END: Type Check

        batch_size, seq_length, hidden_dim = buffers.shape[0], buffers.shape[1], buffers.shape[2]
        transitions = transitions.T
        assert len(transitions) == seq_length

        buffers = [F.split_axis(b, seq_length, axis=0, force_tuple=True)
                    for b in buffers]
        buffers_t = [0 for _ in buffers]

        # Initialize stack with at least one item, otherwise gradient might
        # not propogate.
        stacks = [[] for b in buffers]

        def pseudo_reduce(lefts, rights):
            for l, r in zip(lefts, rights):
                yield l + r

        def better_reduce(lefts, rights, h):
            lstm_state = self.reduce(lefts, rights, h, train=train)
            batch_size = lstm_state.shape[0]
            lstm_state = F.split_axis(lstm_state, batch_size, axis=0, force_tuple=True)
            for state in lstm_state:
                yield state

        for ii, ts in enumerate(transitions):
            assert len(ts) == batch_size
            assert len(ts) == len(buffers)
            assert len(ts) == len(stacks)

            # TODO! The tracking inputs for shifts and reduces should be split,
            # in order to do consecutive shifts. This would (maybe) allow us
            # to get the performance benefits from dynamic batch sizes while still
            # predicting actions.
            if self.use_tracking_lstm:
                if self.use_shift_composition:
                    tracking_input = self.tracking_input(stacks, buffers, buffers_t, train)
                    c = F.concat(self.c, axis=0)
                    h = F.concat(self.h, axis=0)

                    c, h, logits = self.tracking_lstm(c, h, tracking_input, train)

                    # Assign appropriate states after they've been calculated.
                    self.c = F.split_axis(c, c.shape[0], axis=0, force_tuple=True)
                    self.h = F.split_axis(h, h.shape[0], axis=0, force_tuple=True)
                else:
                    tracking_size = len([t for t in ts if t == 1])
                    if tracking_size > 0:
                        tracking_ix = [i for (i, t) in enumerate(ts) if t == 1]
                        tracking_stacks    = [x for (t, x) in zip(ts, stacks) if t == 1]
                        tracking_buffers   = [x for (t, x) in zip(ts, buffers) if t == 1]
                        tracking_buffers_t = [x for (t, x) in zip(ts, buffers_t) if t == 1]

                        tracking_input = self.tracking_input(
                            tracking_stacks,
                            tracking_buffers,
                            tracking_buffers_t,
                            train)

                        c = F.concat([x for (t, x) in zip(ts, self.c) if t == 1], axis=0)
                        h = F.concat([x for (t, x) in zip(ts, self.h) if t == 1], axis=0)

                        c, h, logits = self.tracking_lstm(c, h, tracking_input, train)

                        # Assign appropriate states after they've been calculated.
                        _c = F.split_axis(c, tracking_size, axis=0, force_tuple=True)
                        for i, ix in enumerate(tracking_ix):
                            if t == 1:
                                self.c[ix] = _c[i]
                        _h = F.split_axis(h, tracking_size, axis=0, force_tuple=True)
                        for i, ix in enumerate(tracking_ix):
                            if t == 1:
                                self.h[ix] = _h[i]

            lefts = []
            rights = []
            for i, (t, buf, stack) in enumerate(zip(ts, buffers, stacks)):
                if t == -1: # skip
                    # Because sentences are padded, we still need to pop here.
                    pass
                elif t == 0: # shift
                    new_stack_item = buf[buffers_t[i]]
                    stack.append(new_stack_item)
                    assert buffers_t[i] < seq_length
                    buffers_t[i] += 1
                elif t == 1: # reduce
                    for lr in [rights, lefts]:
                        if len(stack) > 0:
                            lr.append(stack.pop())
                        else:
                            lr.append(Variable(
                                self.__mod.zeros((1, hidden_dim,), dtype=self.__mod.float32),
                                volatile=not train))
                else:
                    raise Exception("Action not implemented: {}".format(t))

            assert len(lefts) == len(rights)
            if len(rights) > 0:
                if self.use_tracking_lstm:
                    external = F.concat([x for (t, x) in zip(ts, self.h) if t == 1], axis=0)
                else:
                    external = None
                reduced = iter(better_reduce(lefts, rights, external))
                for i, (t, buf, stack) in enumerate(zip(ts, buffers, stacks)):
                    if t == -1 or t == 0:
                        continue
                    elif t == 1:
                        composition = next(reduced)
                        stack.append(composition)
                    else:
                        raise Exception("Action not implemented: {}".format(t))

        ret = F.concat([s.pop() for s in stacks], axis=0)
        assert ret.shape == (batch_size, hidden_dim)

        return ret


class SentencePairModel(Chain):
    def __init__(self, model_dim, word_embedding_dim,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
                 keep_rate,
                 gpu=-1,
                 tracking_lstm_hidden_dim=4,
                 use_tracking_lstm=True,
                 use_shift_composition=True,
                 make_logits=False,
                 **kwargs
                ):
        super(SentencePairModel, self).__init__(
            projection=L.Linear(word_embedding_dim, model_dim, nobias=True),
            x2h=SPINN(model_dim,
                tracking_lstm_hidden_dim=tracking_lstm_hidden_dim,
                use_tracking_lstm=use_tracking_lstm,
                use_shift_composition=use_shift_composition,
                make_logits=make_logits,
                gpu=gpu, keep_rate=keep_rate),
            # batch_norm_0=L.BatchNormalization(model_dim*2, model_dim*2),
            # batch_norm_1=L.BatchNormalization(mlp_dim, mlp_dim),
            # batch_norm_2=L.BatchNormalization(mlp_dim, mlp_dim),
            l0=L.Linear(model_dim*2, mlp_dim),
            l1=L.Linear(mlp_dim, mlp_dim),
            l2=L.Linear(mlp_dim, num_classes)
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

        if self.__gpu >= 0:
            x = cuda.to_gpu(x)

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
        self.x2h.reset_state(batch_size, train)
        h_both = self.x2h(x, t, train=train)
        h_premise, h_hypothesis = F.split_axis(h_both, 2, axis=0)

        # Pass through MLP Classifier.
        h = F.concat([h_premise, h_hypothesis], axis=1)
        # h = self.batch_norm_0(h, test=not train)
        # h = F.dropout(h, ratio, train)
        # h = F.relu(h)
        h = self.l0(h)
        # h = self.batch_norm_1(h, test=not train)
        # h = F.dropout(h, ratio, train)
        h = F.relu(h)
        h = self.l1(h)
        # h = self.batch_norm_2(h, test=not train)
        # h = F.dropout(h, ratio, train)
        h = F.relu(h)
        h = self.l2(h)
        y = h

        # Calculate Loss & Accuracy.
        accum_loss = self.classifier(y, Variable(y_batch, volatile=not train), train)
        self.accuracy = self.accFun(y, self.__mod.array(y_batch))

        return y, accum_loss
