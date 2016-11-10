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
import argparse
import itertools

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


# Chainer already has a method for moving a variable to/from GPU in-place,
# but that messes with backpropagation. So I do it with a copy. Note that
# cuda.get_device() actually returns the dummy device, not the current one
# -- but F.copy will move it to the active GPU anyway (!)
def to_gpu(var, gpu=-1):
    return F.copy(var, gpu)


def to_cpu(var):
    return F.copy(var, -1)


def arr_to_gpu(arr, gpu=-1):
    if gpu >= 0:
        return cuda.to_gpu(arr)
    else:
        return arr


class LSTMState:
    """Class for intelligent LSTM state object.

    It can be initialized from either a tuple ``(c, h)`` or a single variable
    `both`, and provides lazy attribute access to ``c``, ``h``, and ``both``.
    Since the SPINN conducts all LSTM operations on GPU and all tensor
    shuffling on CPU, ``c`` and ``h`` are automatically moved to GPU while
    ``both`` is automatically moved to CPU.

    Args:
        inpt: Either a tuple of ~chainer.Variable objects``(c, h)`` or a single
        concatenated ~chainer.Variable containing both.

    Attributes:
        c (~chainer.Variable): LSTM memory state, moved to GPU if necessary.
        h (~chainer.Variable): LSTM hidden state, moved to GPU if necessary.
        both (~chainer.Variable): Concatenated LSTM state, moved to CPU if
            necessary.

    """
    def __init__(self, inpt):
        if isinstance(inpt, tuple):
            self._c, self._h = inpt
        else:
            self._both = inpt
            self.size = inpt.data.shape[1] // 2

    @property
    def h(self):
        if not hasattr(self, '_h'):
            self._h = to_gpu(self._both[:, self.size:])
        return self._h

    @property
    def c(self):
        if not hasattr(self, '_c'):
            self._c = to_gpu(self._both[:, :self.size])
        return self._c

    @property
    def both(self):
        if not hasattr(self, '_both'):
            self._both = F.concat(
                (to_cpu(self._c), to_cpu(self._h)), axis=1)
        return self._both


def get_c(state, hidden_dim):
    return state[:, hidden_dim:]

def get_h(state, hidden_dim):
    return state[:, :hidden_dim]

def get_state(c, h):
    return F.concat([h, c], axis=1)


def bundle(lstm_iter):
    """Bundle an iterable of concatenated LSTM states into a batched LSTMState.

    Used between CPU and GPU computation. Reversed by :func:`~unbundle`.

    Args:
        lstm_iter: Iterable of ``B`` ~chainer.Variable objects, each with
            shape ``(1,2*S)``, consisting of ``c`` and ``h`` concatenated on
            axis 1.

    Returns:
        state: :class:`~LSTMState` object, with ``c`` and ``h`` attributes
            each with shape ``(B,S)``.
    """
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return LSTMState(F.concat(lstm_iter, axis=0))


def unbundle(state):
    """Unbundle a batched LSTM state into a tuple of concatenated LSTM states.

    Used between GPU and CPU computation. Reversed by :func:`~bundle`.

    Args:
        state: :class:`~LSTMState` object, with ``c`` and ``h`` attributes
            each with shape ``(B,S)``, or an ``inpt`` to
            :func:`~LSTMState.__init__` that would produce such an object.

    Returns:
        lstm_iter: Iterable of ``B`` ~chainer.Variable objects, each with
            shape ``(1,2*S)``, consisting of ``c`` and ``h`` concatenated on
            axis 1.
    """
    if state is None:
        return itertools.repeat(None)
    if not isinstance(state, LSTMState):
        state = LSTMState(state)
    return F.split_axis(
        state.both, state.both.data.shape[0], axis=0, force_tuple=True)


class Embed(Chain):
    """Flexible word embedding module for SPINN.

    It supports either word vectors trained from scratch or word vectors
    defined by a trainable projection layer on top of fixed embeddings.

    Args:
        size: The size of the model state; word vectors are treated as LSTM
            states with both ``c`` and ``h`` so their dimension is ``2*size``.
        vocab_size: The size of the vocabulary.
        dropout: The drop fraction for dropout on the word vectors.
        vectors: None, or an ~numpy.ndarray containing the fixed embeddings.
        normalization: A normalization link class (typically
            ~chainer.links.BatchNormalization).
    """

    def __init__(self, size, vocab_size, dropout, vectors, normalization=None,
                 make_buffers=True, activation=None):
        size = 2 * size if make_buffers else size
        if vectors is None:
            super(Embed, self).__init__(embed=L.EmbedID(vocab_size, size))
        else:
            super(Embed, self).__init__(projection=L.Linear(vectors.shape[1], size))
        # self.add_link('normalization', normalization(size))
        self.vectors = vectors
        self.dropout = dropout
        self.make_buffers = make_buffers
        self.activation = (lambda x: x) if activation is None else activation

    def __call__(self, tokens):
        """Embed a tensor of tokens into a list of SPINN buffers.

        Returns the embeddings as a list of lists of ~chainer.Variable objects
        each augmented with ``tokens`` and ``transitions`` attributes, where
        each buffer (list of variables) is reversed to provide queue-like
        ``pop()`` behavior.

        Args:
            tokens (~chainer.Variable): Tensor of token ids with shape
                ``(B,T)``

        Returns:
            buffers: List of ``B`` lists, each of which contains ``T``
                variables in reverse timestep order, each of which contains
                the embedding of the corresponding token along with ``tokens``
                and ``transitions`` attributes.
        """
        b, l = tokens.data.shape
        if self.vectors is None:
            embeds = self.embed(to_gpu(F.reshape(tokens, (-1,))))
        else:
            embeds = Variable(
                arr_to_gpu(
                    self.vectors.take(tokens.data.ravel(), axis=0)),
                volatile=tokens.volatile)
            embeds = self.projection(embeds)
        if not self.make_buffers:
            return self.activation(F.reshape(embeds, (b, l, -1)))
        # embeds = self.normalization(embeds, embeds.volatile == 'on')
        # embeds = F.dropout(embeds, self.dropout, embeds.volatile == 'off')
        # if not self.make_buffers:
        #     return F.reshape(embeds, (b, l, -1))
        embeds = F.split_axis(to_cpu(embeds), b, axis=0, force_tuple=True)
        embeds = [F.split_axis(x, l, axis=0, force_tuple=True) for x in embeds]
        buffers = [list(reversed(x)) for x in embeds]
        for ex, buf in zip(list(tokens), buffers):
            for tok, var in zip(ex, reversed(buf)):
                var.tokens = [tok]
                var.transitions = [0]
        return buffers


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
        self.optimizer = optimizers.Adam(alpha=0.0003, beta1=0.9, beta2=0.999, eps=1e-08)
        # self.optimizer = optimizers.SGD(lr=0.001)
        self.optimizer.setup(self.model)
        # self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))
        # self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.00003))


def treelstm(c_left, c_right, gates):
    hidden_dim = c_left.shape[1]

    assert gates.shape[1] == hidden_dim * 5, "Need to have 5 gates."

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
    c_t = fl_gate * c_left + fr_gate * c_right + i_gate * cell_inp
    h_t = o_gate * F.tanh(c_t)

    return get_state(c_t, h_t)


class Reduce(Chain):
    """TreeLSTM composition module for SPINN.

    The TreeLSTM has two to four inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model; the fourth is an optional
    attentional input.

    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
        attend (bool): Whether to accept an additional attention input.
        attn_fn (function): A callback function to compute the attention value
            given the left, right, tracker, and attention inputs. TODO
    """

    def __init__(self, size, tracker_size=None, attend=False, attn_fn=None):
        super(Reduce, self).__init__(
            left=L.Linear(size, 5 * size),
            right=L.Linear(size, 5 * size, nobias=True))
        if tracker_size is not None:
            self.add_link('track',
                          L.Linear(tracker_size, 5 * size, nobias=True))
        if attend:
            self.add_link('attend', L.Linear(size, 5 * size, nobias=True))
        self.attn_fn = lambda l, r, t, a: a if attn_fn is None else attn_fn

    def __call__(self, left_in, right_in, tracking=None, attend=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.

        The TreeLSTM has two to four inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model; the fourth is an
        optional attentional input. All are provided as iterables and batched
        internally into tensors.

        Additionally augments each new node with pointers to its children as
        well as concatenated attributes ``transitions`` and ``tokens`` and
        the states of the buffer, stack, and tracker prior to the first SHIFT
        of a token that is part of the node. This allows restarting the
        encoding process with modifications to a particular node.

        Args:
            left_in: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.
            attend: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the attention state to be fed
                into each node in the batch, or None.

        Returns:
            out: Tuple of ``B`` ~chainer.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left``, ``right``,
                ``tokens``, ``transitions``, ``buf``, ``stack``, and
                ``tracking`` attributes.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking, attend = bundle(tracking), bundle(attend)
        lstm_in = self.left(left.h)
        lstm_in += self.right(right.h)
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking.h)
        if hasattr(self, 'attend'):
            lstm_in += self.attend(attend.h)
        out = unbundle(treelstm(left.c, right.c, lstm_in))
        for o, l, r in zip(out, left_in, right_in):
            o.left, o.right = l, r
            # o.left.parent, o.right.parent = o, o
            o.transitions = o.left.transitions + o.right.transitions + [1]
            o.tokens = o.left.tokens + o.right.tokens
            o.buf = o.left.buf
            o.stack = o.left.stack
            o.tracking = o.left.tracking
        return out


class Tracker(Chain):

    def __init__(self, size, tracker_size, predict):
        super(Tracker, self).__init__(
            lateral=L.Linear(tracker_size, 4 * tracker_size),
            buf=L.Linear(size, 4 * tracker_size, nobias=True),
            stack1=L.Linear(size, 4 * tracker_size, nobias=True),
            stack2=L.Linear(size, 4 * tracker_size, nobias=True))
        if predict:
            self.add_link('transition', L.Linear(tracker_size, 3))
        self.state_size = tracker_size
        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, bufs, stacks):
        self.batch_size = len(bufs)
        buf = bundle(buf[-1] for buf in bufs)
        stack1 = bundle(stack[-1] for stack in stacks)
        stack2 = bundle(stack[-2] for stack in stacks)
        lstm_in = self.buf(buf.h)
        lstm_in += self.stack1(stack1.h)
        lstm_in += self.stack2(stack2.h)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            self.c = Variable(
                self.xp.zeros((self.batch_size, self.state_size),
                              dtype=lstm_in.data.dtype),
                volatile='auto')
        self.c, self.h = F.lstm(self.c, lstm_in)
        if hasattr(self, 'transition'):
            return self.transition(self.h)
        return None

    @property
    def states(self):
        return unbundle((self.c, self.h))

    @states.setter
    def states(self, state_iter):
        if state_iter is not None:
            state = bundle(state_iter)
            self.c, self.h = state.c, state.h


class SPINN(Chain):

    def __init__(self, args, vocab, normalization=L.BatchNormalization,
                 attention=False, attn_fn=None):
        super(SPINN, self).__init__(
            embed=Embed(args.size, vocab.size, args.embed_dropout,
                        vectors=vocab.vectors, normalization=normalization),
            reduce=Reduce(args.size, args.tracker_size, attention, attn_fn))
        if args.tracker_size is not None:
            self.add_link('tracker', Tracker(
                args.size, args.tracker_size,
                predict=args.transition_weight is not None))
        self.transition_weight = args.transition_weight

    def __call__(self, example, attention=None, print_transitions=False):
        self.bufs = self.embed(example.tokens)
        if not hasattr(self.embed, 'wh'):
            self.embed.add_param('wh', self.bufs[0][0].data.shape)
            # initializers.init_weight(self.embed.wh.data,
            #                          initializers.Normal(1.0))
            self.embed.wh.to_cpu()
            self.embed.wh.zerograd()
            self.embed.wh.tokens = ['<X>']
            self.embed.wh.transitions = [0]
        # prepend with NULL NULL:
        self.stacks = [[buf[0], buf[0]] for buf in self.bufs]
        for stack, buf in zip(self.stacks, self.bufs):
            for ss in stack:
                ss.buf = buf[:]
                ss.stack = stack[:]
                ss.tracking = None
        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if hasattr(example, 'transitions'):
            self.transitions = example.transitions
        self.attention = attention
        return self.run()

    def run(self, print_transitions=False, run_internal_parser=False,
            use_internal_parser=False):
        # how to use:
        # encoder.bufs = bufs, unbundled
        # encoder.stacks = stacks, unbundled
        # encoder.tracker.state = trackings, unbundled
        # encoder.transitions = ExampleList of Examples, padded with n
        # encoder.run()
        self.history = [[] for buf in self.bufs]
        if hasattr(self, '_hist_tensor'):
            del self._hist_tensor

        transition_loss, transition_acc = 0, 0
        if hasattr(self, 'transitions'):
            num_transitions = self.transitions.shape[1]
        else:
            num_transitions = len(self.bufs[0]) * 2 - 3
        for i in range(num_transitions):
            if hasattr(self, 'transitions'):
                transitions = self.transitions[:, i]
                transition_arr = list(transitions)
            # else:
            #     transition_arr = [0]*len(self.bufs)
            if hasattr(self, 'tracker'):
                transition_hyp = self.tracker(self.bufs, self.stacks)
                if transition_hyp is not None and run_internal_parser:
                    transition_hyp = to_cpu(transition_hyp)
                    transition_preds = transition_hyp.data.argmax(axis=1)
                    if hasattr(self, 'transitions'):
                        transition_loss += F.softmax_cross_entropy(
                            transition_hyp, transitions.tensor,
                            normalize=False)
                        transition_acc += F.accuracy(
                            transition_hyp, transitions.tensor, ignore_label=2)
                    if use_internal_parser:
                        transition_arr = [[0, 1, -1][x] for x in
                                          transition_preds.tolist()]

            lefts, rights, trackings, attentions = [], [], [], []
            batch = zip(transition_arr, self.bufs, self.stacks, self.history,
                        self.tracker.states if hasattr(self, 'tracker')
                        else itertools.repeat(None),
                        self.attention if self.attention is not None
                        else itertools.repeat(None))

            assert len(transition_arr) == len(self.bufs)
            assert len(self.stacks) == len(self.bufs)

            for transition, buf, stack, history, tracking, attention in batch:
                must_shift = len(stack) < 2

                if transition == 0: # shift
                    buf[-1].buf = buf[:]
                    buf[-1].stack = stack[:]
                    buf[-1].tracking = tracking
                    stack.append(buf.pop())
                    history.append(stack[-1])
                elif transition == 1: # reduce
                    for lr in [rights, lefts]:
                        if len(stack) > 0:
                            lr.append(stack.pop())
                        else:
                            zeros = buf[0]
                            zeros.buf = buf[:]
                            zeros.stack = stack[:]
                            zeros.tracking = tracking
                            lr.append(zeros)
                    trackings.append(tracking)
                    attentions.append(attention)
                else:
                    history.append(buf[-1])  # pad history so it can be stacked/transposed
            if len(rights) > 0:
                reduced = iter(self.reduce(
                    lefts, rights, trackings, attentions))
                for transition, stack, history in zip(
                        transition_arr, self.stacks, self.history):
                    if transition == 1: # reduce
                        stack.append(next(reduced))
                        history.append(stack[-1])
        if print_transitions:
            print()
        if self.transition_weight is not None and transition_loss is not 0:
            reporter.report({'accuracy': transition_acc / num_transitions,
                             'loss': transition_loss / num_transitions}, self)
            transition_loss *= self.transition_weight

        return [stack.pop() for stack in self.stacks], transition_loss

    @property
    def histories(self):
        #just h for now
        return [to_gpu(F.stack(tuple(var[:, var.data.shape[1] // 2:] for var in b), axis=1)) for b in self.history]

    @property
    def hist_tensor(self):
        if not hasattr(self, '_hist_tensor'):
            self._hist_tensor = to_gpu(F.concat((F.stack(tuple(var[:, var.data.shape[1] // 2:] for var in b), axis=1) for b in self.history), axis=0))
        return self._hist_tensor


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

        args = {
            'size': model_dim/2,
            'embed_dropout': 1 - keep_rate,
            'tracker_size': None,
            'transition_weight': None,
        }
        args = argparse.Namespace(**args)

        vocab = {
            'size': initial_embeddings.shape[0],
            'vectors': initial_embeddings,
        }
        vocab = argparse.Namespace(**vocab)

        self.add_link('spinn', SPINN(args, vocab, normalization=L.BatchNormalization,
                 attention=False, attn_fn=None))

    def __call__(self, sentences, transitions, y_batch=None, train=True):
        batch_size = sentences.shape[0]

        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)
        example = {
            'tokens': Variable(x, volatile=not train),
            # 'transitions': Variable(t, volatile=not train),
            'transitions': t
        }
        example = argparse.Namespace(**example)

        h_both, transition_acc = self.spinn(example)

        h_premise = F.concat(h_both[:batch_size], axis=0)
        h_hypothesis = F.concat(h_both[batch_size:], axis=0)

        # h_premise, h_hypothesis = F.split_axis(h_both, 2, axis=0)

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

        return y, accum_loss, self.accuracy.data, transition_acc
