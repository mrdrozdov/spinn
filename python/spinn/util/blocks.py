import numpy as np
import random

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


def l2_cost(model, l2_lambda):
    cost = 0.0
    for w in model.parameters():
        cost += l2_lambda * torch.sum(torch.pow(w, 2))
    return cost


def flatten(l):
    if hasattr(l, '__len__'):
        return reduce(lambda x, y: x + flatten(y), l, [])
    else:
        return [l]


def the_gpu():
    return the_gpu.gpu

the_gpu.gpu = -1

def to_cuda(var, gpu):
    if gpu >= 0:
        return var.cuda()
    return var

# Chainer already has a method for moving a variable to/from GPU in-place,
# but that messes with backpropagation. So I do it with a copy. Note that
# cuda.get_device() actually returns the dummy device, not the current one
# -- but F.copy will move it to the active GPU anyway (!)
def to_gpu(var):
    return to_cuda(var, the_gpu())


def to_cpu(var):
    return to_cuda(var, -1)


def arr_to_gpu(arr):
    if the_gpu() >= 0:
        return torch.cuda.FloatTensor(arr)
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
            self.size = inpt.data.size()[1] // 2

    @property
    def h(self):
        if not hasattr(self, '_h'):
            self._h = to_gpu(get_h(self._both, self.size))
        return self._h

    @property
    def c(self):
        if not hasattr(self, '_c'):
            self._c = to_gpu(get_c(self._both, self.size))
        return self._c

    @property
    def both(self):
        if not hasattr(self, '_both'):
            self._both = torch.cat(
                (to_cpu(self._c), to_cpu(self._h)), 1)
        return self._both


def get_h(state, hidden_dim):
    return state[:, hidden_dim:]

def get_c(state, hidden_dim):
    return state[:, :hidden_dim]

def get_state(c, h):
    return torch.cat([h, c], 1)


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
    return LSTMState(torch.cat(lstm_iter, 0))


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
    return torch.chunk(
        state.both, state.both.data.size()[0], 0)


def extract_gates(x, n):
    r = x.view(x.size(0), x.size(1) // n, n)
    return [r[:, :, i] for i in range(n)]


def lstm(c_prev, x):
    a, i, f, o = extract_gates(x, 4)

    a = F.tanh(a)
    i = F.sigmoid(i)
    f = F.sigmoid(f)
    o = F.sigmoid(o)

    c = a * i + f * c_prev
    h = o * F.tanh(c)

    return c, h


def treelstm(c_left, c_right, gates, use_dropout=False):
    hidden_dim = c_left.size()[1]

    assert gates.size()[1] == hidden_dim * 5, "Need to have 5 gates."

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
    i_val = i_gate * cell_inp
    dropout_rate = 0.1
    if use_dropout:
        i_val = F.dropout(i_val, dropout_rate, train=i_val.volatile == False)
    c_t = fl_gate * c_left + fr_gate * c_right + i_val
    h_t = o_gate * F.tanh(c_t)

    return (c_t, h_t)


class BaseSentencePairTrainer(object):

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def forward(self, x_batch, y_batch=None, train=True,
                use_internal_parser=False, validate_transitions=True):
        assert "sentences" in x_batch and "transitions" in x_batch, \
            "Input must contain dictionary of sentences and transitions."

        sentences = x_batch["sentences"]
        transitions = x_batch["transitions"]

        ret = self.model(sentences, transitions, y_batch, train=train,
            use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)
        return ret

    def save(self, filename, step, best_dev_error):
        torch.save({
            'step': step,
            'best_dev_error': best_dev_error,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step'], checkpoint['best_dev_error']


class Embed(nn.Module):
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

    def __init__(self, size, vocab_size, dropout, vectors,
                 make_buffers=True, activation=None,
                 use_input_dropout=False, use_input_norm=False):
        size = 2 * size if make_buffers else size
        super(Embed, self).__init__()
        if vectors is None:
            raise NotImplementedError
        else:
            self.projection = nn.Linear(vectors.shape[1], size)
        self.vectors = vectors
        self.dropout = dropout
        self.make_buffers = make_buffers
        self.activation = (lambda x: x) if activation is None else activation
        self.use_input_dropout = use_input_dropout
        self.use_input_norm = use_input_norm

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
        b, l = tokens.size()[:2]
        if self.vectors is None:
            embeds = self.embed(to_gpu(F.reshape(tokens, (-1,))))
        else:
            embeds = self.vectors.take(tokens.data.cpu().numpy().ravel(), axis=0)
            embeds = to_gpu(Variable(torch.from_numpy(embeds), volatile=tokens.volatile))
            embeds = self.projection(embeds)
        if not self.make_buffers:
            return self.activation(embeds.view(b, l, -1))

        if self.use_input_dropout:
            embeds = F.dropout(embeds, self.dropout, embeds.volatile == 'off')

        embeds = torch.chunk(to_cpu(embeds), b, 0)
        embeds = [torch.chunk(x, l, 0) for x in embeds]
        buffers = [list(reversed(x)) for x in embeds]
        return buffers


class Reduce(nn.Module):
    """TreeLSTM composition module for SPINN.

    The TreeLSTM has two to three inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model.

    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
    """

    def __init__(self, size, tracker_size=None):
        super(Reduce, self).__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        if tracker_size is not None:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def __call__(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.

        The TreeLSTM has two to three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided
        as iterables and batched internally into tensors.

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

        Returns:
            out: Tuple of ``B`` ~chainer.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left``, ``right``,
                ``tokens``, ``transitions``, ``buf``, ``stack``, and
                ``tracking`` attributes.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left.h)
        lstm_in += self.right(right.h)
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking.h)
        out = unbundle(treelstm(left.c, right.c, lstm_in))
        for o, l, r in zip(out, left_in, right_in):
            if hasattr(l, 'buf'):
                o.left, o.right = l, r
                o.buf = o.left.buf
                o.transitions = o.left.transitions + o.right.transitions + [1]
                o.tokens = o.left.tokens + o.right.tokens
                o.stack = o.left.stack
                o.tracking = o.left.tracking
        return out
