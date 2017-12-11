import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
from torch.nn.parameter import Parameter

from spinn.util.misc import recursively_set_device
from functools import reduce


def flatten(l):
    if hasattr(l, '__len__'):
        return reduce(lambda x, y: x + flatten(y), l, [])
    else:
        return [l]


def the_gpu():
    return the_gpu.gpu


the_gpu.gpu = -1


def to_gpu(var):
    if the_gpu.gpu >= 0:
        return var.cuda(the_gpu.gpu)
    return var


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
                (self._c, self._h), 1)
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


class LayerNormalization(nn.Module):
    # From: https://discuss.pytorch.org/t/lstm-with-layer-normalization/2150

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z)
        sigma = torch.std(z)

        ln_out = (z - mu) / (sigma + self.eps)

        ln_out = ln_out * self.a2 + self.b2
        return ln_out


class ReduceTreeGRU(nn.Module):
    """
    Computes the following TreeGRU (x is optional):

    hprev = left + right
    r = sigm(Ur_x(x) + Wr(hprev))
    z = sigm(Uz_x(x) + Wz(hprev))
    c = tanh(Uc_x(x) + V_l(left * r) + V_r(right * r))
    h = (1-z) * hprev + z * c
    or:
    h = hprev + z * (c - hprev)

    Standard GRU would be:

    r = sigm(Ur_x(x) + Wr(hprev))
    z = sigm(Uz_x(x) + Wz(hprev))
    c = tanh(Uc_x(x) + V(hprev * r))
    h = (1-z) * hprev + z * c
    or:
    h = hprev + z * (c - hprev)

    # TODO: Add layer normalization.

    """

    def __init__(self, size, tracker_size=None,
                 use_tracking_in_composition=None):
        super(ReduceTreeGRU, self).__init__()
        self.size = size
        self.W = Linear()(size, 2 * size)
        self.Vl = Linear()(size, size)
        self.Vr = Linear()(size, size)
        if tracker_size is not None and use_tracking_in_composition:
            self.U = Linear()(
                tracker_size,
                3 * size)

    def forward(self, left, right, tracking=None):
        def slice_gate(gate_data, hidden_dim, i):
            return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

        batch_size = len(left)
        size = self.size

        left = torch.cat(left, 0)
        right = torch.cat(right, 0)
        hprev = left + right

        W = self.W(hprev)
        r, z = [slice_gate(W, size, i) for i in range(2)]
        c = 0

        if hasattr(self, "U"):
            tracking = bundle(tracking)
            U = self.U(tracking.h)
            Ur, Uz, Uc = [slice_gate(U, size, i) for i in range(3)]
            r = Ur + r
            z = Uz + z
            c = Uc + c

        r = F.sigmoid(r)
        z = F.sigmoid(z)
        c = F.tanh(c + self.Vl(left * r) + self.Vr(right * r))
        h = hprev + z * (c - hprev)

        return torch.chunk(h, batch_size, 0)


def treelstm(c_left, c_right, gates):
    hidden_dim = c_left.size()[1]

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Compute and slice gate values
    i_gate, fl_gate, fr_gate, o_gate, cell_inp = \
        [slice_gate(gates, i) for i in range(5)]

    # Apply nonlinearities
    i_gate = F.sigmoid(i_gate)
    # Lazy alternative to bias initialization, from Choi
    fl_gate = F.sigmoid(fl_gate + 1.)
    # Lazy alternative to bias initialization, from Choi
    fr_gate = F.sigmoid(fr_gate + 1.)
    o_gate = F.sigmoid(o_gate)
    cell_inp = F.tanh(cell_inp)

    # Compute new cell and hidden value
    i_val = i_gate * cell_inp
    c_t = fl_gate * c_left + fr_gate * c_right + i_val
    h_t = o_gate * F.tanh(c_t)

    return (c_t, h_t)


class ModelTrainer(object):

    def __init__(self, model, optimizer, sparse_optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.sparse_optimizer = sparse_optimizer

    def save(self, filename, step, best_dev_error, best_dev_step):
        if the_gpu() >= 0:
            recursively_set_device(self.model.state_dict(), gpu=-1)
            recursively_set_device(self.optimizer.state_dict(), gpu=-1)

        # Always sends Tensors to CPU.
        save_dict = {
            'step': step,
            'best_dev_error': best_dev_error,
            'best_dev_step': best_dev_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }
        if self.sparse_optimizer is not None:
            save_dict['sparse_optimizer_state_dict'] = self.sparse_optimizer.state_dict()
        torch.save(save_dict, filename)

        if the_gpu() >= 0:
            recursively_set_device(self.model.state_dict(), gpu=the_gpu())
            recursively_set_device(self.optimizer.state_dict(), gpu=the_gpu())

    def load(self, filename, cpu=False):
        if cpu:
            # Load GPU-based checkpoints on CPU
            checkpoint = torch.load(
                filename, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']

        # HACK: Compatability for saving supervised SPINN and loading RL SPINN.
        if 'baseline' in self.model.state_dict().keys(
        ) and 'baseline' not in model_state_dict:
            model_state_dict['baseline'] = torch.FloatTensor([0.0])

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.load_state_dict(checkpoint['sparse_optimizer_state_dict'])

        if 'best_dev_step' in checkpoint:
            best_dev_step = checkpoint['best_dev_step']
        else:
            best_dev_step = 0

        return checkpoint['step'], checkpoint['best_dev_error'], best_dev_step


class Embed(nn.Module):
    def __init__(self, size, vocab_size, vectors, fine_tune=False):
        super(Embed, self).__init__()

        if vectors is None:
            self.embed = nn.Embedding(vocab_size, size)
        else:
            if fine_tune:
                self.embed = nn.Embedding(vocab_size, size, sparse=True)
                self.embed.weight.data.copy_(torch.from_numpy(vectors))
            else:
                self.vectors = vectors
                self.embed = None

    def forward(self, tokens):
        if self.embed is not None:
            embeds = self.embed(tokens.contiguous().view(-1).long())
        else:
            embeds = self.vectors.take(
                tokens.data.cpu().numpy().ravel(), axis=0)
            embeds = to_gpu(
                Variable(
                    torch.from_numpy(embeds),
                    volatile=tokens.volatile))
        return embeds


class GRU(nn.Module):
    def __init__(self, inp_dim, model_dim, num_layers=1,
                 reverse=False, bidirectional=False, dropout=None):
        super(GRU, self).__init__()
        self.model_dim = model_dim
        self.reverse = reverse
        self.bidirectional = bidirectional
        self.bi = 2 if self.bidirectional else 1
        self.num_layers = num_layers
        self.rnn = nn.GRU(inp_dim, model_dim / self.bi, num_layers=num_layers,
                          batch_first=True,
                          bidirectional=self.bidirectional)

    def forward(self, x, h0=None):
        bi = self.bi
        num_layers = self.num_layers
        batch_size, seq_len = x.size()[:2]
        model_dim = self.model_dim

        if self.reverse:
            x = reverse_tensor(x, dim=1)

        # Initialize state unless it is given.
        if h0 is None:
            h0 = to_gpu(
                Variable(
                    torch.zeros(
                        num_layers * bi,
                        batch_size,
                        model_dim / bi),
                    volatile=not self.training))

        # Expects (input, h_0):
        #   input => seq_len x batch_size x model_dim
        #   h_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        output, hn = self.rnn(x, h0)

        if self.reverse:
            output = reverse_tensor(output, dim=1)

        return output, hn


class IntraAttention(nn.Module):
    def __init__(self, inp_size, outp_size, distance_bias=True):
        super(IntraAttention, self).__init__()
        self.outp_size = outp_size
        self.distance_bias = distance_bias
        self.f = Linear()(inp_size, outp_size)

    def d(self, batch_size, seq_len, max_distance=10, scale=0.01):
        """
        Generates a bias term based on distance. Something like:

        [[[0, 1, 2, 3],
          [1, 0, 1, 2],
          [2, 1, 0, 1],
          ...
          ],
         [[0, 1, 2, 3],
          [1, 0, 1, 2],
          [2, 1, 0, 1],
          ...
          ],
          ...
         ]

        """

        bias = torch.range(
            0,
            seq_len -
            1).float().unsqueeze(0).repeat(
            seq_len,
            1)
        diff = torch.range(0, seq_len - 1).float().unsqueeze(1)
        bias = (bias - diff).abs()
        bias = bias.clamp(0, max_distance)

        bias = bias * scale

        bias = bias.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        bias = bias.contiguous()

        bias = to_gpu(Variable(bias, volatile=not self.training))

        return bias

    def forward(self, x):
        hidden_dim = self.outp_size
        batch_size, seq_len, _ = x.size()

        f = self.f(x.view(batch_size * seq_len, -1)
                   ).view(batch_size, seq_len, -1)

        f_broadcast = f.unsqueeze(1).expand(
            batch_size, seq_len, seq_len, hidden_dim)

        # assert f_broadcast[0, 0, 0, :] == f_broadcast[0, 1, 0, :]

        e = torch.bmm(f, f.transpose(1, 2))
        e = e.view(batch_size * seq_len, seq_len)

        if self.distance_bias:
            d = self.d(batch_size, seq_len).view(batch_size * seq_len, seq_len)
            e = e + d

        e = F.softmax(e)
        e = e.view(batch_size, seq_len, seq_len, 1)
        e = e

        # assert e[0, 0, :, 0].sum() == 1

        a = (f_broadcast * e).sum(2).squeeze()

        return a


class EncodeGRU(GRU):
    def __init__(
            self,
            inp_dim,
            model_dim,
            bidirectional=False,
            mix=True,
            *args,
            **kwargs):
        if mix and bidirectional:
            self.mix = True
            assert model_dim % 4 == 0, "Model dim must be divisible by 4 to use bidirectional GRU encoder."
            self.half_state_dim = model_dim / 4
        else:
            self.mix = False
        super(
            EncodeGRU,
            self).__init__(
            inp_dim,
            model_dim,
            *args,
            bidirectional=bidirectional,
            **kwargs)

    def forward(self, x, h0=None):
        output, _ = super(EncodeGRU, self).forward(x, h0)
        if self.mix:
            # Prevent feeding only forward state into h and only backward state
            # into c
            a = output[:, :, 0:self.half_state_dim]
            b = output[:, :, self.half_state_dim:2 * self.half_state_dim]
            c = output[:, :, 2 * self.half_state_dim:3 * self.half_state_dim]
            d = output[:, :, 3 * self.half_state_dim:4 * self.half_state_dim]
            output = torch.cat([a, c, b, d], 2)
        return output.contiguous()


class LSTM(nn.Module):
    def __init__(self, inp_dim, model_dim, num_layers=1,
                 reverse=False, bidirectional=False, dropout=None):
        super(LSTM, self).__init__()
        self.model_dim = model_dim
        self.reverse = reverse
        self.bidirectional = bidirectional
        self.bi = 2 if self.bidirectional else 1
        self.num_layers = num_layers
        self.rnn = nn.LSTM(inp_dim, model_dim / self.bi, num_layers=num_layers,
                           batch_first=True,
                           bidirectional=self.bidirectional,
                           dropout=dropout)

    def forward(self, x, h0=None, c0=None):
        bi = self.bi
        num_layers = self.num_layers
        batch_size, seq_len = x.size()[:2]
        model_dim = self.model_dim

        if self.reverse:
            x = reverse_tensor(x, dim=1)

        # Initialize state unless it is given.
        if h0 is None:
            h0 = to_gpu(
                Variable(
                    torch.zeros(
                        num_layers * bi,
                        batch_size,
                        model_dim / bi),
                    volatile=not self.training))
        if c0 is None:
            c0 = to_gpu(
                Variable(
                    torch.zeros(
                        num_layers * bi,
                        batch_size,
                        model_dim / bi),
                    volatile=not self.training))

        # Expects (input, h_0, c_0):
        #   input => seq_len x batch_size x model_dim
        #   h_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        #   c_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        output, (hn, _) = self.rnn(x, (h0, c0))

        if self.reverse:
            output = reverse_tensor(output, dim=1)

        return output


class ReduceTreeLSTM(nn.Module):
    """TreeLSTM composition module for SPINN.

    The TreeLSTM has two to three inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model.

    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
        use_tracking_in_composition: If specified, use the tracking state as input.
        composition_ln: Whether to use layer normalization.
    """

    def __init__(self, size, tracker_size=None,
                 use_tracking_in_composition=None, composition_ln=True):
        super(ReduceTreeLSTM, self).__init__()
        self.composition_ln = composition_ln
        self.left = Linear()(size, 5 * size)
        self.right = Linear()(
            size, 5 * size, bias=False)
        if composition_ln:
            self.left_ln = LayerNormalization(size)
            self.right_ln = LayerNormalization(size)
        if tracker_size is not None and use_tracking_in_composition:
            self.track = Linear()(
                tracker_size, 5 * size, bias=False)
            if composition_ln:
                self.track_ln = LayerNormalization(tracker_size)

    def forward(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.

        The TreeLSTM has two to three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided
        as iterables and batched internally into tensors.

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
                ``h`` concatenated for the LSTM state of each new node.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)

        if self.composition_ln:
            lstm_in = self.left(self.left_ln(left.h))
            lstm_in += self.right(self.right_ln(right.h))
        else:
            lstm_in = self.left(left.h)
            lstm_in += self.right(right.h)

        if hasattr(self, 'track'):
            if self.composition_ln:
                lstm_in += self.track(self.track_ln(tracking.h))
            else:
                lstm_in += self.track(tracking.h)

        return unbundle(treelstm(left.c, right.c, lstm_in))


class Lift(nn.Module):
    def __init__(self, in_features, out_features):
        super(Lift, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lift = Linear()(self.in_features, self.out_features * 2)

    def forward(self, input):
        return F.tanh(self.lift(input))


class ReduceTensor(nn.Module):
    def __init__(self, size):
        super(ReduceTensor, self).__init__()
        
        assert size is not None

        self.dim = size
        self.weight = Parameter(torch.Tensor(self.dim, self.dim))
        self.b1 = Parameter(torch.Tensor(self.dim, self.dim))
        self.b2 = Parameter(torch.Tensor(self.dim, self.dim))

        self.left = Linear()(self.dim * self.dim, 4 * (self.dim * self.dim))
        self.right = Linear()(self.dim * self.dim, 4 * (self.dim * self.dim))

        self.reset_parameters()
        
    def reset_parameters(self):
        kaiming_normal(self.weight)
        ZeroInitializer(self.b1)
        ZeroInitializer(self.b2)

    def forward(self, left_in, right_in):
        left, right = bundle(left_in), bundle(right_in)
        lstm_gates = self.left(left.h)
        lstm_gates += self.right(right.h)

        # Core composition
        # Retrieve hidden state from left_in and feed it into weight matrix
        cell_inp = []
        hidden_dim = self.dim * self.dim

        h = left.h.contiguous().view(-1, self.dim, self.dim)
        cell_inp = torch.matmul(self.weight, h)
        cell_inp = F.tanh(torch.add(cell_inp, self.b1))

        # Retrieve hidden state from right_in
        h = right.h.contiguous().view(-1, self.dim, self.dim)
        cell_inp = F.tanh(torch.baddbmm(self.b2, cell_inp, h))
        cell_inp = cell_inp.view(-1, hidden_dim)

        out = unbundle(treelstmtensor(left.c, right.c, lstm_gates, cell_inp, training=self.training))

        return out


def treelstmtensor(c_left, c_right, gates, cell_inp, use_dropout=False, training=None):
    hidden_dim = c_left.size()[1]

    assert gates.size()[1] == hidden_dim * 4, "Need to have 4 gates."

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Compute and slice gate values
    i_gate, fl_gate, fr_gate, o_gate = [slice_gate(gates, i) for i in range(4)]

    # Apply nonlinearities
    i_gate = F.sigmoid(i_gate)
    fl_gate = F.sigmoid(fl_gate)
    fr_gate = F.sigmoid(fr_gate)
    o_gate = F.sigmoid(o_gate)

    # Compute new cell and hidden value
    i_val = i_gate * cell_inp
    dropout_rate = 0.1
    if use_dropout:
        i_val = F.dropout(i_val, dropout_rate, training=training)
    c_t = fl_gate * c_left + fr_gate * c_right + i_val
    h_t = o_gate * F.tanh(c_t)

    return (c_t, h_t)


class MLP(nn.Module):
    def __init__(
            self,
            mlp_input_dim,
            mlp_dim,
            num_classes,
            num_mlp_layers,
            mlp_ln,
            classifier_dropout_rate=0.0):
        super(MLP, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.mlp_ln = mlp_ln
        self.classifier_dropout_rate = classifier_dropout_rate

        features_dim = mlp_input_dim

        if mlp_ln:
            self.ln_inp = LayerNormalization(mlp_input_dim)

        for i in range(num_mlp_layers):
            setattr(self, 'l{}'.format(i), Linear()(features_dim, mlp_dim))
            if mlp_ln:
                setattr(self, 'ln{}'.format(i), LayerNormalization(mlp_dim))
            features_dim = mlp_dim
        setattr(
            self,
            'l{}'.format(num_mlp_layers),
            Linear()(
                features_dim,
                num_classes))

    def forward(self, h):
        if self.mlp_ln:
            h = self.ln_inp(h)
        h = F.dropout(h, self.classifier_dropout_rate, training=self.training)
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(h)
            h = F.relu(h)
            if self.mlp_ln:
                ln = getattr(self, 'ln{}'.format(i))
                h = ln(h)
            h = F.dropout(
                h,
                self.classifier_dropout_rate,
                training=self.training)
        layer = getattr(self, 'l{}'.format(self.num_mlp_layers))
        y = layer(h)
        return y


def ZeroInitializer(param):
    shape = param.size()
    init = np.zeros(shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def Linear(initializer=kaiming_normal,
           bias_initializer=ZeroInitializer):
    class CustomLinear(nn.Linear):
        def reset_parameters(self):
            initializer(self.weight)
            if self.bias is not None:
                bias_initializer(self.bias)
    return CustomLinear
