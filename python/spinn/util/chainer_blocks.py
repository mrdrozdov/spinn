import numpy as np

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

from chainer.utils import type_check



class EmbedChain(Chain):
    def __init__(self, embedding_dim, vocab_size, initial_embeddings, projection_dim, prefix="EmbedChain", gpu=-1):
        super(EmbedChain, self).__init__()
        assert initial_embeddings is not None, "Depends on pre-trained embeddings."
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.raw_embeddings = self.__mod.array(initial_embeddings, dtype=self.__mod.float32)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype == 'i',
            x_type.ndim >= 1,
        )

    def __call__(self, x_batch, train=True):
        """
        Compute an integer lookup on an embedding matrix.

        Args:
            x_batch: List of B x S
        Returns:
            x_emb:   List of flatten embeddings B x S x E
        """

        # BEGIN: Type Check
        in_data = tuple([x.data for x in [x_batch]])
        in_types = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_types)
        # END: Type Check

        emb = self.raw_embeddings.take(x_batch.data, axis=0)
        return emb


class TreeLSTMChain(Chain):
    def __init__(self, hidden_dim, prefix="TreeLSTMChain", gpu=-1):
        super(TreeLSTMChain, self).__init__(
            W_l=L.Linear(hidden_dim, hidden_dim*5),
            W_r=L.Linear(hidden_dim, hidden_dim*5),
            )
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def __call__(self, c_l, h_l, c_r, h_r, train=True, keep_hs=False):
        # TODO: Figure out bias. In this case, both left and right
        # weights have intrinsic bias, but this was not the strategy
        # in the previous code base. I think the trick is to use 
        # add_param, and then F.broadcast when doing the addition.
        gates = self.W_l(h_l) + self.W_r(h_r)

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


class LSTMChain(Chain):
    def __init__(self, input_dim, hidden_dim, seq_length, prefix="LSTMChain", gpu=-1):
        super(LSTMChain, self).__init__(
            i_fwd=L.Linear(input_dim, 4 * hidden_dim, nobias=True),
            h_fwd=L.Linear(hidden_dim, 4 * hidden_dim),
        )
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def __call__(self, x_batch, train=True, keep_hs=False):
        batch_size = x_batch.data.shape[0]
        c = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        h = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        hs = []
        batches = F.split_axis(x_batch, self.seq_length, axis=1)
        for x in batches:
            ii = self.i_fwd(x)
            hh = self.h_fwd(h)
            ih = ii + hh
            c, h = F.lstm(c, ih)

            if keep_hs:
                # Convert from (#batch_size, #hidden_dim) ->
                #              (#batch_size, 1, #hidden_dim)
                # This is important for concatenation later.
                h_reshaped = F.reshape(h, (batch_size, 1, self.hidden_dim))
                hs.append(h_reshaped)

        if keep_hs:
            # This converts list of: [(#batch_size, 1, #hidden_dim)]
            # To single tensor:       (#batch_size, #seq_length, #hidden_dim)
            # Which matches the input shape.
            hs = F.concat(hs, axis=1)
        else:
            hs = None

        return c, h, hs

class RNNChain(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 seq_length,
                 initial_embeddings,
                 keep_rate,
                 prefix="RNNChain",
                 gpu=-1
                 ):
        super(RNNChain, self).__init__(
            rnn=LSTMChain(word_embedding_dim, model_dim, seq_length, gpu=gpu),
            batch_norm=L.BatchNormalization(model_dim, model_dim)
        )

        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.keep_rate = keep_rate
        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim

    def __call__(self, x, train=True):
        ratio = 1 - self.keep_rate

        # gamma = Variable(self.__mod.array(1.0, dtype=self.__mod.float32), volatile=not train, name='gamma')
        # beta = Variable(self.__mod.array(0.0, dtype=self.__mod.float32),volatile=not train, name='beta')
        # x = batch_normalization(x, gamma, beta, eps=2e-5, running_mean=None,running_var=None, decay=0.9, use_cudnn=False)
        x = self.batch_norm(x)
        x = F.dropout(x, ratio, train)

        c, h, hs = self.rnn(x, train)
        return h

class CrossEntropyClassifier(Chain):
    def __init__(self, gpu=-1):
        super(CrossEntropyClassifier, self).__init__()
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        pred_type, y_type = in_types

        type_check.expect(
            pred_type.dtype == 'f',
            pred_type.ndim >= 1,
        )

        type_check.expect(
            y_type.dtype == 'i',
            y_type.ndim >= 1,
        )

    def __call__(self, y, y_batch, train=True):
        # BEGIN: Type Check
        in_data = tuple([x.data for x in [y, y_batch]])
        in_types = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_types)
        # END: Type Check

        accum_loss = 0 if train else None
        if train:
            if self.__gpu >= 0:
                y_batch = cuda.to_gpu(y_batch.data)
            accum_loss = F.softmax_cross_entropy(y, y_batch)

        return accum_loss

class MLP(ChainList):
    def __init__(self, dimensions,
                 prefix="MLP",
                 keep_rate=0.5,
                 gpu=-1,
                 ):
        super(MLP, self).__init__()
        self.keep_rate = keep_rate
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.layers = []

        assert len(dimensions) >= 2, "Must initialize MLP with 2 or more layers."
        for l0_dim, l1_dim in zip(dimensions[:-1], dimensions[1:]):
            self.add_link(L.Linear(l0_dim, l1_dim))

    def __call__(self, x_batch, train=True):
        ratio = 1 - self.keep_rate
        layers = self.layers
        h = x_batch

        for l0 in self.children():
            h = F.dropout(h, ratio, train)
            h = F.relu(l0(h))
        y = h
        return y

class SentencePairTrainer(object):

    def __init__(self, model, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length,
                 num_classes,
                 mlp_dim,
                 keep_rate,
                 initial_embeddings=None,
                 use_sentence_pair=False,
                 gpu=-1,
                 **kwargs):

        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        self._compose_network = compose_network
        self.initial_embeddings = initial_embeddings
        self.seq_length = seq_length
        self.keep_rate = keep_rate
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

        self.model = model

        self.init_params()
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu()

    def init_params(self):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, clip, decay, lr=0.01, alpha=0.9, eps=1e-6):
        self.optimizer = optimizers.SGD(lr=0.01)
        self.optimizer.setup(self.model)

        # Clip Gradient
        # self.optimizer.add_hook(chainer.optimizer.GradientClipping(clip))

        # L2 Regularization
        # self.optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    def update(self):
        self.optimizer.update()

    def forward(self, x_batch, y_batch=None, train=True, predict=False):
        assert "sentences" in x_batch and "transitions" in x_batch, \
            "Input must contain dictionary of sentences and transitions."

        sentences = x_batch["sentences"]
        transitions = x_batch["transitions"]

        y, loss = self.model(sentences, transitions, y_batch, train=train)
        if predict:
            preds = self.__mod.argmax(y.data, 1).tolist()
        else:
            preds = None
        return y, loss, preds

    def save(self, filename):
        chainer.serializers.save_npz(filename, self.model)

    @staticmethod
    def load(filename, n_units, gpu):
        self = SentenceModel(n_units, gpu)
        chainer.serializers.load_npz(filename, self.model)
        return self
