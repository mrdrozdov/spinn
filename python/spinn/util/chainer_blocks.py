import numpy as np
import random

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
from chainer import testing

from chainer.utils import type_check


def gradient_check(model, get_loss, rtol=0, atol=1e-2, to_check=10):
    epsilon = 1e-3
    cached_grads = [w.grad.ravel().copy() for (n,w) in model.namedparams()]
    checked = []

    # TODO: Only consider non-zero gradients.
    for (_, w), cached in zip(model.namedparams(), cached_grads):
        chosen = range(len(cached))
        random.shuffle(chosen)
        chosen = chosen[:to_check]
        for c in chosen:
            # Find Y1
            w.data.ravel()[c] += epsilon
            check_loss_1 = get_loss()

            # Find Y2
            w.data.ravel()[c] -= 2 * epsilon
            check_loss_2 = get_loss()

            # Reset Param
            w.data.ravel()[c] += epsilon

            # Check that: Gradient ~ (Y1 - Y2) / (2 * epsilon)
            estimate = (check_loss_1.data - check_loss_2.data) / (2 * epsilon)
            checked.append((estimate, cached[c]))

    estimates, grads = zip(*checked)
    estimates, grads = np.array(estimates), np.array(grads)
    testing.assert_allclose(estimates, grads, rtol=rtol, atol=atol, verbose=True), "Gradient check failed."


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


class BaseSentencePairTrainer(object):

    def __init__(self, model, model_dim, word_embedding_dim,
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

    def init_params(self, **kwargs):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, lr=0.01, **kwargs):
        self.optimizer = optimizers.SGD(lr=lr)
        self.optimizer.setup(self.model)

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
