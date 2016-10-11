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

LSTM = F.lstm

class EmbedChain(Chain):
    def __init__(self, embedding_dim, vocab_size, initial_embeddings, prefix="EmbedChain", gpu=-1):
        super(EmbedChain, self).__init__()
        assert initial_embeddings is not None, "Depends on pre-trained embeddings."
        self.raw_embeddings = initial_embeddings
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def __call__(self, x_batch, train=True, keep_hs=False):
        # x_batch: batch_size * seq_len
        b, l = x_batch.shape[0], x_batch.shape[1]

        # Perform indexing and reshaping on the CPU.
        x = self.raw_embeddings.take(x_batch.ravel(), axis=0)
        x = x.reshape(b, l, -1)

        return x

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

    def __call__(self, y, y_batch, train=True):
        accum_loss = 0 if train else None
        if train:
            if self.__gpu >= 0:
                y_batch = cuda.to_gpu(y_batch)
            y_batch = Variable(y_batch, volatile=not train)
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
