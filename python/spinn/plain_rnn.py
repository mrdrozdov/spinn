""" Chainer-based RNN implementations.

    Important Notes:
    * setting volatile to OFF during evaluation 
      is a performance boost.
"""

import numpy as np
from spinn import util

# Chainer imports
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
from chainer.functions.connection import embed_id
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer.training import extensions

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

    def _forward(self, x_batch, train=True, keep_hs=False):
        batch_size = x_batch.shape[0]
        c = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        h = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        hs = []
        for _x in range(self.seq_length):
            x = self.__mod.array(x_batch[:, _x].data)
            x = Variable(x, volatile=not train)
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
                 prefix="RNNChain",
                 gpu=-1
                 ):
        super(RNNChain, self).__init__(
            rnn=LSTMChain(word_embedding_dim, model_dim, seq_length)
        )

        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.W = self.__mod.array(initial_embeddings, dtype=self.__mod.float32)
        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim

    def _forward(self, x_batch, train=True):
        x_batch = Variable(x_batch, volatile=not train)
        x = embed_id.embed_id(x_batch, self.W)
        c, h, hs = self.rnn._forward(x, train)
        return h

class CrossEntropyClassifier(Chain):
    def __init__(self, gpu=-1):
        super(CrossEntropyClassifier, self).__init__()
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def _forward(self, y, y_batch, train=True):
        accum_loss = 0 if train else None
        if train:
            y_batch = Variable(y_batch, volatile=not train)
            if self.__gpu >= 0:
                y_batch = cuda.to_gpu(y_batch)
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

    def _forward(self, x_batch, train=True):
        layers = self.layers
        h = x_batch
        for l0 in self.children(): 
            h_dropped = F.dropout(h, ratio=(1-self.keep_rate), train=train)
            h = F.relu(l0(h_dropped))
        y = h
        return y

class SentenceModel(Chain):
    """docstring for SentenceModel"""
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length, initial_embeddings, num_classes,
                 mlp_dim,
                 keep_rate,
                 gpu=-1,
                 ):
        super(SentenceModel, self).__init__(
            x2h=RNNChain(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings),
            h2y=MLP(dimensions=[model_dim, mlp_dim, num_classes],
                    keep_rate=keep_rate, gpu=gpu),
            classifier=CrossEntropyClassifier(gpu),
        )
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.keep_rate = keep_rate
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()

    def _forward(self, x_batch, y_batch=None, train=True):
        h = self.x2h._forward(x_batch, train=train)
        y = self.h2y._forward(h, train)
        accum_loss = self.classifier._forward(y, y_batch, train)
        self.accuracy = self.accFun(y, y_batch)
        return y, accum_loss
     

class SentencePairModel(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length, initial_embeddings, num_classes,
                 mlp_dim,
                 keep_rate,
                 gpu=-1,
                 ):
        super(SentencePairModel, self).__init__(
            x2h_premise=RNNChain(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings),
            x2h_hypothesis=RNNChain(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings),
            h2y=MLP(dimensions=[model_dim*2, mlp_dim, num_classes],
                    keep_rate=keep_rate, gpu=gpu),
            classifier=CrossEntropyClassifier(gpu),
        )
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.keep_rate = keep_rate
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()

    def _forward(self, x_batch, y_batch=None, train=True):
        h_premise = self.x2h_premise._forward(x_batch[:, :, 0:1], train=train)
        h_hypothesis = self.x2h_hypothesis._forward(x_batch[:, :, 1:2], train=train)
        h = F.concat([h_premise, h_hypothesis], axis=1)
        y = self.h2y._forward(h, train)
        accum_loss = self.classifier._forward(y, y_batch, train)
        self.accuracy = self.accFun(y, y_batch)
        return y, accum_loss

class RNN(object):
    """Plain RNN encoder implementation. Can use any activation function.
    """

    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length,
                 num_classes,
                 mlp_dim,
                 keep_rate,
                 initial_embeddings=None,
                 use_sentence_pair=False,
                 gpu=-1,
                 **kwargs):
        """Construct an RNN.

        Args:
            model_dim: Dimensionality of hidden state.
            vocab_size: Number of unique tokens in vocabulary.
            compose_network: Blocks-like function which accepts arguments
              `prev_hidden_state, inp, inp_dim, hidden_dim, vs, name` (see e.g. `util.LSTMLayer`).
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
        """

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

        if use_sentence_pair:
            self.model = SentencePairModel(
                model_dim, word_embedding_dim, vocab_size, compose_network,
                     seq_length, initial_embeddings, num_classes, mlp_dim,
                     keep_rate,
                     gpu=-1,
                    )
        else:
            self.model = SentenceModel(
                model_dim, word_embedding_dim, vocab_size, compose_network,
                     seq_length, initial_embeddings, num_classes, mlp_dim,
                     keep_rate,
                     gpu=-1,
                    )

        self.init_params()

    def init_params(self):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, clip, decay, lr=0.001, alpha=0.9, eps=1e-6):
        self.optimizer = optimizers.RMSprop(lr=lr, alpha=alpha, eps=eps)
        self.optimizer.setup(self.model)

        # Clip Gradient
        # QUESTION: Should this be applied prior to calculating the loss?
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(clip))

        # L2 Regularization
        # self.optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    def update(self):
        self.optimizer.update()

    def forward(self, x_batch, y_batch=None, train=True, predict=False):
        y, loss = self.model._forward(x_batch, y_batch, train=train)
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
