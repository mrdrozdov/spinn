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
from chainer.functions.normalization.batch_normalization import batch_normalization
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer.training import extensions

from spinn.util.chainer_blocks import EmbedChain, LSTM, LSTMChain, RNNChain
from spinn.util.chainer_blocks import MLP
from spinn.util.chainer_blocks import CrossEntropyClassifier


class SentencePairModel(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length, initial_embeddings, num_classes,
                 mlp_dim,
                 keep_rate,
                 gpu=-1,
                 ):
        super(SentencePairModel, self).__init__(
            embed=EmbedChain(word_embedding_dim, vocab_size, initial_embeddings, gpu=gpu),
            x2h_premise=RNNChain(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings, gpu=gpu, keep_rate=keep_rate),
            x2h_hypothesis=RNNChain(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings, gpu=gpu, keep_rate=keep_rate),
            h2y=MLP(dimensions=[model_dim*2, mlp_dim, mlp_dim/2, num_classes],
                    keep_rate=keep_rate, gpu=gpu),
            classifier=CrossEntropyClassifier(gpu),
        )
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.keep_rate = keep_rate

    def __call__(self, x_batch, y_batch=None, train=True):
        ratio = 1 - self.keep_rate

        # x_prem = Variable(self.__mod.array(self.embed(x_batch[:, :, 0:1]), dtype=self.__mod.float32))
        # x_hyp = Variable(self.__mod.array(self.embed(x_batch[:, :, 1:2]), dtype=self.__mod.float32))

        x_prem = self.embed(x_batch[:, :, 0:1])
        x_hyp = self.embed(x_batch[:, :, 1:2])

        x_prem = Variable(self.__mod.array(x_prem, dtype=self.__mod.float32), volatile=not train)
        x_hyp = Variable(self.__mod.array(x_hyp, dtype=self.__mod.float32), volatile=not train)

        h_premise = self.x2h_premise(x_prem, train=train)
        h_hypothesis = self.x2h_hypothesis(x_hyp, train=train)
        h = F.concat([h_premise, h_hypothesis], axis=1)

        h = F.dropout(h, ratio, train)
        y = self.h2y(h, train)

        y = F.dropout(y, ratio, train)
        accum_loss = self.classifier(y, y_batch, train)
        self.accuracy = self.accFun(y, self.__mod.array(y_batch))

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
                     gpu,
                    )
        else:
            raise Exception("Not implemented error.")

        self.init_params()
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu()

    def init_params(self):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, clip, decay, lr=0.001, alpha=0.9, eps=1e-6):
        self.optimizer = optimizers.RMSprop(lr=lr, alpha=alpha, eps=eps)
        self.optimizer.setup(self.model)

        # Clip Gradient
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(clip))

        # L2 Regularization
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    def update(self):
        self.optimizer.update()

    def forward(self, x_batch, y_batch=None, train=True, predict=False):
        y, loss = self.model(x_batch, y_batch, train=train)
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
