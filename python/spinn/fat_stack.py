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


import spinn.util.chainer_blocks as CB

from spinn.util.chainer_blocks import EmbedChain, LSTM, LSTMChain, RNNChain
from spinn.util.chainer_blocks import MLP
from spinn.util.chainer_blocks import CrossEntropyClassifier

class SLSTMChain(Chain):
    def __init__(self, input_dim, hidden_dim, seq_length, prefix="LSTMChain", gpu=-1):
        super(SLSTMChain, self).__init__(
            i_l_fwd=L.Linear(input_dim, 4 * hidden_dim, nobias=True),
            h_l_fwd=L.Linear(hidden_dim, 4 * hidden_dim),
            i_r_fwd=L.Linear(input_dim, 4 * hidden_dim, nobias=True),
            h_r_fwd=L.Linear(hidden_dim, 4 * hidden_dim),
        )
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

        self.c_l, self.c_r = None, None
        self.h_l, self.h_r = None, None

    def __call__(self, left_x, right_x, train=True, keep_hs=False):

        batch_size = len(left_x)

        if self.c_l is None:
            self.c_l = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        if self.c_r is None:
            self.c_r = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        if self.h_l is None:
            self.h_l = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        if self.h_l is None:
            self.h_l = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)

        left_x = F.split_axis(np.array(left_x), self.seq_length, axis=1)
        right_x = F.split_axis(np.array(right_x), self.seq_length, axis=1)
        for x_l, x_r in zip(left_x, right_x):
            il = self.i_l_fwd(x_l)
            ir = self.i_r_fwd(x_r)
            hl = self.h_l_fwd(h_l)
            hr = self.h_r_fwd(h_r)
            ihl = il + hl
            ihr = ir + hr
            self.c_l, self.c_r, self.h_l, self.h_r = F.slstm(hl, hr, il, ir)

        return c, h, hs

    def reset_state(self):
        self.c_l, self.c_r = None, None
        self.h_l, self.h_r = None, None

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

    def __call__(self, sentences, transitions, train=True, keep_hs=False):

        batch_size = sentences.data.shape[0]
        c_prev_l, c_prev_r, b_prev_l, b_prev_r = [self.__mod.zeros(
            (batch_size, self.hidden_dim), dtype=self.__mod.float32)
            for _ in range(4)]

        buffers = F.split_axis(sentences, batch_size, axis=0)
        buffers = [x.data[0].tolist() for x in buffers]

        sentences = F.split_axis(sentences, self.seq_length, axis=1)
        transitions = F.split_axis(transitions, self.seq_length, axis=1)
        transitions = [x.data[0][0] for x in transitions]

        # MAYBE: Initialize stack with None, None in case of initial reduces.
        stacks = [[]] * len(buffers)


        for ts in transitions:
            lefts = []
            rights = []
            for i, (t, buf, stack) in enumerate(zip(ts, buffers, stacks)):
                # Setup Actions
                if t == 0: # shift
                    stack.append(buf.pop())
                elif t == 1: # reduce
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                else:
                    raise Exception("This action is not implemented: %s" % t)

            # Complete Actions
            if len(rights) > 0:
                reduced = iter(self.reduce(lefts, rights))
                for i, t, buf, stack in enumerate(zip(ts, bufs, stacks)):
                    if t == 1:
                        composition = next(reduced)
                        import ipdb; ipdb.set_trace()
                        stack.append(composition)

        import ipdb; ipdb.set_trace()

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
            batch_norm=L.BatchNormalization(model_dim, model_dim)
        )

        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.keep_rate = keep_rate
        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim

    def __call__(self, x, t, train=True):
        ratio = 1 - self.keep_rate

        # gamma = Variable(self.__mod.array(1.0, dtype=self.__mod.float32), volatile=not train, name='gamma')
        # beta = Variable(self.__mod.array(0.0, dtype=self.__mod.float32),volatile=not train, name='beta')
        # x = batch_normalization(x, gamma, beta, eps=2e-5, running_mean=None,running_var=None, decay=0.9, use_cudnn=False)
        # x = self.batch_norm(x)
        # x = F.dropout(x, ratio, train)

        c, h, hs = self.spinn(x, t, train)
        return h

class SentencePairModel(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length, initial_embeddings, num_classes,
                 mlp_dim,
                 keep_rate,
                 gpu=-1,
                 ):
        super(SentencePairModel, self).__init__(
            embed=EmbedChain(word_embedding_dim, vocab_size, initial_embeddings, gpu=gpu),
            x2h_premise=SPINN(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings, gpu=gpu, keep_rate=keep_rate),
            x2h_hypothesis=SPINN(model_dim, word_embedding_dim, vocab_size,
                    seq_length, initial_embeddings, gpu=gpu, keep_rate=keep_rate),
            h2y=MLP(dimensions=[model_dim*2, mlp_dim, mlp_dim/2, num_classes],
                    keep_rate=keep_rate, gpu=gpu),
            classifier=CrossEntropyClassifier(gpu),
        )
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.keep_rate = keep_rate

    def __call__(self, sentences, transitions, y_batch=None, train=True):
        ratio = 1 - self.keep_rate

        x_prem = self.embed(sentences[:, :, 0:1])
        x_hyp = self.embed(sentences[:, :, 1:2])

        t_prem = transitions[:, :, 0:1]
        t_hyp = transitions[:, :, 1:2]

        x_prem = Variable(self.__mod.array(x_prem, dtype=self.__mod.float32), volatile=not train)
        x_hyp = Variable(self.__mod.array(x_hyp, dtype=self.__mod.float32), volatile=not train)

        h_premise = self.x2h_premise(x_prem, t_prem, train=train)
        h_hypothesis = self.x2h_hypothesis(x_hyp, t_hyp, train=train)
        h = F.concat([h_premise, h_hypothesis], axis=1)

        h = F.dropout(h, ratio, train)
        y = self.h2y(h, train)

        y = F.dropout(y, ratio, train)
        accum_loss = self.classifier(y, y_batch, train)
        self.accuracy = self.accFun(y, self.__mod.array(y_batch))

        return y, accum_loss

class TransitionModel(object):

    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
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

        self.model = SentencePairModel(
                model_dim, word_embedding_dim, vocab_size, compose_network,
                     seq_length, initial_embeddings, num_classes, mlp_dim,
                     keep_rate,
                     gpu,
                    )

        # self.init_params()
        # if gpu >= 0:
        #     cuda.get_device(gpu).use()
        #     self.model.to_gpu()

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
