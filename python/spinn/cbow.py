"""Theano-based sum-of-words implementations."""

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

from chainer.utils import type_check

import spinn.util.chainer_blocks as CB

from chainer.functions.loss import softmax_cross_entropy
from spinn.util.chainer_blocks import LSTMChain, RNNChain, EmbedChain
from spinn.util.chainer_blocks import MLP
from spinn.util.chainer_blocks import CrossEntropyClassifier
from spinn.util.chainer_blocks import BaseSentencePairTrainer


class SentencePairTrainer(BaseSentencePairTrainer):
    def init_params(self, **kwargs):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            data[:] = np.random.uniform(-0.005, 0.005, data.shape)

    def init_optimizer(self, lr=0.01, **kwargs):
        # self.optimizer = optimizers.Adam(alpha=0.0003, beta1=0.9, beta2=0.999, eps=1e-08)
        self.optimizer = optimizers.SGD(lr=0.1)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5))
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.00003))


class SentencePairModel(Chain):
    def __init__(self, model_dim, word_embedding_dim,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
                 keep_rate,
                 gpu=-1,
                 ):
        super(SentencePairModel, self).__init__(
            batch_norm_0=L.BatchNormalization(model_dim*2, model_dim*2),
            batch_norm_1=L.BatchNormalization(mlp_dim, mlp_dim),
            batch_norm_2=L.BatchNormalization(mlp_dim, mlp_dim),
            l0=L.Linear(model_dim*2, mlp_dim),
            l1=L.Linear(mlp_dim, mlp_dim),
            l2=L.Linear(mlp_dim, num_classes)
        )
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.lossFun = softmax_cross_entropy.softmax_cross_entropy
        self.keep_rate = keep_rate
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.initial_embeddings = initial_embeddings

    def __call__(self, sentences, transitions, y_batch=None, train=True):
        ratio = 1 - self.keep_rate

        # Get Embeddings
        sentences = self.initial_embeddings.take(sentences, axis=0
            ).astype(np.float32)

        # Get Embeddings
        x_prem = Variable(sentences[:,:,0])
        x_hyp = Variable(sentences[:,:,1])

        # Sum and Concatenate both Sentences
        h_l = F.sum(x_prem, axis=1)
        h_r = F.sum(x_hyp, axis=1)
        h = F.concat([h_l, h_r], axis=1)

        # Pass through Classifier
        h = self.batch_norm_0(h, test=not train)
        h = F.dropout(h, ratio, train)
        h = F.relu(h)
        h = self.l0(h)
        h = self.batch_norm_1(h, test=not train)
        h = F.dropout(h, ratio, train)
        h = F.relu(h)
        h = self.l1(h)
        h = self.batch_norm_2(h, test=not train)
        h = F.dropout(h, ratio, train)
        h = F.relu(h)
        h = self.l2(h)
        y = h
        
        accum_loss = self.lossFun(y, y_batch)
        self.accuracy = self.accFun(y, y_batch)

        return y, accum_loss
