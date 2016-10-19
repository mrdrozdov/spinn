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


class SentencePairModel(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length, initial_embeddings, num_classes,
                 mlp_dim,
                 keep_rate,
                 gpu=-1,
                 ):
        super(SentencePairModel, self).__init__(
            embed=EmbedChain(word_embedding_dim, vocab_size, initial_embeddings, model_dim, gpu=gpu),
            l0=L.Linear(word_embedding_dim*2, 1024),
            l1=L.Linear(1024, 512),
            l2=L.Linear(512, 3),
        )
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.lossFun = softmax_cross_entropy.softmax_cross_entropy
        self.keep_rate = keep_rate
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        self.num_classes = num_classes

    def __call__(self, sentences, transitions, y_batch=None, train=True):
        ratio = 1 - self.keep_rate

        # Get Embeddings
        x_prem = self.embed(Variable(sentences[:,:,0]))
        x_hyp = self.embed(Variable(sentences[:,:,1]))

        # Sum and Concatenate both Sentences
        h_l = F.sum(x_prem, axis=1)
        h_r = F.sum(x_hyp, axis=1)
        h = F.concat([h_l, h_r], axis=1)

        # Pass through Classifier
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = h
        
        accum_loss = self.lossFun(y, y_batch)
        self.accuracy = self.accFun(y_hat, y_batch)

        return y_hat, accum_loss
