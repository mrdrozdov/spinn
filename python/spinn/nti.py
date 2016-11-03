import math
import sys
import time
import copy
import numpy as np
import six
from math import log
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import chainer.links as L
import chainer

from chainer.functions.evaluation import accuracy
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
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.00003))

class SentencePairModel(chainer.Chain):
    def __init__(self, model_dim, word_embedding_dim,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
                 keep_rate,
                 gpu=-1,
                 ):
        super(SentencePairModel, self).__init__(
            nti=NTIFullTreeMatching(model_dim, gpu)
        )
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.keep_rate = keep_rate
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.initial_embeddings = initial_embeddings

    def __call__(self, sentences, transitions, y_batch=None, train=True):
        sentences = self.initial_embeddings.take(sentences, axis=0
            ).astype(np.float32)

        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        preds, accum_loss, y = self.nti(train, x, y_batch=y_batch)
        self.accuracy = self.accFun(y, self.__mod.array(y_batch))

        return y, accum_loss

class NTIFullTreeMatching(chainer.Chain):

    """docstring for NTIFullTreeMatching"""
    def __init__(self, n_units, gpu):
        super(NTIFullTreeMatching, self).__init__(
            h_lstm = L.LSTM(n_units, n_units),
            m_lstm = L.LSTM(n_units, n_units),
            h_x = F.Linear(n_units, 4*n_units),
            h_h = F.Linear(n_units, 4*n_units),
            w_ap = F.Linear(n_units, n_units),
            w_we = F.Linear(n_units, 1),
            w_c = F.Linear(n_units, n_units),
            w_q = F.Linear(n_units, n_units),
            h_l1 = F.Linear(2*n_units, 1024),
            l_y = F.Linear(1024, 3))
        self.__n_units = n_units
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        for param in self.params():
            data = param.data
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()

    def init_optimizer(self):
        self.__opt = optimizers.Adam(alpha=0.0003, beta1=0.9, beta2=0.999, eps=1e-08)
        self.__opt.setup(self)
        self.__opt.add_hook(chainer.optimizer.GradientClipping(40))
        self.__opt.add_hook(chainer.optimizer.WeightDecay(0.00003))

    def save(self, filename):
        chainer.serializers.save_npz(filename, self)

    @staticmethod
    def load(filename, n_units, gpu):
        self = NTIFullTreeMatching(n_units, gpu)
        chainer.serializers.load_npz(filename, self)
        return self

    def reset_state(self):
        self.h_lstm.reset_state()
        self.m_lstm.reset_state()

    def __attend_f_tree(self, hs, hsq, q, batch_size, train):
        n_units = self.__n_units
        mod = self.__mod
        
        # calculate attention weights
        x_len = len(hs)
        depth = int(log(x_len, 2)) + 1

        w_a = F.reshape(F.batch_matmul(F.dropout(hsq, ratio=0.0, train=train), self.w_ap(q)), (batch_size, -1))
        w_a = F.exp(w_a)
        list_e = F.split_axis(w_a, x_len, axis=1)

        for d in reversed(range(1, depth)):
            for s in range(2**d-1, 2**(d+1)-1, 2):
                l = hs[s]
                r = hs[s+1]
                lr = hs[(s-1)/2]
                le = list_e[s]
                re = list_e[s+1]
                lre = list_e[(s-1)/2]
                sum_e = le + re + lre
                lr = F.batch_matmul(lr, lre/sum_e)
                lr += F.batch_matmul(l, le/sum_e)
                lr += F.batch_matmul(r, re/sum_e)
                hs[(s-1)/2] = F.reshape(lr, (batch_size, -1))

        
        s_c = hs[0]
        s_c = F.relu(self.w_c(s_c) + self.w_q(q))

        return s_c

    def __attend_fast(self, hs, q, batch_size, train):
        n_units = self.__n_units
        mod = self.__mod

        w_a = F.reshape(F.batch_matmul(F.dropout(hs, ratio=0.0, train=train), self.w_ap(q)), (batch_size, -1))
        w_a = F.softmax(w_a)
        s_c = F.reshape(F.batch_matmul(w_a, hs, transa=True), (batch_size, -1))

        h = F.relu(self.w_c(s_c) + self.w_q(q))
        return h


    def __call__(self, train, x_batch, y_batch = None):
        model = self
        n_units = self.__n_units
        mod = self.__mod
        gpu = self.__gpu
        batch_size = len(x_batch)
        x_len = len(x_batch[0])
        depth = int(log(x_len, 2)) + 1

        self.reset_state()

        list_a = [[] for i in range(2**depth-1)]
        list_c = [[] for i in range(2**depth-1)]
        zeros = mod.zeros((batch_size, n_units), dtype=np.float32)
        for l in xrange(x_len):
            x_data = mod.array([x_batch[k][l] for k in range(batch_size)])
            x_data = Variable(x_data, volatile=not train)
            x_data = model.h_lstm(F.dropout(x_data, ratio=0.2, train=train))
            list_a[x_len-1+l] = x_data
            list_c[x_len-1+l] = model.h_lstm.c #Variable(zeros, volatile=not train)
        
        for d in reversed(range(1, depth)):
            for s in range(2**d-1, 2**(d+1)-1, 2):
                l = model.h_x(F.dropout(list_a[s], ratio=0.2, train=train))
                r = model.h_h(F.dropout(list_a[s+1], ratio=0.2, train=train))
                c_l = list_c[s]
                c_r = list_c[s+1]
                c, h = F.slstm(c_l, c_r, l, r)
                list_a[(s-1)/2] = h
                list_c[(s-1)/2] = c

        list_p = []
        list_h = []
        for a in list_a:
            n_hs = F.split_axis(a, 2, axis=0)
            list_p.append(n_hs[0])
            list_h.append(n_hs[1])
        
        list_pq = F.concat([F.reshape(h, (batch_size/2, 1, n_units)) for h in list_p], axis=1)
        list_aoa = []
        for d in reversed(range(1, depth)):
            for s in range(2**d-1, 2**(d+1)-1, 1):
                a = self.__attend_fast(list_pq, list_h[s], batch_size/2, train)
                # a = self.__attend_f_tree(list_p[:], list_pq, list_h[s], batch_size/2, train)
                hs = model.m_lstm(F.dropout(a, ratio=0.2, train=train))
                list_aoa.append(hs)
        list_aoa = F.concat([F.reshape(h, (batch_size/2, 1, n_units)) for h in list_aoa[:-1]], axis=1)
        hs = self.__attend_fast(list_aoa, hs, batch_size/2, train)

        model.m_lstm.reset_state()
        
        list_pq = F.concat([F.reshape(h, (batch_size/2, 1, n_units)) for h in list_h], axis=1)
        list_aoa = []
        for d in reversed(range(1, depth)):
            for s in range(2**d-1, 2**(d+1)-1, 1):
                a = self.__attend_fast(list_pq, list_p[s], batch_size/2, train)
                # a = self.__attend_f_tree(list_h[:], list_pq, list_p[s], batch_size/2, train)
                hs1 = model.m_lstm(F.dropout(a, ratio=0.2, train=train))
                list_aoa.append(hs1)
        list_aoa = F.concat([F.reshape(h, (batch_size/2, 1, n_units)) for h in list_aoa[:-1]], axis=1)
        hs1 = self.__attend_fast(list_aoa, hs1, batch_size/2, train)
        
        hs = F.relu(model.h_l1(F.concat([hs, hs1], axis=1)))
        y = model.l_y(F.dropout(hs, ratio=0.2, train=train))
        preds = mod.argmax(y.data, 1).tolist()

        accum_loss = 0 if train else None
        if train:
            if gpu >= 0:
                y_batch = cuda.to_gpu(y_batch)
            lbl = Variable(y_batch, volatile=not train)
            accum_loss = F.softmax_cross_entropy(y, lbl)
        
        return preds, accum_loss, y

    def train(self, x_batch, y_batch):
        self.__opt.zero_grads()
        preds, accum_loss = self.__forward(True, x_batch, y_batch=y_batch)
        accum_loss.backward()
        self.__opt.update()
        return preds, accum_loss

    def predict(self, x_batch):
        return self.__forward(False, x_batch)[0]