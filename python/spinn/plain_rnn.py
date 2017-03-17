from functools import partial
import argparse
import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import Embed, to_gpu
from spinn.util.misc import Args, Vocab


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS):
    if data_manager.SENTENCE_PAIR_DATA:
        model_cls = SentencePairModel
        use_sentence_pair = True
    else:
        model_cls = SentenceModel
        use_sentence_pair = False

    return model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         num_classes=num_classes,
         mlp_dim=FLAGS.mlp_dim,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
         transition_weight=FLAGS.transition_weight,
         encode_style=FLAGS.encode_style,
         encode_reverse=FLAGS.encode_reverse,
         encode_bidirectional=FLAGS.encode_bidirectional,
         encode_num_layers=FLAGS.encode_num_layers,
         use_sentence_pair=use_sentence_pair,
         use_skips=FLAGS.use_skips,
         lateral_tracking=FLAGS.lateral_tracking,
         use_tracking_in_composition=FLAGS.use_tracking_in_composition,
         predict_use_cell=FLAGS.predict_use_cell,
         use_lengths=FLAGS.use_lengths,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
         rl_mu=FLAGS.rl_mu,
         rl_baseline=FLAGS.rl_baseline,
         rl_reward=FLAGS.rl_reward,
         rl_weight=FLAGS.rl_weight,
         rl_whiten=FLAGS.rl_whiten,
         rl_entropy=FLAGS.rl_entropy,
         rl_entropy_beta=FLAGS.rl_entropy_beta,
         predict_leaf=FLAGS.predict_leaf,
         gen_h=FLAGS.gen_h,
        )


class BaseModel(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 mlp_dim=None,
                 embedding_keep_rate=None,
                 classifier_keep_rate=None,
                 use_sentence_pair=False,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.model_dim = model_dim

        args = Args()
        args.size = model_dim
        args.input_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(args.size, vocab.size,
                        vectors=vocab.vectors,
                        )

        self.rnn = nn.LSTM(args.size, model_dim, num_layers=1, batch_first=True)

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim

        self.l0 = nn.Linear(mlp_input_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)

    def run_rnn(self, x):
        batch_size, seq_len, model_dim = x.data.size()

        num_layers = 1
        bidirectional = False
        bi = 2 if bidirectional else 1
        h0 = Variable(to_gpu(torch.zeros(num_layers * bi, batch_size, self.model_dim)), volatile=not self.training)
        c0 = Variable(to_gpu(torch.zeros(num_layers * bi, batch_size, self.model_dim)), volatile=not self.training)

        # Expects (input, h_0):
        #   input => batch_size x seq_len x model_dim
        #   h_0   => (num_layers x num_directions[1,2]) x batch_size x model_dim
        #   c_0   => (num_layers x num_directions[1,2]) x batch_size x model_dim
        output, (hn, cn) = self.rnn(x, (h0, c0))

        return hn

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        emb = self.embed(x)
        emb = torch.cat([b.unsqueeze(0) for b in torch.chunk(emb, batch_size, 0)], 0)

        return emb

    def run_mlp(self, h):
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = h
        return y


class SentencePairModel(BaseModel):

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = self.build_example(sentences, transitions)

        emb = self.run_embed(x)

        hh = torch.squeeze(self.run_rnn(emb))
        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        output = self.run_mlp(h)

        return output


class SentenceModel(BaseModel):

    def build_example(self, sentences, transitions):
        return to_gpu(Variable(torch.from_numpy(sentences), volatile=not self.training))

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        # Build Tokens
        x = self.build_example(sentences, transitions)

        emb = self.run_embed(x)

        h = torch.squeeze(self.run_rnn(emb))
        output = self.run_mlp(h)

        return output
