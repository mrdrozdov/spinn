import itertools
import copy

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import BaseSentencePairTrainer, Reduce
from spinn.util.blocks import LSTMState, Embed, MLP
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, treelstm, lstm
from spinn.util.blocks import get_h, get_c, get_seq_h
from spinn.util.misc import Args, Vocab, Example

from spinn.fat_stack import BaseModel, SentenceModel, SentencePairModel
from spinn.fat_stack import SPINN


T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1


class SentencePairTrainer(BaseSentencePairTrainer): pass


class SentenceTrainer(SentencePairTrainer): pass


class GenSPINN(SPINN):

    def __init__(self, args, vocab, use_skips=False):
        super(GenSPINN, self).__init__(args, vocab, use_skips)

        vocab_size = vocab.vectors.shape[0]
        self.inp_dim = args.size

        # TODO: This can be a hyperparam. Use input dim for now.
        self.decoder_dim = self.inp_dim

        # TODO: Include additional features for decoder, such as
        # top of the stack or tracker state.
        features_dim = self.decoder_dim

        self.decoder_rnn = nn.LSTM(self.inp_dim, self.decoder_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            )

        self.decoder = nn.Linear(self.decoder_dim, vocab_size)

    def reset_decoder(self, example):
        """Run decoder on input to initialize rnn states."""
        batch_size = len(example.bufs)

        # TODO: Would prefer to run decoder forwards or backwards?
        batch = torch.cat([torch.cat(b, 0).unsqueeze(0) for b in example.bufs], 0)

        init = to_gpu(Variable(torch.zeros(1, batch_size, self.decoder_dim), volatile=not self.training))
        self.dec_h = list(torch.chunk(init, batch_size, 1))
        self.dec_c = list(torch.chunk(init, batch_size, 1))

        self.run_decoder_rnn(range(batch_size), batch)

    def run_decoder_rnn(self, idxs, x):
        x = get_seq_h(x, self.inp_dim)
        batch_size, seq_len, inp_dim = x.size()

        h_prev = torch.cat([self.dec_h[batch_idx] for batch_idx in idxs], 1)
        c_prev = torch.cat([self.dec_c[batch_idx] for batch_idx in idxs], 1)

        # Expects (input, h_0, c_0):
        #   input => batch_size x seq_len x inp_dim
        #   h_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        #   c_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        output, (hn, cn) = self.decoder_rnn(x, (h_prev, c_prev))

        h_parts = torch.chunk(hn, batch_size, 1)
        c_parts = torch.chunk(cn, batch_size, 1)
        for i, batch_idx in enumerate(idxs):
            self.dec_h[batch_idx] = h_parts[i]
            self.dec_c[batch_idx] = c_parts[i]

        return hn, cn

    def loss_phase_hook(self):
        dummy_inp = Variable(torch.ones(1,1))
        dummy_loss = nn.Linear(1,2)(dummy_inp).sum()
        self.gen_loss = dummy_loss

    def forward(self, example, use_internal_parser=False, validate_transitions=True):
        # TODO: Only run in train mode for now.
        if self.training:
            tokens = example.tokens.data.numpy().tolist()
            tokens = [list(reversed(t)) for t in tokens]
            self.tokens = tokens

            self.reset_decoder(example)

        return super(GenSPINN, self).forward(
            example, use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)

class GenBaseModel(BaseModel):

    def build_spinn(self, args, vocab, use_skips):
        return GenSPINN(args, vocab, use_skips=use_skips)

    def output_hook(self, output, sentences, transitions, y_batch=None):
        pass


class SentencePairModel(GenBaseModel):

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def run_spinn(self, example, use_internal_parser=False, validate_transitions=True):
        state_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            example, use_internal_parser, validate_transitions)
        batch_size = len(state_both) / 2
        h_premise = get_h(torch.cat(state_both[:batch_size], 0), self.hidden_dim)
        h_hypothesis = get_h(torch.cat(state_both[batch_size:], 0), self.hidden_dim)
        return [h_premise, h_hypothesis], transition_acc, transition_loss


class SentenceModel(GenBaseModel):

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def run_spinn(self, example, use_internal_parser=False, validate_transitions=True):
        state, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            example, use_internal_parser, validate_transitions)
        h = get_h(torch.cat(state, 0), self.hidden_dim)
        return [h], transition_acc, transition_loss
