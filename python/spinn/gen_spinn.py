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

from spinn.util.blocks import Reduce
from spinn.util.blocks import LSTMState, Embed, MLP
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, treelstm, lstm
from spinn.util.blocks import get_h, get_c, get_seq_h
from spinn.util.misc import Args, Vocab, Example

from spinn.fat_stack import BaseModel as _BaseModel
from spinn.fat_stack import SPINN


T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

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
         gen_h=FLAGS.gen_h,
        )


class GenSPINN(SPINN):

    def __init__(self, args, vocab, use_skips, predict_use_cell, use_lengths):
        super(GenSPINN, self).__init__(args, vocab, use_skips, predict_use_cell, use_lengths)

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

        # TODO: Right now the decoder runs over the entire sentence, which is a bit like cheating!
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

    def shift_phase(self, tops, trackings, stacks, idxs):
        """SHIFT: Should dequeue buffer and item to stack."""

        # Generative Component.
        if len(stacks) > 0:
            h_prev = torch.cat([self.dec_h[batch_idx] for batch_idx in idxs], 1)
            c_prev = torch.cat([self.dec_h[batch_idx] for batch_idx in idxs], 1)

            if self.training:
                # First predict, then run one step of RNN in preparation for next decode.
                w = self.decoder(h_prev.squeeze(0))
                logits = F.log_softmax(w)
                target = np.array([self.tokens[batch_idx].pop() for batch_idx in idxs])

                self.memory['gen_logits'] = logits
                self.memory['gen_target'] = target

            # Run decoder one step in preparation for next shift phase.
            self.run_decoder_rnn(idxs, torch.cat(tops, 0).unsqueeze(1))

        # TODO: Experiment adding the predicted the word to the stack rather than
        # the top of the buffer.

        if len(stacks) > 0:
            shift_candidates = iter(tops)
            for stack in stacks:
                new_stack_item = next(shift_candidates)
                stack.append(new_stack_item)

    def loss_phase_hook(self):
        if self.training:
            target = np.array(reduce(lambda x, y: x + y.tolist(),
            [m["gen_target"] for m in self.memories if "gen_target" in m], []))
            logits = torch.cat([m["gen_logits"] for m in self.memories if "gen_logits" in m], 0)

            # TODO: Probably only the first or last words have any chance of being predicted.
            # Calculate loss.
            target = torch.from_numpy(target).long()
            self.gen_loss = nn.NLLLoss()(logits, Variable(target, volatile=not self.training)) / target.size(0)

            # Calculate accuracy.
            pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability
            self.gen_acc = pred.eq(target).sum() / float(target.size(0))

    def forward(self, example, use_internal_parser=False, validate_transitions=True):
        tokens = example.tokens.data.numpy().tolist()
        tokens = [list(reversed(t)) for t in tokens]
        self.tokens = tokens

        self.reset_decoder(example)

        return super(GenSPINN, self).forward(
            example, use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)


class BaseModel(_BaseModel):

    def __init__(self, gen_h=None, **kwargs):
        self.gen_h = gen_h
        super(BaseModel, self).__init__(**kwargs)

    def build_spinn(self, args, vocab, use_skips, predict_use_cell, use_lengths):
        return GenSPINN(args, vocab, use_skips, predict_use_cell, use_lengths)

    def output_hook(self, output, sentences, transitions, y_batch=None):
        pass

    def get_features_dim(self):
        features_dim = super(BaseModel, self).get_features_dim()
        if self.gen_h:
            features_dim += self.spinn.decoder_dim
        return features_dim

    def build_features(self, h):
        features = super(BaseModel, self).build_features(h)
        if self.gen_h:
            decoder_h = torch.cat(self.spinn.dec_h, 0).squeeze()
            features = torch.cat([features, decoder_h], 1)
        return features
