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
from spinn.util.blocks import get_h, get_c
from spinn.util.misc import Args, Vocab, Example

from spinn.fat_stack import BaseModel, SentenceModel, SentencePairModel
from spinn.fat_stack import SPINN


import spinn.cbow


T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1


class SentencePairTrainer(BaseSentencePairTrainer): pass


class SentenceTrainer(SentencePairTrainer): pass


class RAESPINN(SPINN):

    def __init__(self, args, vocab, use_skips=False):
        super(RAESPINN, self).__init__(args, vocab, use_skips=use_skips)
        model_dim = args.size * 2
        self.decompose = nn.Linear(model_dim, model_dim * 2)

    def reduce_phase_hook(self, lefts, rights, trackings, reduce_stacks):
        if len(reduce_stacks) > 0:
            for left, right, stack in zip(lefts, rights, reduce_stacks):
                new_stack_item = stack[-1]
                new_stack_item.isleaf = False
                new_stack_item.left = left
                new_stack_item.right = right
                if not hasattr(left, 'isleaf'):
                    left.isleaf = True
                if not hasattr(right, 'isleaf'):
                    right.isleaf = True

    def reconstruct(self, roots):
        if len(roots) == 0:
            return []

        LR = F.tanh(self.decompose(torch.cat(roots, 0)))
        left, right = torch.chunk(LR, 2, 1)
        lefts = torch.chunk(left, len(roots), 0)
        rights = torch.chunk(right, len(roots), 0)

        done = []
        new_roots = []

        for L, R, root in zip(lefts, rights, roots):
            done.append((L, Variable(root.left.data, volatile=not self.training)))
            done.append((R, Variable(root.right.data, volatile=not self.training)))
            if not root.left.isleaf:
                new_roots.append(root.left)
            if not root.right.isleaf:
                new_roots.append(root.right)

        return done + self.reconstruct(new_roots)

    def loss_phase_hook(self):
        if self.training: # only RAE during train time.
            done = self.reconstruct([stack[-1] for stack in self.stacks if not stack[-1].isleaf])
            inp, target = zip(*done)
            inp = torch.cat(inp, 0)
            target = torch.cat(target, 0)
            similarity = Variable(torch.ones(inp.size(0)), volatile=not self.training)
            self.rae_loss = nn.CosineEmbeddingLoss()(inp, target, similarity)


class RAEBaseModel(BaseModel):

    def build_spinn(self, args, vocab, use_skips):
        return RAESPINN(args, vocab, use_skips=use_skips)


class SentencePairModel(RAEBaseModel):

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


class SentenceModel(RAEBaseModel):

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
