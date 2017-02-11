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

from spinn.util.blocks import BaseSentencePairTrainer, Embed, to_gpu
from spinn.util.misc import Args, Vocab


class SentencePairTrainer(BaseSentencePairTrainer): pass


class SentenceTrainer(SentencePairTrainer): pass


class BaseModel(nn.Module):

    def __init__(self,
                 model_dim=None,
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

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(args.size, vocab.size,
                        vectors=vocab.vectors,
                        )

        mlp_input_dim = word_embedding_dim * 2 if use_sentence_pair else model_dim

        self.l0 = nn.Linear(mlp_input_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)

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

        hh = torch.squeeze(torch.sum(emb, 1))
        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        output = self.run_mlp(h)

        return output


class SentenceModel(BaseModel):

    def build_example(self, sentences, transitions):
        return to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        # Build Tokens
        x = self.build_example(sentences, transitions)

        emb = self.run_embed(x)

        h = torch.squeeze(torch.sum(emb, 1))
        output = self.run_mlp(h)

        return output
