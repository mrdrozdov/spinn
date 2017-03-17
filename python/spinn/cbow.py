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

from spinn.util.blocks import Embed, to_gpu, MLP
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
         use_sentence_pair=use_sentence_pair,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
        )


class BaseModel(nn.Module):

    def __init__(self,
                 model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 mlp_dim=None,
                 num_mlp_layers=2,
                 mlp_bn=False,
                 embedding_keep_rate=None,
                 classifier_keep_rate=None,
                 use_sentence_pair=False,
                 use_embed=True,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.model_dim = model_dim

        args = Args()
        args.size = model_dim

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        if use_embed:
            self.embed = Embed(args.size, vocab.size, vectors=vocab.vectors)

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_bn)

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        emb = self.embed(x)
        emb = torch.cat([b.unsqueeze(0) for b in torch.chunk(emb, batch_size, 0)], 0)

        return emb


class SentencePairModel(BaseModel):

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        if hasattr(self, 'embed'):
            # Build Tokens
            x = self.build_example(sentences, transitions)
            emb = self.run_embed(x)
        else:
            emb = sentences
        batch_size = emb.size(0) / 2

        hh = torch.squeeze(torch.sum(emb, 1))
        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        output = self.mlp(h)

        return output


class SentenceModel(BaseModel):

    def build_example(self, sentences, transitions):
        return to_gpu(Variable(torch.from_numpy(sentences), volatile=not self.training))

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        if hasattr(self, 'embed'):
            # Build Tokens
            x = self.build_example(sentences, transitions)
            emb = self.run_embed(x)
        else:
            emb = sentences

        h = torch.squeeze(torch.sum(emb, 1))
        output = self.mlp(h)

        return output
