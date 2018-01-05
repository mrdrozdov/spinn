
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

from spinn.util.blocks import Embed, to_gpu, MLP
from spinn.util.misc import Args, Vocab


def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args, **kwargs):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(
        model_dim=FLAGS.model_dim,
        word_embedding_dim=FLAGS.word_embedding_dim,
        vocab_size=vocab_size,
        initial_embeddings=initial_embeddings,
        fine_tune_loaded_embeddings=FLAGS.fine_tune_loaded_embeddings,
        num_classes=num_classes,
        embedding_keep_rate=FLAGS.embedding_keep_rate,
        use_sentence_pair=use_sentence_pair,
        use_difference_feature=FLAGS.use_difference_feature,
        use_product_feature=FLAGS.use_product_feature,
        classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
        mlp_dim=FLAGS.mlp_dim,
        num_mlp_layers=FLAGS.num_mlp_layers,
        mlp_ln=FLAGS.mlp_ln,
        context_args=context_args,
    )


class BaseModel(nn.Module):

    def __init__(self,
                 model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 fine_tune_loaded_embeddings=None,
                 use_difference_feature=None,
                 use_product_feature=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 classifier_keep_rate=None,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_ln=None,
                 use_sentence_pair=False,
                 context_args=None,
                 **kwargs
                 ):
        super(BaseModel, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature

        self.model_dim = model_dim

        classifier_dropout_rate = 1. - classifier_keep_rate

        args = Args()
        args.size = model_dim

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(
            word_embedding_dim,
            vocab.size,
            vectors=vocab.vectors,
            fine_tune=fine_tune_loaded_embeddings)

        mlp_input_dim = self.get_features_dim()

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
                       num_mlp_layers, mlp_ln, classifier_dropout_rate)

        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        embeds = self.embed(x)
        embeds = self.reshape_input(embeds, batch_size, seq_length)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, batch_size, seq_length)
        embeds = torch.cat([b.unsqueeze(0)
                            for b in torch.chunk(embeds, batch_size, 0)], 0)

        return embeds

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        sentences.shape[0]

        x = self.unwrap(sentences, transitions)
        emb = self.run_embed(x)
        hh = torch.squeeze(torch.sum(emb, 1))
        h = self.wrap(hh)
        output = self.mlp(self.build_features(h))

        return output

    def get_features_dim(self):
        features_dim = self.model_dim * 2 if self.use_sentence_pair else self.model_dim
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += self.model_dim
            if self.use_product_feature:
                features_dim += self.model_dim
        return features_dim

    def build_features(self, h):
        if self.use_sentence_pair:
            h_prem, h_hyp = h
            features = [h_prem, h_hyp]
            if self.use_difference_feature:
                features.append(h_prem - h_hyp)
            if self.use_product_feature:
                features.append(h_prem * h_hyp)
            features = torch.cat(features, 1)
        else:
            features = h
        return features

    # --- Sentence Style Switches ---

    def unwrap(self, sentences, transitions):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, transitions)
        return self.unwrap_sentence(sentences, transitions)

    def wrap(self, hh):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(hh)
        return self.wrap_sentence(hh)

    # --- Sentence Specific ---

    def unwrap_sentence_pair(self, sentences, transitions):
        sentences.shape[0]

        x_prem = sentences[:, :, 0]
        x_hyp = sentences[:, :, 1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        return to_gpu(
            Variable(
                torch.from_numpy(x),
                volatile=not self.training))

    def wrap_sentence_pair(self, hh):
        batch_size = hh.size(0) // 2
        h = ([hh[:batch_size], hh[batch_size:]])
        return h

    # --- Sentence Pair Specific ---

    def unwrap_sentence(self, sentences, transitions):
        return to_gpu(
            Variable(
                torch.from_numpy(sentences),
                volatile=not self.training))

    def wrap_sentence(self, hh):
        return hh
