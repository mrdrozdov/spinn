
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from spinn.util.blocks import Embed, to_gpu, MLP, Linear, HeKaimingInitializer, gumbel_sample
from spinn.util.misc import Args, Vocab
from spinn.util.blocks import SimpleTreeLSTM
from spinn.util.sparks import sparks


def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args, logger=None):
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA
    model_cls = Pyramid

    return model_cls(model_dim=FLAGS.model_dim,
                     word_embedding_dim=FLAGS.word_embedding_dim,
                     vocab_size=vocab_size,
                     initial_embeddings=initial_embeddings,
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
                     gated=FLAGS.pyramid_gated,
                     trainable_temperature=FLAGS.pyramid_trainable_temperature,
                     test_temperature_mulitplier=FLAGS.pyramid_test_time_temperature_multiplier,
                     logger=logger,
                     gumbel=FLAGS.pyramid_gumbel,
                     )


class Pyramid(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 use_sentence_pair=False,
                 classifier_keep_rate=None,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_ln=None,
                 context_args=None,
                 gated=None,
                 trainable_temperature=None,
                 test_temperature_mulitplier=None,
                 logger=None,
                 gumbel=None,
                 **kwargs
                 ):
        super(Pyramid, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.model_dim = model_dim
        self.gated = gated
        self.test_temperature_mulitplier = test_temperature_mulitplier
        self.trainable_temperature = trainable_temperature
        self.logger = logger
        self.gumbel = gumbel

        classifier_dropout_rate = 1. - classifier_keep_rate

        args = Args()
        args.size = model_dim
        args.input_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(word_embedding_dim, vocab.size, vectors=vocab.vectors)

        self.composition_fn = SimpleTreeLSTM(model_dim / 2,
                                             composition_ln=False)
        self.selection_fn = Linear(initializer=HeKaimingInitializer)(model_dim, 1)

        # TODO: Set up layer norm.

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
                       num_mlp_layers, mlp_ln, classifier_dropout_rate)

        if self.trainable_temperature:
            self.temperature = nn.Parameter(torch.ones(1, 1), requires_grad=True)

        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

    def run_hard_pyramid(self, x):
        batch_size, seq_len, model_dim = x.data.size()
        # If we "merge left", that means words before the merge point stay put.
        # Example:
        # 012345678
        #     ++ : 4,5 -> X
        # 0123X678

        words = torch.chunk(x, seq_len, 1) # A list of word vectors
        words = [torch.squeeze(word) for word in words]

        while len(words) > 1:
            composition_results = []
            selection_logits_list = []

            for position in range(len(words) - 1):
                left = words[position]
                right = words[position + 1]
                composed = self.composition_fn(left, right)
                selection_score = self.selection_fn(composition_results[position])
                composition_results.append(composed)
                selection_logits_list.append(selection_score)

            selection_logits = torch.cat(selection_logits_list, 1) # B x L
            merge_points = Variable(selection_logits.data.max(1)[1]) # B

            new_words = []
            for position in range(len(words) - 1): # destination position
                left = words[position])
                right = words[position + 1]

                # If merge at 4, positions 0-3 is copy left, 5-... is copy right
                is_copy_left = Variable(merge_points.data.gt(position).float())
                is_copy_right = Variable(merge_points.data.lt(position).float())
                is_merged = (1 - is_copy_left) * (1 - is_copy_right)

                next_level_word = \
                        is_copy_left.expand_as(left) * left + \
                        is_copy_right.expand_as(right) * right + \
                        is_merged.expand_as(left) * composition_results[position]
                new_words.append(next_level_word)

            words = new_words

        return words[0]

    def run_pyramid(self, x, show_sample=False, temperature_multiplier=1.0):
        batch_size, seq_len, model_dim = x.data.size()

        all_state_pairs = []
        all_state_pairs.append(torch.chunk(x, seq_len, 1))

        if show_sample:
            self.logger.Log('')
            if self.trainable_temperature:
                self.logger.Log('Temp: ' + str(self.temperature.data.cpu().numpy()[0][0]))

        temperature = temperature_multiplier
        if self.trainable_temperature:
            temperature *= self.temperature
        if not self.training:
            temperature *= \
                self.test_temperature_mulitplier

        for layer in range(seq_len - 1, 0, -1):
            composition_results = []
            selection_logits_list = []

            for position in range(layer):
                left = torch.squeeze(all_state_pairs[-1][position])
                right = torch.squeeze(all_state_pairs[-1][position + 1])
                composition_results.append(self.composition_fn(left, right))

            if self.gated:
                for position in range(layer):
                    selection_logits_list.append(self.selection_fn(composition_results[position]))

                selection_logits = torch.cat(selection_logits_list, 1)

                local_temperature = temperature
                if type(local_temperature) is not float:
                    local_temperature = local_temperature.expand_as(selection_logits)

                if self.training and self.gumbel:
                    selection_probs = gumbel_sample(selection_logits, local_temperature)
                else:
                    # Plain softmax
                    selection_logits = selection_logits / local_temperature
                    selection_probs = F.softmax(selection_logits)
 
                if show_sample:
                    self.logger.Log(
                        sparks(np.transpose(selection_probs[0, :].data.cpu().numpy()).tolist()))

                layer_state_pairs = []
                for position in range(layer):
                    if position < (layer - 1):
                        copy_left = torch.sum(selection_probs[:, position + 1:], 1)
                    else:
                        copy_left = to_gpu(Variable(torch.zeros(1, 1)))
                    if position > 0:
                        copy_right = torch.sum(selection_probs[:, :position], 1)
                    else:
                        copy_right = to_gpu(Variable(torch.zeros(1, 1)))
                    select = selection_probs[:, position]

                    left = torch.squeeze(all_state_pairs[-1][position])
                    right = torch.squeeze(all_state_pairs[-1][position + 1])
                    composition_result = composition_results[position]
                    new_state_pair = copy_left.expand_as(left) * left \
                        + copy_right.expand_as(right) * right \
                        + select.unsqueeze(1).expand_as(composition_result) * composition_result
                    layer_state_pairs.append(new_state_pair)
            else:
                layer_state_pairs = composition_results

            all_state_pairs.append(layer_state_pairs)

        return all_state_pairs[-1][-1]

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        embeds = self.embed(x)
        embeds = self.reshape_input(embeds, batch_size, seq_length)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, batch_size, seq_length)
        embeds = torch.cat([b.unsqueeze(0) for b in torch.chunk(embeds, batch_size, 0)], 0)

        return embeds

    def forward(self, sentences, transitions, y_batch=None, show_sample=False, pyramid_temperature_multiplier=1.0, **kwargs):
        # Useful when investigating dynamic batching:
        # self.seq_lengths = sentences.shape[1] - (sentences == 0).sum(1)

        x = self.unwrap(sentences, transitions)
        emb = self.run_embed(x)
        if pyramid_temperature_multiplier == 0.0:
            hh = self.run_hard_pyramid(emb, show_sample, temperature_multiplier=pyramid_temperature_multiplier)
        else:
            hh = self.run_pyramid(emb, show_sample, temperature_multiplier=pyramid_temperature_multiplier)
        h = self.wrap(hh)
        output = self.mlp(h)

        return output

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
        x_prem = sentences[:, :, 0]
        x_hyp = sentences[:, :, 1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))

    def wrap_sentence_pair(self, hh):
        batch_size = hh.size(0) / 2
        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        return h

    # --- Sentence Pair Specific ---

    def unwrap_sentence(self, sentences, transitions):
        return to_gpu(Variable(torch.from_numpy(sentences), volatile=not self.training))

    def wrap_sentence(self, hh):
        return hh
