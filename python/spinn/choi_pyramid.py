# Source: https://github.com/nyu-mll/unsupervised-treelstm/commit/bbe1946e123e396362ecd071d1673766013463f2
# Original author of core encoder: Jihun Choi, Seoul National Univ.

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import kaiming_normal

from spinn.util.blocks import Embed, to_gpu, MLP, Linear, LayerNormalization
from spinn.util.misc import Args, Vocab


def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args, **kwargs):
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA
    model_cls = ChoiPyramid

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
                     composition_ln=FLAGS.composition_ln,
                     context_args=context_args,
                     trainable_temperature=FLAGS.pyramid_trainable_temperature,
                     )


class ChoiPyramid(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 use_product_feature=None,
                 use_difference_feature=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 use_sentence_pair=False,
                 classifier_keep_rate=None,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_ln=None,
                 composition_ln=None,
                 context_args=None,
                 trainable_temperature=None,
                 **kwargs
                 ):
        super(ChoiPyramid, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature
        self.model_dim = model_dim
        self.trainable_temperature = trainable_temperature

        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.embedding_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(
            word_embedding_dim,
            vocab.size,
            vectors=vocab.vectors)

        self.binary_tree_lstm = BinaryTreeLSTM(
            word_embedding_dim,
            model_dim / 2,
            False,
            composition_ln=composition_ln,
            trainable_temperature=trainable_temperature)

        mlp_input_dim = self.get_features_dim()

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
                       num_mlp_layers, mlp_ln, self.classifier_dropout_rate)

        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

        # For sample printing and logging
        self.mask_memory = None
        self.inverted_vocabulary = None
        self.temperature_to_display = 0.0

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        embeds = self.embed(x)
        embeds = self.reshape_input(embeds, batch_size, seq_length)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, batch_size, seq_length)
        embeds = torch.cat([b.unsqueeze(0)
                            for b in torch.chunk(embeds, batch_size, 0)], 0)
        embeds = F.dropout(
            embeds,
            self.embedding_dropout_rate,
            training=self.training)

        return embeds

    def forward(
            self,
            sentences,
            _,
            __=None,
            example_lengths=None,
            store_parse_masks=False,
            pyramid_temperature_multiplier=1.0,
            **kwargs):
        # Useful when investigating dynamic batching:
        # self.seq_lengths = sentences.shape[1] - (sentences == 0).sum(1)

        x, example_lengths = self.unwrap(sentences, example_lengths)

        emb = self.run_embed(x)

        batch_size, seq_len, model_dim = emb.data.size()
        example_lengths_var = to_gpu(
            Variable(torch.from_numpy(example_lengths))).long()

        hh, _, masks, temperature = self.binary_tree_lstm(
            emb, example_lengths_var, temperature_multiplier=pyramid_temperature_multiplier)

        if self.training:
            self.temperature_to_display = temperature

        if store_parse_masks:
            self.mask_memory = [mask.data.cpu().numpy() for mask in masks]

        h = self.wrap(hh)
        output = self.mlp(self.build_features(h))

        return output

    def get_features_dim(self):
        features_dim = self.model_dim if self.use_sentence_pair else self.model_dim / 2
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += self.model_dim / 2
            if self.use_product_feature:
                features_dim += self.model_dim / 2
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

    # --- Sample printing ---

    def get_samples(self, x, vocabulary, only_one=False):
        # TODO: Don't show padding.

        if not self.inverted_vocabulary:
            self.inverted_vocabulary = dict(
                [(vocabulary[key], key) for key in vocabulary])

        token_sequences = []
        batch_size = x.shape[0]
        for s in (range(int(self.use_sentence_pair) + 1)
                  if not only_one else [0]):
            for b in (range(batch_size) if not only_one else [0]):
                if self.use_sentence_pair:
                    token_sequence = [self.inverted_vocabulary[token]
                                      for token in x[b, :, s]]
                else:
                    token_sequence = [self.inverted_vocabulary[token]
                                      for token in x[b, :]]

                for merge in self.get_sample_merge_sequence(b, s, batch_size):
                    if len(token_sequence) <= 1:
                        # For padding quirks around single-word sentences.
                        break
                    token_sequence[merge] = (
                        token_sequence[merge], token_sequence[merge + 1])
                    del token_sequence[merge + 1]
                assert len(token_sequence) == 1
                token_sequences.append(token_sequence[0])
        return token_sequences

    def get_sample_merge_sequence(self, b, s, batch_size):
        merge_sequence = []
        index = b + (s * batch_size)
        for mask in self.mask_memory:
            merge_sequence.append(np.argmax(mask[index, :]))
        merge_sequence.append(0)
        return merge_sequence

    # --- Sentence Style Switches ---

    def unwrap(self, sentences, lengths=None):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, lengths)
        return self.unwrap_sentence(sentences, lengths)

    def wrap(self, hh):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(hh)
        return self.wrap_sentence(hh)

    # --- Sentence Specific ---

    def unwrap_sentence_pair(self, sentences, lengths=None):
        x_prem = sentences[:, :, 0]
        x_hyp = sentences[:, :, 1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        if lengths is not None:
            len_prem = lengths[:, 0]
            len_hyp = lengths[:, 1]
            lengths = np.concatenate([len_prem, len_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(
            x), volatile=not self.training)), lengths

    def wrap_sentence_pair(self, hh):
        batch_size = hh.size(0) / 2
        h = ([hh[:batch_size], hh[batch_size:]])
        return h

    # --- Sentence Pair Specific ---

    def unwrap_sentence(self, sentences, lengths=None):
        return to_gpu(Variable(torch.from_numpy(sentences),
                               volatile=not self.training)), lengths

    def wrap_sentence(self, hh):
        return hh

    # --- From Choi's 'treelstm.py' ---


class BinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_dim, intra_attention,
                 composition_ln=False, trainable_temperature=False):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.intra_attention = intra_attention
        self.treelstm_layer = BinaryTreeLSTMLayer(
            hidden_dim, composition_ln=composition_ln)

        # TODO: Add something to blocks to make this use case more elegant.
        self.comp_query = Linear(
            initializer=kaiming_normal)(
            in_features=hidden_dim,
            out_features=1)
        self.trainable_temperature = trainable_temperature
        if self.trainable_temperature:
            self.temperature_param = nn.Parameter(
                torch.ones(1, 1), requires_grad=True)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(
            self,
            old_state,
            new_state,
            mask,
            temperature_multiplier=1.0):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = dot_nd(
            query=self.comp_query.weight.squeeze(),
            candidates=new_h)
        if self.training:
            temperature = temperature_multiplier
            if self.trainable_temperature:
                temperature *= F.relu(self.temperature_param)
            if not isinstance(temperature, float):
                temperature_to_display = float(temperature.data.cpu().numpy())
            else:
                temperature_to_display = temperature

            local_temperature = temperature
            if not isinstance(local_temperature, float):
                local_temperature = local_temperature

            select_mask = st_gumbel_softmax(
                logits=comp_weights, temperature=local_temperature,
                mask=mask)
        else:
            select_mask = greedy_select(logits=comp_weights, mask=mask)
            select_mask = select_mask.float()
            temperature_to_display = None
        select_mask_expand = select_mask.unsqueeze(2)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2)
        right_mask_leftmost_col = Variable(
            select_mask_cumsum.data.new(new_h.size(0), 1).zero_())
        right_mask = torch.cat(
            [right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h, temperature_to_display

    def forward(self, input, length, temperature_multiplier=1.0):
        max_depth = input.size(1)
        length_mask = sequence_mask(sequence_length=length,
                                    max_length=max_depth)
        select_masks = []
        state = input.chunk(num_chunks=2, dim=2)
        nodes = []
        # For one or two-word trees where we never compute a temperature
        temperature_to_display = -1.0
        if self.intra_attention:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            l = (h[:, :-1, :], c[:, :-1, :])
            r = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(l=l, r=r)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h, temperature_to_display = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i + 1:], temperature_multiplier=temperature_multiplier)
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                if self.intra_attention:
                    nodes.append(selected_h)
            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if self.intra_attention and i >= max_depth - 2:
                nodes.append(state[0])
        h, c = state
        if self.intra_attention:
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask = att_mask.float()
            # nodes: (batch_size, num_tree_nodes, hidden_dim)
            nodes = torch.cat(nodes, dim=1)
            att_mask_expand = att_mask.unsqueeze(2)
            nodes = nodes * att_mask_expand
            # nodes_mean: (batch_size, hidden_dim, 1)
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            # att_weights: (batch_size, num_tree_nodes)
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = masked_softmax(
                logits=att_weights, mask=att_mask)
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            att_weights_expand = att_weights.unsqueeze(2)
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1)
        assert h.size(1) == 1 and c.size(1) == 1
        return h.squeeze(1), c.squeeze(1), select_masks, temperature_to_display

    # --- From Choi's 'basic.py' ---


def apply_nd(fn, input):
    """
    Apply fn whose output only depends on the last dimension values
    to an arbitrary n-dimensional input.
    It flattens dimensions except the last one, applies fn, and then
    restores the original size.
    """

    x_size = input.size()
    x_flat = input.view(-1, x_size[-1])
    output_flat = fn(x_flat)
    output_size = x_size[:-1] + (output_flat.size(-1),)
    return output_flat.view(*output_size)


def dot_nd(query, candidates):
    """
    Perform a dot product between a query and n-dimensional candidates.

    Args:
        query (Variable): A vector to query, whose size is
            (query_dim,)
        candidates (Variable): A n-dimensional tensor to be multiplied
            by query, whose size is (d0, d1, ..., dn, query_dim)

    Returns:
        output: The result of the dot product, whose size is
            (d0, d1, ..., dn)
    """

    cands_size = candidates.size()
    cands_flat = candidates.view(-1, cands_size[-1])
    output_flat = torch.mv(cands_flat, query)
    output = output_flat.view(*cands_size[:-1])
    return output


def convert_to_one_hot(indices, num_classes):
    """
    Args:
        indices (Variable): A vector containing indices,
            whose size is (batch_size,).
        num_classes (Variable): The number of classes, which would be
            the second dimension of the resulting one-hot matrix.

    Returns:
        result: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = Variable(indices.data.new(batch_size, num_classes).zero_()
                       .scatter_(1, indices.data, 1))
    return one_hot


def masked_softmax(logits, mask=None):
    eps = 1e-20
    probs = F.softmax(logits)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(1, keepdim=True)
    return probs


def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1],
                                 num_classes=logits.size(1))
    return one_hot


def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.

    Args:
        logits (Variable): A un-normalized probability values,
            which has the size (batch_size, num_classes)
        temperature (float): A temperature parameter. The higher
            the value is, the smoother the distribution is.
        mask (Variable, optional): If given, it masks the softmax
            so that indices of '0' mask values are not selected.
            The size is (batch_size, num_classes).

    Returns:
        y: The sampled output, which has the property explained above.
    """

    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = Variable(-torch.log(-torch.log(u + eps) + eps))
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(
        indices=y_argmax,
        num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y


def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1)
    return seq_range_expand < seq_length_expand


class BinaryTreeLSTMLayer(nn.Module):
    # TODO: Unify with SimpleTreeLSTM

    def __init__(self, hidden_dim, composition_ln=False):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = Linear(
            initializer=kaiming_normal)(
            in_features=2 * hidden_dim,
            out_features=5 * hidden_dim)
        self.composition_ln = composition_ln
        if composition_ln:
            self.left_h_ln = LayerNormalization(hidden_dim)
            self.right_h_ln = LayerNormalization(hidden_dim)
            self.left_c_ln = LayerNormalization(hidden_dim)
            self.right_c_ln = LayerNormalization(hidden_dim)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """

        hl, cl = l
        hr, cr = r

        if self.composition_ln:
            hl = self.left_h_ln(hl)
            hr = self.right_h_ln(hr)
            cl = self.left_c_ln(cl)
            cr = self.right_c_ln(cr)

        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = apply_nd(fn=self.comp_linear, input=hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(num_chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c
