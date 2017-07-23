
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from spinn.util.blocks import Embed, to_gpu, MLP, Linear
from spinn.util.misc import Args, Vocab
from spinn.util.blocks import SimpleTreeLSTM
from spinn.util.sparks import sparks

# Source: https://github.com/nyu-mll/unsupervised-treelstm/commit/bbe1946e123e396362ecd071d1673766013463f2
# Original author of core encoder: Jihun Choi, Seoul National Univ.

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
                     test_temperature_multiplier=FLAGS.pyramid_test_time_temperature_multiplier,
                     selection_dim=FLAGS.pyramid_selection_dim,
                     gumbel=FLAGS.pyramid_gumbel,
                     )


class BinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, intra_attention,
                 gumbel_temperature):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.gumbel_temperature = gumbel_temperature

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
        self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim)
        self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal(self.leaf_rnn_cell.weight_hh.data)
            init.constant(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
        self.treelstm_layer.reset_parameters()
        init.normal(self.comp_query.data, mean=0, std=0.01)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2).expand_as(new_h)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = dot_nd(query=self.comp_query, candidates=new_h)
        if self.training:
            select_mask = st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature,
                mask=mask)
        else:
            select_mask = greedy_select(logits=comp_weights, mask=mask)
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask_leftmost_col = Variable(
            select_mask_cumsum.data.new(new_h.size(0), 1).zero_())
        right_mask = torch.cat(
            [right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, input, length, return_select_masks=False):
        max_depth = input.size(1)
        length_mask = sequence_mask(sequence_length=length,
                                          max_length=max_depth)
        select_masks = []

        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = input.size()
            zero_state = Variable(input.data.new(batch_size, self.hidden_dim)
                                  .zero_())
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=input[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)
            state = (hs, cs)
        else:
            state = input.chunk(num_chunks=2, dim=2)
        nodes = []
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
                new_h, new_c, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i+1:])
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                if self.intra_attention:
                    nodes.append(selected_h)
            done_mask = length_mask[:, i+1]
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
            att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes)
            nodes = nodes * att_mask_expand
            # nodes_mean: (batch_size, hidden_dim, 1)
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            # att_weights: (batch_size, num_tree_nodes)
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = masked_softmax(
                logits=att_weights, mask=att_mask)
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1)
        assert h.size(1) == 1 and c.size(1) == 1
        if not return_select_masks:
            return h.squeeze(1), c.squeeze(1)
        else:
            return h.squeeze(1), c.squeeze(1), select_masks


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
                 test_temperature_multiplier=None,
                 selection_dim=None,
                 gumbel=None,
                 **kwargs
                 ):
        super(ChoiPyramid, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature
        self.model_dim = model_dim
        self.test_temperature_multiplier = test_temperature_multiplier
        self.trainable_temperature = trainable_temperature
        self.gumbel = gumbel
        self.selection_dim = selection_dim

        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.embedding_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(word_embedding_dim, vocab.size, vectors=vocab.vectors)

        self.binary_tree_lstm = BinaryTreeLSTM(word_embedding_dim, model_dim / 2, False, False, 1.0)

        mlp_input_dim = self.get_features_dim()

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
                       num_mlp_layers, mlp_ln, self.classifier_dropout_rate)

        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

        # For sample printing and logging
        self.merge_sequence_memory = None
        self.inverted_vocabulary = None
        self.temperature_to_display = 0.0

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        embeds = self.embed(x)
        embeds = self.reshape_input(embeds, batch_size, seq_length)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, batch_size, seq_length)
        embeds = torch.cat([b.unsqueeze(0) for b in torch.chunk(embeds, batch_size, 0)], 0)
        embeds = F.dropout(embeds, self.embedding_dropout_rate, training=self.training)

        return embeds

    def forward(self, sentences, _, __=None, example_lengths=None, show_sample=False,
                pyramid_temperature_multiplier=1.0, **kwargs):
        # Useful when investigating dynamic batching:
        # self.seq_lengths = sentences.shape[1] - (sentences == 0).sum(1)

        x, example_lengths = self.unwrap(sentences, example_lengths)

        emb = self.run_embed(x)

        batch_size, seq_len, model_dim = emb.data.size()
        example_lengths_var = to_gpu(Variable(torch.from_numpy(example_lengths))).long()

        hh, _ = self.binary_tree_lstm(emb, example_lengths_var, return_select_masks=False)

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

        return to_gpu(Variable(torch.from_numpy(x), volatile=not self.training)), lengths

    def wrap_sentence_pair(self, hh):
        batch_size = hh.size(0) / 2
        h = ([hh[:batch_size], hh[batch_size:]])
        return h

    # --- Sentence Pair Specific ---

    def unwrap_sentence(self, sentences, lengths=None):
        return to_gpu(Variable(torch.from_numpy(sentences), volatile=not self.training)), lengths

    def wrap_sentence(self, hh):
        return hh

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


def affine_nd(input, weight, bias):
    """
    An helper function to make applying the "wx + b" operation for
    n-dimensional x easier.

    Args:
        input (Variable): An arbitrary input data, whose size is
            (d0, d1, ..., dn, input_dim)
        weight (Variable): A matrix of size (output_dim, input_dim)
        bias (Variable): A bias vector of size (output_dim,)

    Returns:
        output: The result of size (d0, ..., dn, output_dim)
    """

    input_size = input.size()
    input_flat = input.view(-1, input_size[-1])
    bias_expand = bias.unsqueeze(0).expand(input_flat.size(0), bias.size(0))
    output_flat = torch.addmm(bias_expand, input_flat, weight)
    output_size = input_size[:-1] + (weight.size(1),)
    output = output_flat.view(*output_size)
    return output


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
        probs = probs / probs.sum(1).expand_as(probs)
    return probs


def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1].squeeze(1),
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
    y_argmax = y.max(1)[1].squeeze(1)
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
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
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.comp_linear.weight.data)
        init.constant(self.comp_linear.bias.data, val=0)

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
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = apply_nd(fn=self.comp_linear, input=hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(num_chunks=5, dim=2)
        c = (cl*(fl + 1).sigmoid() + cr*(fr + 1).sigmoid()
             + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


