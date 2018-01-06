# Source: https://github.com/nyu-mll/unsupervised-treelstm/commit/bbe1946e123e396362ecd071d1673766013463f2
# Original author of core encoder: Jihun Choi, Seoul National Univ.

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from spinn.util.blocks import Embed, to_gpu, MLP, Linear, LayerNormalization
from spinn.util.misc import Args, Vocab


def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args, **kwargs):
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA
    model_cls = Maillard

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
                     right_branching=FLAGS.right_branching,
                     debug_branching=FLAGS.debug_branching,
                     uniform_branching=FLAGS.uniform_branching,
                     random_branching=FLAGS.random_branching,
                     st_gumbel=FLAGS.st_gumbel
                     )


class Maillard(nn.Module):

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
                 right_branching=None,
                 debug_branching=None,
                 uniform_branching=None,
                 random_branching=None,
                 st_gumbel=None,
                 **kwargs
                 ):
        super(Maillard, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature
        self.model_dim = model_dim
        self.trainable_temperature = trainable_temperature
        self.right_branching = right_branching
        self.debug_branching = debug_branching
        self.uniform_branching = uniform_branching
        self.random_branching = random_branching
        self.st_gumbel = st_gumbel

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
            model_dim // 2,
            False,
            composition_ln=composition_ln,
            trainable_temperature=trainable_temperature,
            right_branching=right_branching,
            debug_branching=debug_branching,
            uniform_branching=uniform_branching,
            random_branching=random_branching,
            st_gumbel=st_gumbel)

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
            pyramid_temperature_multiplier=None,
            **kwargs):
        # Useful when investigating dynamic batching:
        # self.seq_lengths = sentences.shape[1] - (sentences == 0).sum(1)

        x, example_lengths = self.unwrap(sentences, example_lengths)
        emb = self.run_embed(x)

        batch_size, seq_len, model_dim = emb.data.size()
        example_lengths_var = to_gpu(
            Variable(torch.from_numpy(example_lengths))).long()

        # TODO: chunk once, pass chunked embs to tree_lstm
        hh, _, masks, temperature = self.binary_tree_lstm(
            emb, example_lengths_var, temperature_multiplier=pyramid_temperature_multiplier)

        self.mask_memory = masks

        if self.training:
            self.temperature_to_display = temperature

        #if store_parse_masks:
        #    self.mask_memory = [mask.data.cpu().numpy() for mask in masks]

        h = self.wrap(hh)
        output = self.mlp(self.build_features(h))

        return output

    def get_features_dim(self):
        features_dim = self.model_dim if self.use_sentence_pair else self.model_dim // 2
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += self.model_dim // 2
            if self.use_product_feature:
                features_dim += self.model_dim // 2
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

                token_sequences.append(self.get_sample_merge_sequence(b, s, batch_size, token_sequence))
        return token_sequences

    """
    def get_sample_merge_sequence(self, b, s, batch_size):
        merge_sequence = []
        index = b + (s * batch_size)
        for mask in self.mask_memory:
            merge_sequence.append(np.argmax(mask[index, :]))
        merge_sequence.append(0)
        return merge_sequence
    """

    def get_sample_merge_sequence(self, b, s, batch_size, sent):
        mask = self.mask_memory
        index = b + (s * batch_size)
        def compose(l, r):
            return "( " + l + " " + r + " )"

        chart = [sent]
        choices = [sent]
        for row in range(1, len(sent)): 
            chart.append([])
            choices.append([])
            for col in range(len(sent) - row):
                chart[row].append(None)
                choices[row].append(None)
        
        for row in range(1, len(sent)): # = len(l_hiddens)
            for col in range(len(sent) - row):
                versions = []
                for i in range(row):
                    versions.append(compose(chart[row-i-1][col], chart[i][row+col-i]))
                max_ind = torch.max(self.mask_memory[row][col], dim=1)[1][index]
                max_ind = int(max_ind.data.cpu().numpy())
                chart[row][col] = versions[max_ind] #.gather(1, ids.view(-1,1))
                choices[row][col] = versions
                l = len(versions)
        
        max_ind = torch.max(self.mask_memory[-1][-1], dim=1)[1][index]
        max_ind = int(max_ind.data.cpu().numpy())
        return choices[-1][-1][max_ind]
    
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
        batch_size = hh.size(0) // 2
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
                 composition_ln=False, trainable_temperature=False, right_branching=False, debug_branching=False, uniform_branching=False, random_branching=False, st_gumbel=False):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.intra_attention = intra_attention
        self.treelstm_layer = BinaryTreeLSTMLayer(
            hidden_dim, composition_ln=composition_ln)
        self.right_branching = right_branching
        self.debug_branching = debug_branching
        self.uniform_branching = uniform_branching
        self.random_branching = random_branching
        self.st_gumbel = st_gumbel

        # TODO: Add something to blocks to make this use case more elegant.
        self.comp_query = Linear()(
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

    def compute_compositions(
            self,
            state,
            mask,
            temperature_multiplier=1.0):
        """
        (In a parallelized manner) Compute all compositions we will need 
        for the current step.
        Input to || computation: (h,c) l and r.
        Input to serial computation: list of (h,c) l and r.
        don't pass full chart around. only values that'll be needed for computation

        currently: passing full sentence at each laye. FIX!!

        Example:
        [['A', 'B', 'C', 'D', 'E', 'F', 'G', '.'],
        ['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'G.'],
        ['ABC', 'BCD', 'CDE', 'DEF', 'EFG', 'FG.'],
        ['ABCD', 'BCDE', 'CDEF', 'DEFG', 'EFG.'],
        ['ABCDE', 'BCDEF', 'CDEFG', 'DEFG.'],
        ['ABCDEF', 'BCDEFG', 'CDEFG.'],
        ['ABCDEFG', 'BCDEFG.'],
        ['ABCDEFG.']]

        TODO: do masking to prevent composing with padding.
        """
        # make sure are embeddigns
        h, c = state # batch, length, dim
        mask = mask # batch, length

        length = h.size(1)
        word_hiddens = h.chunk(length, dim=1)
        word_cells = c.chunk(length, dim=1)
        length_masks = mask.chunk(length, dim=1) # (batch,1), ...
        temperature = temperature_multiplier
        

        if self.debug_branching:
            hiddens = word_hiddens
            cells = word_cells
            (h, c) = (hiddens[-1], cells[-1])
            for i in range(length-1, -1, -1):
                r = (h, c) #* length_masks[i+1]
                l  = (hiddens[i], cells[i]) #* length_masks[i+1]
                h, c = self.treelstm_layer(l=l, r=r)
            return h, c, None

        else: 
            chart = [[]]
            all_weights = [[]]

            for i in range(length):
                chart[0].append((word_hiddens[i], word_cells[i]))
                all_weights[0].append(None)
            for row in range(1, length):
                chart.append([])
                all_weights.append([])
                for col in range(length - row):
                    chart[row].append(None)
                    all_weights[row].append(None)

            for row in range(1, length):
                for col in range(length - row):
                    states = []
                    hiddens = []
                    cells = []
                    scores = []
                    for i in range(row):
                        l = chart[row-i-1][col]
                        r = chart[i][row+col-i]
                        states.append(self.treelstm_layer(l=l, r=r))
                        hiddens.append(states[-1][0])
                        cells.append(states[-1][1])
                        comp_weights = dot_nd(
                            query=self.comp_query.weight.squeeze(),
                            candidates=hiddens[-1]) # batch, 1
                        scores.append(comp_weights) # [(batch, 1), ...]

                    if self.right_branching:
                        weights =  to_gpu(Variable(torch.zeros(torch.cat(scores, dim=1).size())))
                        for k in range(weights.size(0)): 
                            weights[k, -1] = 1.0
                    elif self.uniform_branching:
                        l = torch.cat(scores, dim=1).size(1)
                        weights = to_gpu(Variable(torch.ones(torch.cat( scores, dim=1).size()) / l ))
                    elif self.random_branching:
                        w_rand = torch.rand(torch.cat(scores, dim=1).size())
                        weights = to_gpu(Variable(w_rand / w_rand.sum(1).unsqueeze(1)))
                    elif self.st_gumbel:
                        weights = st_gumbel_softmax(torch.cat(scores, dim=1), temperature)
                        # TODO: get index/mask and don;t do a linear combination.
                    else:
                        weights = gumbel_softmax(torch.cat(scores, dim=1), temperature) # cat: batch, num_versions, out: batch, num_states

                    h_new = torch.sum(torch.mul(weights.unsqueeze(2), torch.cat(hiddens, dim=1)), dim=1)
                    c_new = torch.sum(torch.mul(weights.unsqueeze(2), torch.cat(cells, dim=1)), dim=1) # batch, num_states, dim

                    chart[row][col] = (h_new.unsqueeze(1), c_new.unsqueeze(1))
                    all_weights[row][col] = weights

            # TODO: FIX, place below into above code. replace all_weights.
            mask = [[]]
            for i in range(length):
                mask[0].append(None)
            for row in range(1, length):
                mask.append([])
                for col in range(length - row):
                    mask[row].append(None)

            for row in range(1, length):
                for col in range(length - row):
                    mask[row][col] = create_max_mask(all_weights[row][col])
            return chart[length-1][0][0], chart[length-1][0][1], mask


    def forward(self, input, length, temperature_multiplier=None):
        max_depth = input.size(1)
        length_mask = sequence_mask(sequence_length=length,
                                    max_length=max_depth)

        #select_masks = []
        state = input.chunk(chunks=2, dim=2)
        #nodes = []
        # For one or two-word trees where we never compute a temperature
        temperature_to_display = -1.0
        """
        if self.intra_attention:
            nodes.append(state[0])
        """

        h, c, masks = self.compute_compositions(state, length_mask, temperature_multiplier)
        
        """
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
        """
        assert h.size(1) == 1 and c.size(1) == 1
        return h.squeeze(1), c.squeeze(1), masks, temperature_to_display


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
    probs = F.softmax(logits, dim=1)
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
    if temperature == 0.0:
        temperature = 1.0
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(
        indices=y_argmax,
        num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y

def create_max_mask(y):
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(
        indices=y_argmax,
        num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y

def gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Gumbel-Softmax Estimation.

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
    if temperature == 0.0:
        temperature = 1.0
    y = masked_softmax(logits=y / temperature, mask=mask)
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

"""
class BinaryTreeLSTMLayer(nn.Module):
    # TODO: Unify with SimpleTreeLSTM

    def __init__(self, hidden_dim, composition_ln=False):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = Linear(
            initializer=HeKaimingInitializer)(
            in_features=2 * hidden_dim,
            out_features=5 * hidden_dim)
        self.composition_ln = composition_ln
        if composition_ln:
            self.left_h_ln = LayerNormalization(hidden_dim)
            self.right_h_ln = LayerNormalization(hidden_dim)
            self.left_c_ln = LayerNormalization(hidden_dim)
            self.right_c_ln = LayerNormalization(hidden_dim)

    def forward(self, l=None, r=None):


        hl, cl = l
        hr, cr = r

        if self.composition_ln:
            hl = self.left_h_ln(hl)
            hr = self.right_h_ln(hr)
            cl = self.left_c_ln(cl)
            cr = self.right_c_ln(cr)

        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = apply_nd(fn=self.comp_linear, input=hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c
"""

class BinaryTreeLSTMLayer(nn.Module):
    def __init__(self, hidden_dim, composition_ln=False):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = Linear()(
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
        i, fl, fr, u, o = treelstm_vector.chunk(5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c
