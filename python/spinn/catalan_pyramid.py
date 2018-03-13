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
from spinn.util.misc import Args, Vocab, Example
from spinn.util.catalan import Catalan

from spinn.spinn_core_model import SPINN

def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args, **kwargs):
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA
    model_cls = CatalanPyramid

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
                     parent_selection=FLAGS.parent_selection,
                     enforce_right=FLAGS.enforce_right,
                     composition_args=composition_args,
                     predict_use_cell=FLAGS.predict_use_cell,
                     low_dim = FLAGS.low_dim,
                     topk = FLAGS.topk,
                     cp_num = FLAGS.cp_num
                     )


class CatalanPyramid(nn.Module):

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
                 enforce_right=None,
                 parent_selection=None,
                 composition_args=None,
                 predict_use_cell=None,
                 low_dim=None,
                 topk=None,
                 cp_num=None,
                 **kwargs
                 ):
        super(CatalanPyramid, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature
        self.model_dim = model_dim
        self.low_dim = low_dim
        self.topk = topk
        self.cp_num = cp_num
        self.trainable_temperature = trainable_temperature
        self.parent_selection = parent_selection
        self.enforce_right = enforce_right

        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.embedding_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(
            word_embedding_dim,
            vocab.size,
            vectors=vocab.vectors)

        self.chart_parser = ChartParser(
            word_embedding_dim,
            model_dim // 2,
            low_dim,
            composition_ln=composition_ln,
            trainable_temperature=trainable_temperature,
            parent_selection=parent_selection,
            use_sentence_pair=use_sentence_pair)

        # assert FLAGS.lateral_tracking == False
        # TODO: move assertion flag to base.
        
        self.spinn = self.build_spinn(
            composition_args, vocab, predict_use_cell)

        mlp_input_dim = self.get_features_dim()

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
                       num_mlp_layers, mlp_ln, self.classifier_dropout_rate)

        # SPINN vars
        self.encode = context_args.encoder
        #self.encode = Linear()(word_embedding_dim, model_dim)
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context
        self.input_dim = context_args.input_dim
        self.wrap_items = composition_args.wrap_items
        self.extract_h = composition_args.extract_h

        # For sample printing and logging
        self.parse_memory = None
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

    def run_embed_spinn(self, x):
        batch_size, seq_length = x.size()

        embeds = self.embed(x)
        embeds = self.reshape_input(embeds, batch_size, seq_length)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, batch_size, seq_length)
        embeds = F.dropout(
            embeds,
            self.embedding_dropout_rate,
            training=self.training)

        return embeds

    def build_spinn(self, args, vocab, predict_use_cell):
        return SPINN(args, vocab, predict_use_cell)

    def forward(
            self,
            sentences,
            _,
            __=None,
            example_lengths=None,
            store_parse_masks=False,
            pyramid_temperature_multiplier=None,
            use_internal_parser=False,
            validate_transitions=True,
            **kwargs):
        # Useful when investigating dynamic batching:
        # self.seq_lengths = sentences.shape[1] - (sentences == 0).sum(1)

        x, example_lengths = self.unwrap(sentences, example_lengths)
        emb = self.run_embed(x)

        batch_size, seq_len, model_dim = emb.data.size()
        example_lengths_var = to_gpu(
            Variable(torch.from_numpy(example_lengths))).long()
        topk = self.topk
        cp_num = self.cp_num

        # Chart-Parsing Choice
        if self.enforce_right: 
            max_length = 2 * seq_len - 1
            shifts = [0. for i in range(max_length // 2 + 1)]
            reduces = [1. for i in range(max_length // 2)]
            right_branch = shifts + reduces
            rb_parses = [right_branch for i in range(batch_size)]
            sr_transitions = [np.asarray(rb_parses)]
            temperature = -1.0 
        else:
            sr_transitions, weights, temperature = self.chart_parser(
            emb, example_lengths_var, topk, cp_num, temperature_multiplier=pyramid_temperature_multiplier)

        # Use SPINN with CP parses
        embeds = self.run_embed_spinn(x)
        b, l = x.size()[:2]
        ee = torch.chunk(embeds, b * l, 0)[::-1]
        self.parse_memory = sr_transitions
        
        h_versions = []
        for transition in sr_transitions:
            example = self.unwrap_spinn(sentences, transition)
            bb = []
            for ii in range(b):
                ex = list(ee[ii * l:(ii + 1) * l])
                bb.append(ex)
            buffers = bb[::-1]
            example.bufs = buffers
            
            h, transition_acc, transition_loss = self.run_spinn(
                example, use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)
            
            h_versions.append(h)

        # Linear combination
        if self.enforce_right:
            hh = h_versions[0]
        else:
            hs = torch.stack(h_versions, dim=1)
            hh = torch.sum(torch.mul(hs, weights.unsqueeze(2)), dim=1)

        if self.training:
            self.temperature_to_display = temperature

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

    def run_spinn(
            self,
            example,
            use_internal_parser,
            validate_transitions=True):
        self.spinn.reset_state()
        h_list, transition_acc, transition_loss = self.spinn(
            example, use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)
        h = self.wrap_spinn(h_list)
        h = torch.cat(h, dim=0)
        return h, transition_acc, transition_loss

    # --- Sample printing ---

    def get_samples(self, x, vocabulary, only_one=False):
        # n=-1: Show all samples.
        if not self.inverted_vocabulary:
            self.inverted_vocabulary = dict(
                [(vocabulary[key], key) for key in vocabulary])

        transitions = self.parse_memory[0]

        token_sequences = []
        batch_size = x.shape[0]
        for s in (list(range(int(self.use_sentence_pair) + 1))
                  if not only_one else [0]):
            for b in (list(range(batch_size)) if not only_one else [0]):
                if self.use_sentence_pair:
                    token_sequence = [self.inverted_vocabulary[token]
                                      for token in x[b, :, s]]
                else:
                    token_sequence = [self.inverted_vocabulary[token]
                                      for token in x[b, :]]

                stack = []
                token_sequence.reverse()
                for transition in transitions[b + (s * batch_size)]:
                    if transition == 0:
                        stack.append(token_sequence.pop())
                    if transition == 1:
                        r = stack.pop()                        
                        l = stack.pop()
                        stack.append((l, r))
                assert len(stack) == 1
                token_sequences.append(stack[0])
        return token_sequences
    
    # --- Sentence Style Switches ---

    def unwrap(self, sentences, lengths=None):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, lengths)
        return self.unwrap_sentence(sentences, lengths)

    def wrap(self, hh):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(hh)
        return self.wrap_sentence(hh)

    def wrap_spinn(self, h_list):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair_spinn(h_list)
        return self.wrap_sentence_spinn(h_list)

    def unwrap_spinn(self, sentences, transitions):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair_spinn(sentences, transitions)
        return self.unwrap_sentence_spinn(sentences, transitions)

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

    def unwrap_sentence_pair_spinn(self, sentences, transitions):
        # Build Tokens
        x_prem = sentences[:, :, 0]
        x_hyp = sentences[:, :, 1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        #t_prem = transitions[:, :, 0]
        #t_hyp = transitions[:, :, 1]
        #t = np.concatenate([t_prem, t_hyp], axis=0)
        t = transitions

        example = Example()
        example.tokens = to_gpu(
            Variable(
                torch.from_numpy(x),
                volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence_pair(self, hh):
        batch_size = hh.size(0) // 2
        h = ([hh[:batch_size], hh[batch_size:]])
        return h

    def wrap_sentence_pair_spinn(self, items):
        batch_size = len(items) // 2
        h_premise = self.extract_h(self.wrap_items(items[:batch_size]))
        h_hypothesis = self.extract_h(self.wrap_items(items[batch_size:]))
        return [h_premise, h_hypothesis]

    # --- Sentence Pair Specific ---

    def unwrap_sentence(self, sentences, lengths=None):
        return to_gpu(Variable(torch.from_numpy(sentences),
                               volatile=not self.training)), lengths

    def unwrap_sentence_spinn(self, sentences, transitions):
        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(
            Variable(
                torch.from_numpy(x),
                volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence(self, hh):
        return hh

    def wrap_sentence_spinn(self, items):
        h = self.extract_h(self.wrap_items(items))
        return [h]

    # --- From Choi's 'treelstm.py' ---


class ChartParser(nn.Module):

    def __init__(self, word_dim, hidden_dim, low_dim,
                 composition_ln=False, trainable_temperature=False, parent_selection=False, use_sentence_pair=False):
        super(ChartParser, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.low_dim = low_dim
        self.use_sentence_pair = use_sentence_pair

        self.treelstm_layer = BinaryTreeLSTMLayer(
            low_dim, composition_ln=composition_ln) #CAT: low_dim from hidden_dim
        self.parent_selection = parent_selection

        self.cat = Catalan()
        self.reduce_dim = Linear()(in_features=hidden_dim, out_features=low_dim)

        # TODO: Add something to blocks to make this use case more elegant.
        self.comp_query = Linear()(
            in_features=low_dim,
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
        alpha,
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

        chart = [[]]
        all_weights = [[]]
        mask = [[]]
        memory = [[]]

        for i in range(length):
            chart[0].append((word_hiddens[i], word_cells[i]))
            all_weights[0].append(None)
            mask[0].append(None)
            memory[0].append(None)
        for row in range(1, length):
            chart.append([])
            all_weights.append([])
            mask.append([])
            memory.append([])
            for col in range(length - row):
                chart[row].append(None)
                all_weights[row].append(None)
                mask[row].append(None)
                memory[row].append(None)

        for row in range(1, length):
            for col in range(length - row):
                states = []
                hiddens = []
                tr_versions = []
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

                if self.parent_selection == "right_branching":
                    weights =  to_gpu(Variable(torch.zeros(torch.cat(scores, dim=1).size())))
                    for k in range(weights.size(0)): 
                        weights[k, -1] = 1.0
                elif self.parent_selection == "uniform_branching":
                    l = torch.cat(scores, dim=1).size(1)
                    weights = to_gpu(Variable(torch.ones(torch.cat(scores, dim=1).size()) / l ))
                elif self.parent_selection == "random_branching":
                    w_rand = torch.rand(torch.cat(scores, dim=1).size())
                    weights = to_gpu(Variable(w_rand / w_rand.sum(1).unsqueeze(1)))
                elif self.parent_selection == "st_gumbel":
                    weights, w_max, w_argmax = st_gumbel_softmax(torch.cat(scores, dim=1), temperature)
                    alpha *= w_max
                elif self.parent_selection == "softmax":
                    weights = masked_softmax(torch.cat(scores, dim=1) / temperature, None)  
                else:
                    weights = gumbel_softmax(torch.cat(scores, dim=1), temperature) # cat: batch, num_versions, out: batch, num_states

                h_new = torch.sum(torch.mul(weights.unsqueeze(2), torch.cat(hiddens, dim=1)), dim=1)
                c_new = torch.sum(torch.mul(weights.unsqueeze(2), torch.cat(cells, dim=1)), dim=1) # batch, num_states, dim

                chart[row][col] = (h_new.unsqueeze(1), c_new.unsqueeze(1))
                all_weights[row][col] = weights
                mask[row][col] = create_max_mask(all_weights[row][col])
                memory[row][col] = w_argmax

        # Use memory of choices to construct binary parse
        transitions = [[]]
        compose_vec = np.ones((h.size(0),1))
        for i in range(length):
            transitions[0].append(np.zeros((h.size(0),1)))
        for row in range(1, length):
            transitions.append([])
            for col in range(length - row):
                transitions[row].append(None)

        for row in range(1, length):
            for col in range(length - row):
                tr_versions = []
                for i in range(row):
                    tr_versions.append(np.concatenate([transitions[row-i-1][col], transitions[i][row+col-i], compose_vec], axis=1))

                index = memory[row][col].cpu().data.numpy()
                # Following loops through batch! 
                # But it's a simple lookup
                tr_new = []
                for i in range(h.size(0)):
                    tr_select = tr_versions[index[i]]
                    tr_new.append(tr_select[i])
                transitions[row][col] = np.asarray(tr_new)

        return chart[length-1][0][0], chart[length-1][0][1], mask, alpha.unsqueeze(1), transitions[length-1][0]

    def forward(self, input, length, topk, cp_num, temperature_multiplier=None):
        max_depth = input.size(1)
        length_mask = sequence_mask(sequence_length=length,
                                    max_length=max_depth)

        state = input.chunk(chunks=2, dim=2)
        h_low = self.reduce_dim(state[0])
        c_low = self.reduce_dim(state[1])
        batch_size = h_low.size(0) 

        # For one or two-word trees where we never compute a temperature
        temperature_to_display = -1.0

        alphas = []
        parses = []

        num = batch_size 
        if self.use_sentence_pair:
            num = num // 2

        if cp_num is not None:
            num  = cp_num

        h_long = [h_low] * (num)
        h_long = torch.cat(h_long, 0)
        c_long = [c_low] * (num)
        c_long = torch.cat(c_long, 0)
        length_mask_long = [length_mask] * (num)
        length_mask_long = torch.cat(length_mask_long, 0)

        #import pdb; pdb.set_trace()

        alpha = to_gpu(Variable(torch.ones(h_long.size(0))))
        h, c, masks, alpha_w, transitions = self.compute_compositions((h_long, c_long), length_mask_long, alpha, temperature_multiplier=1.0)

        alphas = alpha_w.chunk(num, dim=0)
        parses = torch.from_numpy(transitions)
        parses = parses.unsqueeze(1).chunk(num, dim=0)

        alpha_max, alpha_argmax = torch.cat(alphas, dim=1).topk(topk)
        alpha_maxs = alpha_max.chunk(topk, dim=1)
        alpha_args = alpha_argmax.cpu().chunk(topk, dim=1)
        
        parses = torch.cat(parses, dim=1)

        binary_parses = []
        for i in range(topk):
            alpha_hard = convert_to_one_hot(alpha_args[i].squeeze(), parses.size(1))
            parse_hard = torch.sum(torch.mul(alpha_hard.data.double().unsqueeze(2), parses), 1).numpy()
            binary_parses.append(parse_hard)

        # Normalize weights
        alpha_weights = gumbel_softmax(alpha_max)

        return binary_parses, alpha_weights, temperature_to_display


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

def cosine(query, candidates):
    """
    Perform a cosine similarity between a query and n-dimensional candidates.

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
    output_flat = torch.mv(cands_flat, query) / (torch.norm(cands_flat) * torch.norm(query))
    output = output_flat.view(*cands_size[:-1])
    return output

def get_binary_parse(parse):
    #parse_bin = torch.zeros(parse.size())
    parse_len = len(bin(int(parse[0]))[3:])
    parse_bin = np.empty((0,parse_len))
    for i in range(parse.size(0)):
        #parse_bin[i] = bin(parse[i])[3:]
        p_bin = bin(int(parse[i]))[3:]
        p_bin = list(p_bin)
        p_bin = [int(p) for p in p_bin]
        parse_bin = np.append(parse_bin, [p_bin], axis=0)
    # redundant 1 pre-appended, to avoid loss of initial zeros
    #return parse_bin.unsqueeze(1)
    return parse_bin

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
    y_max = y.max(1)[0]
    y_hard = convert_to_one_hot(
        indices=y_argmax,
        num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y, y_max, y_argmax

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
