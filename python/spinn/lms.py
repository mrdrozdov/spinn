import itertools

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from spinn.util.blocks import Embed, Linear, MLP, Lift
from spinn.util.blocks import bundle, lstm, to_gpu, unbundle
from spinn.util.blocks import LayerNormalization
from spinn.util.misc import Example, Vocab
from spinn.util.blocks import HeKaimingInitializer
from spinn.util.catalan import ShiftProbabilities

from spinn.data import T_SHIFT, T_REDUCE, T_SKIP


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS, context_args, composition_args, **kwargs):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(
        model_dim=FLAGS.model_dim,
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
        composition_args=composition_args,
        detach=FLAGS.transition_detach,
        evolution=FLAGS.evolution,
    )


class LMS(nn.Module):

    def __init__(self, args, vocab):
        super(LMS, self).__init__()

        # Optional debug mode.
        self.debug = False

        self.wrap_items = args.wrap_items
        self.extract_h = args.extract_h

        # Reduce function for semantic composition.
        self.reduce = args.composition
        self.choices = np.array([T_SHIFT, T_REDUCE], dtype=np.int32)

    def reset_state(self):
        self.memories = []

    def forward(self, example, use_internal_parser=False, validate_transitions=True):
        self.n_tokens = (
            example.tokens.data != 0).long().sum(
            1, keepdim=False).tolist()
        # self.buffers_n = (example.tokens.data != 0).long().sum(1).view(-1).tolist()

        if self.debug:
            seq_length = example.tokens.size(1)
            assert all(buf_n <= (seq_length + 1) // 2 for buf_n in self.buffers_n), \
                "All sentences (including cropped) must be the appropriate length."

        self.bufs = example.bufs

        # Notes on adding zeros to bufs/stacks.
        # - After the buffer is consumed, we need one zero on the buffer
        #   used as input to the tracker.
        # - For the first two steps, the stack would be empty, but we add
        #   zeros so that the tracker still gets input.
        zeros = self.zeros = to_gpu(Variable(torch.from_numpy(
            np.zeros(self.bufs[0][0].size(), dtype=np.float32)),
            volatile=self.bufs[0][0].volatile))

        # Initialize Buffers. Trim unused tokens.
        self.bufs = [[zeros] + b[-b_n:]
                     for b, b_n in zip(self.bufs, self.n_tokens)]

        # Initialize Stacks.
        self.stacks = [[zeros, zeros] for buf in self.bufs]

        # Initialize other.
        self.n_reduces = np.zeros(len(self.bufs), dtype=np.int32)
        self.n_steps = np.zeros(len(self.bufs), dtype=np.int32)

        if not hasattr(example, 'transitions'):
            # TODO: Support no transitions. In the meantime, must at least pass dummy transitions.
            raise ValueError('Transitions must be included.')
        return self.run(example.transitions,
                        run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions)

    def validate(self, transitions, preds, stacks, bufs, zero_padded=True):
        # Note: There is one zero added to bufs, and two zeros added to stacks.
        # Make sure to adjust for this if using lengths of either.
        buf_adjust = 1 if zero_padded else 0
        stack_adjust = 2 if zero_padded else 0

        _transitions = np.array(transitions)
        _preds = preds.copy()
        _invalid = np.zeros(preds.shape, dtype=np.bool)

        cant_skip = _transitions != T_SKIP
        must_skip = _transitions == T_SKIP

        # Fixup predicted skips.
        if len(self.choices) > 2:
            raise NotImplementedError("Can only validate actions for 2 choices right now.")

        buf_lens = [len(buf) - buf_adjust for buf in bufs]
        stack_lens = [len(stack) - stack_adjust for stack in stacks]

        # Cannot reduce on too small a stack
        must_shift = np.array([length < 2 for length in stack_lens])
        check_mask = np.logical_and(cant_skip, must_shift)
        _invalid += np.logical_and(_preds != T_SHIFT, check_mask)
        _preds[must_shift] = T_SHIFT

        # Cannot shift on too small buf
        must_reduce = np.array([length < 1 for length in buf_lens])
        check_mask = np.logical_and(cant_skip, must_reduce)
        _invalid += np.logical_and(_preds != T_REDUCE, check_mask)
        _preds[must_reduce] = T_REDUCE

        # If the given action is skip, then must skip.
        _preds[must_skip] = T_SKIP

        return _preds, _invalid

    def predict_actions(self, transition_output):
        transition_logdist = F.log_softmax(transition_output)
        transition_preds = transition_logdist.data.cpu().numpy().argmax(axis=1)
        return transition_logdist, transition_preds

    def get_transitions_per_example(self, style="preds"):
        if style == "preds":
            source = "t_preds"
        elif style == "given":
            source = "t_given"
        else:
            raise NotImplementedError

        t_preds = np.concatenate([m['t_preds']
                                  for m in self.memories if 't_preds' in m])
        t_preds = torch.from_numpy(t_preds).long()
        t_logprobs = torch.cat(
            [m['t_logprobs'] for m in self.memories if 't_logprobs' in m], 0).data.cpu()
        t_logprobs = torch.cat(
            [t_logprobs, torch.zeros(t_logprobs.size(0), 1)], 1)
        t_strength = torch.gather(t_logprobs, 1, t_preds.view(-1, 1))

        _transitions = [m[source].reshape(
            1, -1) for m in self.memories if m.get(source, None) is not None]
        transitions = np.concatenate(_transitions).T

        t_strength = torch.exp(t_strength.view(
            *list(reversed(transitions.shape))).t())

        skip_mask = (torch.from_numpy(transitions) == T_SKIP).byte()
        t_strength[skip_mask] = 0.

        return transitions, t_strength

    def t_shift(self, buf, stack, tracking, buf_tops, trackings):
        """SHIFT: Should dequeue buffer and item to stack."""
        buf_tops.append(buf.pop() if len(buf) > 0 else self.zeros)
        trackings.append(tracking)

    def t_reduce(self, buf, stack, tracking, lefts, rights, trackings):
        """REDUCE: Should compose top two items of the stack into new item."""

        # The right-most input will be popped first.
        for reduce_inp in [rights, lefts]:
            if len(stack) > 0:
                reduce_inp.append(stack.pop())
            else:
                if self.debug:
                    raise IndexError
                # If we try to Reduce, but there are less than 2 items on the stack,
                # then treat any available item as the right input, and use zeros
                # for any other inputs.
                # NOTE: Only happens on cropped data.
                reduce_inp.append(self.zeros)

        trackings.append(tracking)

    def t_skip(self):
        """SKIP: Acts as padding and is a noop."""

    def shift_phase(self, tops, trackings, stacks):
        """SHIFT: Should dequeue buffer and item to stack."""
        if len(stacks) > 0:
            shift_candidates = iter(tops)
            for stack in stacks:
                new_stack_item = next(shift_candidates)
                stack.append(new_stack_item)

    def reduce_phase(self, lefts, rights, trackings, stacks):
        if len(stacks) > 0:
            reduced = iter(self.reduce(
                lefts, rights))
            for stack in stacks:
                new_stack_item = next(reduced)
                stack.append(new_stack_item)

    def reduce_phase_hook(self, lefts, rights, trackings, reduce_stacks):
        pass

    def loss_phase_hook(self):
        pass

    def evolution_params(self):
        """
        The parameters trained by evolution strategy
        """
        return [(k, v) for k, v in zip(self.transition_net.state_dict(
        ).keys(), self.transition_net.state_dict().values())]

    def run(self, inp_transitions, run_internal_parser=False,
            use_internal_parser=False, validate_transitions=True):
        transition_loss = None
        transition_acc = 0.0
        num_transitions = inp_transitions.shape[1]
        batch_size = inp_transitions.shape[0]
        invalid_count = np.zeros(batch_size)

        # Transition Loop
        # ===============

        for t_step in range(num_transitions):
            transitions = inp_transitions[:, t_step]
            transition_arr = list(transitions)

            # A mask based on SKIP transitions.
            cant_skip = np.array(transitions) != T_SKIP
            must_skip = np.array(transitions) == T_SKIP

            # Memories
            # ========
            # Keep track of key values to determine accuracy and loss.
            self.memory = {}

            # Prepare tracker input.
            if self.debug and any(len(buf) < 1 or len(stack)
                                  for buf, stack in zip(self.bufs, self.stacks)):
                # To elaborate on this exception, when cropping examples it is possible
                # that your first 1 or 2 actions is a reduce action. It is unclear if this
                # is a bug in cropping or a bug in how we think about cropping. In the meantime,
                # turn on the truncate batch flag, and set the eval_seq_length
                # very high.
                raise IndexError(
                    "Warning: You are probably trying to encode examples"
                    "with cropped transitions. Although, this is a reasonable"
                    "feature, when predicting/validating transitions, you"
                    "probably will not get the behavior that you expect. Disable"
                    "this exception if you dare.")
            self.memory['top_buf'] = self.wrap_items(
                [buf[-1] if len(buf) > 0 else self.zeros for buf in self.bufs])
            self.memory['top_stack_1'] = self.wrap_items(
                [stack[-1] if len(stack) > 0 else self.zeros for stack in self.stacks])
            self.memory['top_stack_2'] = self.wrap_items(
                [stack[-2] if len(stack) > 1 else self.zeros for stack in self.stacks])

            # Pre-Action Phase
            # ================

            # TODO: See if PyTorch's 'Advanced Indexing for Tensors and Variables' features would simplify this.

            # For SHIFT
            s_stacks, s_tops, s_trackings, s_idxs = [], [], [], []

            # For REDUCE
            r_stacks, r_lefts, r_rights, r_trackings = [], [], [], []

            batch = zip(transition_arr, self.bufs, self.stacks, itertools.repeat(None))

            for batch_idx, (transition, buf, stack,
                            tracking) in enumerate(batch):
                if transition == T_SHIFT: # shift
                    self.t_shift(buf, stack, tracking, s_tops, s_trackings)
                    s_idxs.append(batch_idx)
                    s_stacks.append(stack)
                elif transition == T_REDUCE: # reduce
                    self.t_reduce(
                        buf,
                        stack,
                        tracking,
                        r_lefts,
                        r_rights,
                        r_trackings)
                    r_stacks.append(stack)
                elif transition == T_SKIP: # skip
                    self.t_skip()

            # Action Phase
            # ============

            self.shift_phase(s_tops, s_trackings, s_stacks)
            self.reduce_phase(r_lefts, r_rights, r_trackings, r_stacks)
            self.reduce_phase_hook(r_lefts, r_rights, r_trackings, r_stacks)

            # Memory Phase
            # ============

            # APPEND ALL MEMORIES. MASK LATER.

            self.memories.append(self.memory)

            # Update number of reduces seen so far.
            self.n_reduces += (np.array(transition_arr) == T_REDUCE)

            # Update number of non-skip actions seen so far.
            self.n_steps += (np.array(transition_arr) != T_SKIP)

        # Loss Phase
        # ==========
        self.loss_phase_hook()

        if self.debug:
            assert all(len(stack) == 3 for stack in self.stacks), \
                "Stacks should be fully reduced and have 3 elements: " \
                "two zeros and the sentence encoding."
            assert all(len(buf) == 1 for buf in self.bufs), \
                "Stacks should be fully shifted and have 1 zero."

        return [stack[-1]
                for stack in self.stacks], transition_acc, transition_loss


class BaseModel(nn.Module):

    optimize_transition_loss = True

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 encode_reverse=None,
                 encode_bidirectional=None,
                 encode_num_layers=None,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_ln=None,
                 classifier_keep_rate=None,
                 context_args=None,
                 composition_args=None,
                 detach=None,
                 evolution=None,
                 **kwargs
                 ):
        super(BaseModel, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature

        self.hidden_dim = hidden_dim = model_dim
        self.wrap_items = composition_args.wrap_items
        self.extract_h = composition_args.extract_h

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim

        classifier_dropout_rate = 1. - classifier_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        # Build parsing component.
        self.lms = self.build_lms(
            composition_args, vocab)

        # Build classiifer.
        features_dim = self.get_features_dim()
        self.mlp = MLP(features_dim, mlp_dim, num_classes,
                        num_mlp_layers, mlp_ln, classifier_dropout_rate)

        self.embedding_dropout_rate = 1. - embedding_keep_rate

        # Create dynamic embedding layer.
        self.embed = Embed(
            word_embedding_dim,
            vocab.size,
            vectors=vocab.vectors)

        self.input_dim = context_args.input_dim

        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

        self.inverted_vocabulary = None

        # Create Lift layer
        self.lift = Lift(vocab.vectors.shape[1], model_dim * model_dim)

    def get_features_dim(self):
        features_dim = (self.hidden_dim * self.hidden_dim * 2) if self.use_sentence_pair else (self.hidden_dim * self.hidden_dim)
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += (self.hidden_dim * self.hidden_dim)
            if self.use_product_feature:
                features_dim += (self.hidden_dim * self.hidden_dim)
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
            features = h[0]
        return features

    def build_lms(self, args, vocab):
        return LMS(args, vocab)

    def run_lms(
            self,
            example,
            use_internal_parser,
            validate_transitions=True):
        self.lms.reset_state()
        h_list, transition_acc, transition_loss = self.lms(
            example, use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)
        h = self.wrap(h_list)
        return h, transition_acc, transition_loss

    def forward_hook(self, embeds, batch_size, seq_length):
        pass

    def output_hook(self, output, sentences, transitions, y_batch=None, embeds=None):
        pass

    def forward(
            self,
            sentences,
            transitions,
            y_batch=None,
            use_internal_parser=False,
            validate_transitions=True,
            **kwargs):
        example = self.unwrap(sentences, transitions)

        b, l = example.tokens.size()[:2]

        embeds = self.embed(example.tokens)
        embeds = self.reshape_input(embeds, b, l)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, b, l)
        self.forward_hook(embeds, b, l)
        embeds = F.dropout(
            embeds,
            self.embedding_dropout_rate,
            training=self.training)

        embeds = self.lift(embeds)

        # Make Buffers
        ee = torch.chunk(embeds, b * l, 0)[::-1]
        bb = []
        for ii in range(b):
            ex = list(ee[ii * l:(ii + 1) * l])
            bb.append(ex)
        buffers = bb[::-1]

        example.bufs = buffers

        h, transition_acc, transition_loss = self.run_lms(
            example, use_internal_parser, validate_transitions)

        self.spinn_outp = h

        self.transition_acc = transition_acc
        self.transition_loss = transition_loss

        # Build features
        features = self.build_features(h)

        output = self.mlp(features)

        self.output_hook(output, sentences, transitions, y_batch, embeds)

        return output

    # --- Sentence Style Switches ---

    def unwrap(self, sentences, transitions):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, transitions)
        return self.unwrap_sentence(sentences, transitions)

    def wrap(self, h_list):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(h_list)
        return self.wrap_sentence(h_list)

    # --- Sentence Model Specific ---

    def unwrap_sentence(self, sentences, transitions):
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

    def wrap_sentence(self, items):
        h = self.extract_h(self.wrap_items(items))
        return [h]

    # --- Sentence Pair Model Specific ---

    def unwrap_sentence_pair(self, sentences, transitions):
        # Build Tokens
        x_prem = sentences[:, :, 0]
        x_hyp = sentences[:, :, 1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:, :, 0]
        t_hyp = transitions[:, :, 1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        example = Example()
        example.tokens = to_gpu(
            Variable(
                torch.from_numpy(x),
                volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence_pair(self, items):
        batch_size = len(items) / 2
        h_premise = self.extract_h(self.wrap_items(items[:batch_size]))
        h_hypothesis = self.extract_h(self.wrap_items(items[batch_size:]))
        return [h_premise, h_hypothesis]

    def get_samples(self, x, vocabulary, only_one=False):
        # n=-1: Show all samples.
        if not self.inverted_vocabulary:
            self.inverted_vocabulary = dict(
                [(vocabulary[key], key) for key in vocabulary])

        transitions, _ = self.spinn.get_transitions_per_example()

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

