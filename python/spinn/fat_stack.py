import itertools

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


T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1


class SentencePairTrainer(BaseSentencePairTrainer): pass


class SentenceTrainer(SentencePairTrainer): pass


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, use_tracker_dropout=True, tracker_dropout_rate=0.1):
        super(Tracker, self).__init__()

        # Initialize layers.
        self.lateral = nn.Linear(tracker_size, 4 * tracker_size)
        self.buf = nn.Linear(size, 4 * tracker_size, bias=False)
        self.stack1 = nn.Linear(size, 4 * tracker_size, bias=False)
        self.stack2 = nn.Linear(size, 4 * tracker_size, bias=False)

        self.state_size = tracker_size
        self.tracker_dropout_rate = tracker_dropout_rate
        self.use_tracker_dropout = use_tracker_dropout

        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def forward(self, top_buf, top_stack_1, top_stack_2):
        lstm_in = self.buf(top_buf.h)
        lstm_in += self.stack1(top_stack_1.h)
        lstm_in += self.stack2(top_stack_2.h)

        batch_size = lstm_in.size(0)

        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            self.c = to_gpu(Variable(torch.from_numpy(
                np.zeros((batch_size, self.state_size),
                              dtype=np.float32)),
                volatile=lstm_in.volatile))

        if self.use_tracker_dropout:
            lstm_in = F.dropout(lstm_in, self.tracker_dropout_rate, train=lstm_in.volatile == False)

        # Run tracking lstm.
        self.c, self.h = lstm(self.c, lstm_in)

        return self.c, self.h

    @property
    def states(self):
        return unbundle((self.c, self.h))

    @states.setter
    def states(self, state_iter):
        if state_iter is not None:
            state = bundle(state_iter)
            self.c, self.h = state.c, state.h


class SPINN(nn.Module):

    def __init__(self, args, vocab, use_reinforce=True, use_skips=False):
        super(SPINN, self).__init__()

        # Optional debug mode.
        self.debug = False

        self.transition_weight = args.transition_weight
        self.use_reinforce = use_reinforce
        self.use_skips = use_skips

        # Create dynamic embedding layer.
        self.embed = Embed(args.size, vocab.size,
                        embedding_dropout_rate=args.input_dropout_rate,
                        vectors=vocab.vectors,
                        )

        # Reduce function for semantic composition.
        self.reduce = Reduce(args.size, args.tracker_size)
        if args.tracker_size is not None:
            self.tracker = Tracker(
                args.size, args.tracker_size,
                use_tracker_dropout=args.use_tracker_dropout,
                tracker_dropout_rate=args.tracker_dropout_rate)
            if args.transition_weight is not None:
                # TODO: Might be interesting to try a different network here.
                self.transition_net = nn.Linear(args.tracker_size, 3 if use_skips else 2)

        # Predict 2 or 3 actions depending on whether SKIPs will be predicted.
        choices = [T_SHIFT, T_REDUCE, T_SKIP] if use_skips else [T_SHIFT, T_REDUCE]
        self.choices = np.array(choices, dtype=np.int32)

    def reset_state(self):
        self.memories = []

    def forward(self, example, use_internal_parser=False, validate_transitions=True):
        self.buffers_n = (example.tokens.data != 0).long().sum(1).view(-1).tolist()

        if self.debug:
            seq_length = example.tokens.size(1)
            assert all(buf_n <= (seq_length + 1) // 2 for buf_n in self.buffers_n), \
                "All sentences (including cropped) must be the appropriate length."

        self.bufs = self.embed(example.tokens)

        # Notes on adding zeros to bufs/stacks.
        # - After the buffer is consumed, we need one zero on the buffer
        #   used as input to the tracker.
        # - For the first two steps, the stack would be empty, but we add
        #   zeros so that the tracker still gets input.
        zeros = to_gpu(Variable(torch.from_numpy(
            np.zeros(self.bufs[0][0].size(), dtype=np.float32)),
            volatile=self.bufs[0][0].volatile))

        # Trim unused tokens.
        self.bufs = [[zeros] + b[-b_n:] for b, b_n in zip(self.bufs, self.buffers_n)]

        self.stacks = [[zeros, zeros] for buf in self.bufs]

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
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

        # Fixup predicted skips.
        if len(self.choices) > 2:
            raise NotImplementedError("Can only validate actions for 2 choices right now.")

        buf_lens = [len(buf) - buf_adjust for buf in bufs]
        stack_lens = [len(stack) - stack_adjust for stack in stacks]

        # Cannot reduce on too small a stack
        must_shift = np.array([length < 2 for length in stack_lens])
        preds[must_shift] = T_SHIFT

        # Cannot shift on too small buf
        must_reduce = np.array([length < 1 for length in buf_lens])
        preds[must_reduce] = T_REDUCE

        # If the given action is skip, then must skip.
        preds[_transitions == T_SKIP] = T_SKIP

        return preds

    def predict_actions(self, transition_output):
        if self.use_reinforce:
            transition_dist = F.softmax(transition_output)
            sampled_transitions = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
            sampled_transitions[cant_skip] = [np.random.choice(self.choices, 1, p=t_dist)[0] for t_dist in transition_dist.data[cant_skip]]
            transition_preds = sampled_transitions
        else:
            transition_preds = transition_output.data.max(1)[1].view(-1)
        return transition_preds


    def run(self, inp_transitions, run_internal_parser=False, use_internal_parser=False, validate_transitions=True):
        transition_loss, transition_acc = 0, 0
        num_transitions = inp_transitions.shape[1]

        # Transition Loop
        # ===============

        for t_step in range(num_transitions):
            transitions = inp_transitions[:, t_step]
            transition_arr = list(transitions)

            # A mask to select all non-SKIP transitions.
            cant_skip = np.array([t != T_SKIP for t in transitions])

            # Run if:
            # A. We have a tracking component and,
            # B. There is at least one transition that will not be skipped.
            if hasattr(self, 'tracker') and (self.use_skips or sum(cant_skip) > 0):

                # Prepare tracker input.
                try:
                    top_buf = bundle(buf[-1] for buf in self.bufs)
                    top_stack_1 = bundle(stack[-1] for stack in self.stacks)
                    top_stack_2 = bundle(stack[-2] for stack in self.stacks)
                except:
                    # To elaborate on this exception, when cropping examples it is possible
                    # that your first 1 or 2 actions is a reduce action. It is unclear if this
                    # is a bug in cropping or a bug in how we think about cropping. In the meantime,
                    # turn on the truncate batch flag, and set the eval_seq_length very high.
                    raise NotImplementedError("Warning: You are probably trying to encode examples"
                          "with cropped transitions. Although, this is a reasonable"
                          "feature, when predicting/validating transitions, you"
                          "probably will not get the behavior that you expect. Disable"
                          "this exception if you dare.")
                    # Uncomment to handle weirdly placed actions like discussed in the above exception.
                    # =========
                    # zeros = to_gpu(Variable(torch.from_numpy(
                    #     np.zeros(self.bufs[0][0].size(), dtype=np.float32)),
                    #     volatile=self.bufs[0][0].volatile))
                    # top_buf = bundle(buf[-1] for buf in self.bufs)
                    # top_stack_1 = bundle(stack[-1] if len(stack) > 0 else zeros for stack in self.stacks)
                    # top_stack_2 = bundle(stack[-2] if len(stack) > 1 else zeros for stack in self.stacks)

                # Get hidden output from the tracker. Used to predict transitions.
                tracker_c, tracker_h = self.tracker(top_buf, top_stack_1, top_stack_2)

                if hasattr(self, 'transition_net'):
                    transition_output = self.transition_net(tracker_h)

                if hasattr(self, 'transition_net') and run_internal_parser:

                    # Predict Actions
                    # ===============

                    t_logits = F.log_softmax(transition_output)
                    t_given = transitions
                    # TODO: Mask before predicting. This should simplify things and reduce computation.
                    # The downside is that in the Action Phase, need to be smarter about which stacks/bufs
                    # are selected.
                    transition_preds = self.predict_actions(transition_output)

                    # Constrain to valid actions
                    # ==========================

                    if validate_transitions:
                        transition_preds = self.validate(transition_arr, transition_preds.cpu().numpy(), self.stacks, self.bufs)

                    t_preds = transition_preds

                    # Filter to non-SKIP values
                    # =========================

                    if not self.use_skips:
                        t_preds = t_preds[cant_skip]
                        t_given = t_given[cant_skip]

                        # Be careful when filtering distributions. These values are used to
                        # calculate loss and need to be used in backprop.
                        index = (cant_skip * np.arange(cant_skip.shape[0]))[cant_skip]
                        index = to_gpu(Variable(torch.from_numpy(index).long(), volatile=t_logits.volatile))
                        t_logits = torch.index_select(t_logits, 0, index)


                    # Memories
                    # ========
                    # Keep track of key values to determine accuracy and loss.
                    # (optional) Filter to only non-skipped transitions. When filtering values
                    # that will be backpropagated over, be careful that gradient flow isn't broken.

                    memory = {}

                    # Actual transition predictions. Used to measure transition accuracy.
                    memory["t_preds"] = t_preds

                    # Distribution of transitions use to calculate transition loss.
                    memory["t_logits"] = t_logits

                    # Given transitions.
                    memory["t_given"] = t_given

                    # TODO: Write tests to make sure these values look right in the various settings.

                    # If this FLAG is set, then use the predicted actions rather than the given.
                    if use_internal_parser:
                        transition_arr = transition_preds.tolist()

                    self.memories.append(memory)

            # Action Phase
            # ============

            lefts, rights, trackings = [], [], []
            batch = zip(transition_arr, self.bufs, self.stacks,
                        self.tracker.states if hasattr(self, 'tracker') and self.tracker.h is not None
                        else itertools.repeat(None))

            for batch_idx, (transition, buf, stack, tracking) in enumerate(batch):
                if transition == T_SHIFT: # shift
                    stack.append(buf.pop())
                elif transition == T_REDUCE: # reduce
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
                            zeros = to_gpu(Variable(
                                torch.from_numpy(np.zeros(buf[0].size(), dtype=np.float32)),
                                volatile=buf[0].volatile))
                            reduce_inp.append(zeros)

                    # The tracking output is used in the Reduce function.
                    trackings.append(tracking)

            # Reduce Phase
            # ============

            if len(rights) > 0:
                reduced = iter(self.reduce(
                    lefts, rights, trackings))
                for transition, stack, in zip(
                        transition_arr, self.stacks):
                    if transition == T_REDUCE: # reduce
                        new_stack_item = next(reduced)
                        stack.append(new_stack_item)

        # Loss Phase
        # ==========

        if self.transition_weight is not None:
            # We compute statistics after the fact, since sub-batches can
            # have different sizes when not using skips.
            statistics = zip(*[
                (m["t_preds"], m["t_logits"], m["t_given"])
                for m in self.memories])

            statistics = [
                torch.squeeze(torch.cat([ss.unsqueeze(1) for ss in s], 0))
                if isinstance(s[0], Variable) else
                np.array(reduce(lambda x, y: x + y.tolist(), s, []))
                for s in statistics]

            t_preds, t_logits, t_given = statistics

            self.transition_acc = (t_preds == t_given).sum() / float(t_preds.shape[0])
            transition_loss = nn.NLLLoss()(t_logits, to_gpu(Variable(
                torch.from_numpy(t_given), volatile=t_logits.volatile)))

            transition_loss *= self.transition_weight
            self.transition_loss = transition_loss
        else:
            transition_loss = None

        if self.debug:
            assert all(len(stack) == 3 for stack in self.stacks), \
                "Stacks should be fully reduced and have 3 elements: " \
                "two zeros and the sentence encoding."

        return [stack[-1] for stack in self.stacks], transition_loss


class BaseModel(nn.Module):

    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 initial_embeddings, num_classes, mlp_dim,
                 embedding_keep_rate, classifier_keep_rate,
                 use_tracker_dropout=True, tracker_dropout_rate=0.1,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 use_tracking_lstm=True,
                 use_shift_composition=True,
                 use_reinforce=False,
                 use_skips=False,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 num_mlp_layers=None,
                 mlp_bn=None,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature

        self.hidden_dim = hidden_dim = model_dim / 2
        features_dim = hidden_dim * 2 if use_sentence_pair else hidden_dim

        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += self.hidden_dim
            if self.use_product_feature:
                features_dim += self.hidden_dim

        mlp_input_dim = features_dim

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        self.use_reinforce = use_reinforce
        classifier_dropout_rate = 1. - classifier_keep_rate

        args = Args()
        args.size = model_dim/2
        args.tracker_size = tracking_lstm_hidden_dim if use_tracking_lstm else None
        args.transition_weight = transition_weight
        args.input_dropout_rate = 1. - embedding_keep_rate
        args.use_tracker_dropout = use_tracker_dropout
        args.tracker_dropout_rate = tracker_dropout_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.spinn = SPINN(args, vocab, use_reinforce=use_reinforce, use_skips=use_skips)

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
            num_mlp_layers, mlp_bn, classifier_dropout_rate)

    def build_example(self, sentences, transitions, train):
        raise Exception('Not implemented.')

    def run_spinn(self, example, train, use_internal_parser, validate_transitions=True):
        self.spinn.reset_state()
        state, _ = self.spinn(example,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions)

        transition_acc = self.spinn.transition_acc if hasattr(self.spinn, 'transition_acc') else 0.0
        transition_loss = self.spinn.transition_loss if hasattr(self.spinn, 'transition_loss') else None
        return state, transition_acc, transition_loss

    def forward(self, sentences, transitions, y_batch=None, train=True,
                 use_internal_parser=False, validate_transitions=True):
        example = self.build_example(sentences, transitions, train)
        h, transition_acc, transition_loss = self.run_spinn(example, train, use_internal_parser, validate_transitions)

        # Build features
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

        y = self.mlp(features, train)

        # Calculate Loss & Accuracy.
        logits = F.log_softmax(y)
        target = to_gpu(Variable(torch.from_numpy(y_batch).long(), volatile=not train))
        accum_loss = nn.NLLLoss()(logits, target)

        preds = logits.data.max(1)[1]
        self.accuracy = preds.eq(target.data).sum() / float(preds.size(0))

        return logits, accum_loss, self.accuracy, transition_acc, transition_loss


class SentencePairModel(BaseModel):

    def build_example(self, sentences, transitions, train):
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
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not train))
        example.transitions = t

        return example

    def run_spinn(self, example, train, use_internal_parser=False, validate_transitions=True):
        state_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            example, train, use_internal_parser, validate_transitions)
        batch_size = len(state_both) / 2
        h_premise = get_h(torch.cat(state_both[:batch_size], 0), self.hidden_dim)
        h_hypothesis = get_h(torch.cat(state_both[batch_size:], 0), self.hidden_dim)
        return [h_premise, h_hypothesis], transition_acc, transition_loss


class SentenceModel(BaseModel):

    def build_example(self, sentences, transitions, train):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not train))
        example.transitions = t

        return example

    def run_spinn(self, example, train, use_internal_parser=False, validate_transitions=True):
        state, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            example, train, use_internal_parser, validate_transitions)
        h = get_h(torch.cat(state, 0), self.hidden_dim)
        return [h], transition_acc, transition_loss
