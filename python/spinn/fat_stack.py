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
from spinn.util.blocks import LSTMState, Embed
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

    def __call__(self, bufs, stacks):
        self.batch_size = len(bufs)
        zeros = np.zeros(bufs[0][0].size(), dtype=np.float32)
        zeros = to_gpu(Variable(torch.from_numpy(zeros), volatile=bufs[0][0].volatile))
        buf = bundle(buf[-1] for buf in bufs)
        stack1 = bundle(stack[-1] if len(stack) > 0 else zeros for stack in stacks)
        stack2 = bundle(stack[-2] if len(stack) > 1 else zeros for stack in stacks)

        lstm_in = self.buf(buf.h)
        lstm_in += self.stack1(stack1.h)
        lstm_in += self.stack2(stack2.h)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            self.c = to_gpu(Variable(torch.from_numpy(
                np.zeros((self.batch_size, self.state_size),
                              dtype=np.float32)),
                volatile=zeros.volatile))

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
        self.embed = Embed(args.size, vocab.size, args.input_dropout_rate,
                        vectors=vocab.vectors,
                        use_input_dropout=args.use_input_dropout,
                        use_input_norm=args.use_input_norm,
                        )
        self.reduce = Reduce(args.size, args.tracker_size)
        if args.tracker_size is not None:
            self.tracker = Tracker(
                args.size, args.tracker_size,
                use_tracker_dropout=args.use_tracker_dropout,
                tracker_dropout_rate=args.tracker_dropout_rate)
            if args.transition_weight is not None:
                # TODO: Might be interesting to try a different network here.
                self.transition_net = nn.Linear(args.tracker_size, 3 if use_skips else 2)
        self.transition_weight = args.transition_weight
        self.use_reinforce = use_reinforce
        self.use_skips = use_skips
        choices = [T_SHIFT, T_REDUCE, T_SKIP] if use_skips else [T_SHIFT, T_REDUCE]
        self.choices = np.array(choices, dtype=np.int32)

    def reset_state(self):
        self.memories = []

    def __call__(self, example, use_internal_parser=False, validate_transitions=True):
        self.bufs = self.embed(example.tokens)
        self.stacks = [[] for buf in self.bufs]
        self.buffers_t = [0 for buf in self.bufs]
        # There are 2 * N - 1 transitons, so (|transitions| + 1) / 2 should equal N.
        self.buffers_n = [(len([t for t in ts if t != T_SKIP]) + 1) / 2 for ts in example.transitions]
        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if not hasattr(example, 'transitions'):
            # TODO: Support no transitions. In the meantime, must at least pass dummy transitions.
            raise ValueError('Transitions must be included.')
        return self.run(example.transitions,
                        run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions)

    def validate(self, transitions, preds, stacks, buffers_t, buffers_n):
        preds = preds.cpu().numpy()

        DEFAULT_CHOICE = T_SHIFT
        cant_skip = np.array([p == T_SKIP and t != T_SKIP for t, p in zip(transitions, preds)])
        preds[cant_skip] = DEFAULT_CHOICE

        # Cannot reduce on too small a stack
        must_shift = np.array([len(stack) < 2 for stack in stacks])
        preds[must_shift] = T_SHIFT

        # Cannot shift if stack has to be reduced
        must_reduce = np.array([buf_t >= buf_n for buf_t, buf_n in zip(buffers_t, buffers_n)])
        preds[must_reduce] = T_REDUCE

        must_skip = np.array([t == T_SKIP for t in transitions])
        preds[must_skip] = T_SKIP

        return preds

    def run(self, inp_transitions, run_internal_parser=False, use_internal_parser=False, validate_transitions=True):
        transition_loss, transition_acc = 0, 0
        num_transitions = inp_transitions.shape[1]

        # Transition Loop
        # ===============

        for i in range(num_transitions):
            transitions = inp_transitions[:, i]
            transition_arr = list(transitions)

            # A mask to select all non-SKIP transitions.
            cant_skip = np.array([t != T_SKIP for t in transitions])

            # Run if:
            # A. We have a tracking component and,
            # B. There is at least one transition that will not be skipped.
            if hasattr(self, 'tracker') and (self.use_skips or sum(cant_skip) > 0):

                # Get hidden output from the tracker. Used to predict transitions.
                tracker_c, tracker_h = self.tracker(self.bufs, self.stacks)

                if hasattr(self, 'transition_net'):
                    transition_output = self.transition_net(tracker_h)

                if hasattr(self, 'transition_net') and run_internal_parser:

                    # Predict Actions
                    # ===============

                    t_logits = F.log_softmax(transition_output)
                    t_given = transitions

                    if self.use_reinforce:
                        transition_dist = F.softmax(transition_output)
                        sampled_transitions = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
                        sampled_transitions[cant_skip] = [np.random.choice(self.choices, 1, p=t_dist)[0] for t_dist in transition_dist.data[cant_skip]]
                        transition_preds = sampled_transitions
                    else:
                        transition_preds = transition_output.data.max(1)[1].view(-1)

                    # Constrain to valid actions
                    # ==========================

                    if validate_transitions:
                        transition_preds = self.validate(transition_arr, transition_preds,
                            self.stacks, self.buffers_t, self.buffers_n)

                    t_preds = transition_preds

                    # Filter to non-SKIP values
                    # =========================

                    if not self.use_skips:
                        t_preds = t_preds[cant_skip]
                        t_given = t_given[cant_skip]

                        # Be careful when filtering distributions. These values are used to
                        # calculate loss and need to be used in backprop.
                        t_logits = torch.chunk(transition_output, transition_output.size()[0], 0)
                        t_logits = torch.cat([t_logits[i] for i, y in enumerate(cant_skip) if y], 0)

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

            for ii, (transition, buf, stack, tracking) in enumerate(batch):
                if transition == T_SHIFT: # shift
                    stack.append(buf.pop())
                    self.buffers_t[ii] += 1
                elif transition == T_REDUCE: # reduce
                    for lr in [rights, lefts]:
                        if len(stack) > 0:
                            lr.append(stack.pop())
                        else:
                            # NOTE: Only happens on cropped data.
                            zeros = to_gpu(Variable(
                                torch.from_numpy(np.zeros(buf[0].size(), dtype=np.float32)),
                                volatile=buf[0].volatile))
                            lr.append(zeros)
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

        return [stack[-1] for stack in self.stacks], transition_loss


class BaseModel(nn.Module):

    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 initial_embeddings, num_classes, mlp_dim,
                 embedding_keep_rate, classifier_keep_rate,
                 use_tracker_dropout=True, tracker_dropout_rate=0.1,
                 use_input_dropout=False, use_input_norm=False,
                 use_classifier_norm=True,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 use_tracking_lstm=True,
                 use_shift_composition=True,
                 use_reinforce=False,
                 use_skips=False,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
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

        self.l0 = nn.Linear(features_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)

        self.initial_embeddings = initial_embeddings
        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.use_classifier_norm = use_classifier_norm
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        self.use_reinforce = use_reinforce

        args = Args()
        args.size = model_dim/2
        args.tracker_size = tracking_lstm_hidden_dim if use_tracking_lstm else None
        args.transition_weight = transition_weight
        args.input_dropout_rate = 1. - embedding_keep_rate
        args.use_input_dropout = use_input_dropout
        args.use_input_norm = use_input_norm
        args.use_tracker_dropout = use_tracker_dropout
        args.tracker_dropout_rate = tracker_dropout_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.spinn = SPINN(args, vocab, use_reinforce=use_reinforce, use_skips=use_skips)

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

    def run_mlp(self, h, train):
        # Pass through MLP Classifier.
        if self.use_sentence_pair:
            h_prem, h_hyp = h
            features = [h_prem, h_hyp]
            if self.use_difference_feature:
                features.append(h_prem - h_hyp)
            if self.use_product_feature:
                features.append(h_prem * h_hyp)
            h = torch.cat(features, 1)
        else:
            h = h[0]

        h = to_gpu(h)
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = h

        return y

    def __call__(self, sentences, transitions, y_batch=None, train=True,
                 use_internal_parser=False, validate_transitions=True):
        example = self.build_example(sentences, transitions, train)
        h, transition_acc, transition_loss = self.run_spinn(example, train, use_internal_parser, validate_transitions)
        y = self.run_mlp(h, train)

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
