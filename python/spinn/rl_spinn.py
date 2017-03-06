import itertools
import copy

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

from spinn.fat_stack import BaseModel, SentenceModel, SentencePairModel
from spinn.fat_stack import SPINN


import spinn.cbow


T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1


class SentencePairTrainer(BaseSentencePairTrainer): pass


class SentenceTrainer(SentencePairTrainer): pass


class RLSPINN(SPINN):
    def predict_actions(self, transition_output, cant_skip):
        if self.training:
            transition_dist = F.softmax(transition_output)
            transition_dist = transition_dist.data.cpu().numpy()
            sampled_transitions = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
            sampled_transitions[cant_skip] = [np.random.choice(self.choices, 1, p=t_dist)[0] for t_dist in transition_dist[cant_skip]]
            transition_preds = sampled_transitions
        else:
            transition_dist = F.log_softmax(transition_output)
            transition_dist = transition_dist.data.cpu().numpy()
            transition_preds = transition_dist.argmax(axis=1)
        return transition_preds


class RLBaseModel(BaseModel):

    optimize_transition_loss = False

    def __init__(self,
                 rl_mu=None,
                 rl_baseline=None,
                 rl_reward=None,
                 rl_weight=None,
                 rl_whiten=None,
                 rl_entropy=None,
                 rl_entropy_beta=None,
                 **kwargs):
        super(RLBaseModel, self).__init__(**kwargs)

        self.kwargs = kwargs

        self.rl_mu = rl_mu
        self.rl_baseline = rl_baseline
        self.rl_reward = rl_reward
        self.rl_weight = rl_weight
        self.rl_whiten = rl_whiten
        self.rl_entropy = rl_entropy
        self.rl_entropy_beta = rl_entropy_beta

        self.register_buffer('baseline', torch.FloatTensor([0.0]))

        if self.rl_baseline == "value":
            if kwargs['use_sentence_pair']:
                value_net_cls = spinn.cbow.SentencePairModel
            else:
                value_net_cls = spinn.cbow.SentenceModel
            self.value_net = value_net_cls(
                model_dim=kwargs['model_dim'],
                word_embedding_dim=kwargs['word_embedding_dim'],
                vocab_size=kwargs['vocab_size'],
                initial_embeddings=kwargs['initial_embeddings'],
                mlp_dim=kwargs['mlp_dim'],
                num_mlp_layers=0,
                embedding_keep_rate=kwargs['embedding_keep_rate'],
                classifier_keep_rate=kwargs['classifier_keep_rate'],
                use_sentence_pair=kwargs['use_sentence_pair'],
                num_classes=1,
                use_embed=False,
                )

    def build_spinn(self, *args, **kwargs):
        return RLSPINN(*args, **kwargs)

    def run_greedy(self, sentences, transitions):
        if self.use_sentence_pair:
            inference_model_cls = SentencePairModel
        else:
            inference_model_cls = SentenceModel

        # HACK: This is a pretty simple way to create the inference time version of SPINN.
        # The reason a copy is necessary is because there is some retained state in the
        # memories and loss variables that break deep copy.
        inference_model = inference_model_cls(**self.kwargs)
        inference_model.load_state_dict(copy.deepcopy(self.state_dict()))
        inference_model.eval()

        outputs = inference_model(sentences, transitions,
            use_internal_parser=True,
            validate_transitions=True)

        return outputs

    def build_reward(self, logits, target):
        if self.rl_reward == "standard": # Zero One Loss.
            rewards = torch.eq(logits.max(1)[1], target).float()
        elif self.rl_reward == "xent": # Cross Entropy Loss.
            rewards = torch.cat([F.nll_loss(Variable(ll), Variable(t))
                for ll, t in zip(logits, target.chunk(target.size(0)))], 0).unsqueeze(1).data
        else:
            raise NotImplementedError

        return rewards

    def build_baseline(self, output, rewards, sentences, transitions, y_batch=None, embeds=None):
        if self.rl_baseline == "ema":
            mu = self.rl_mu
            self.baseline[0] = self.baseline[0] * (1 - mu) + rewards.mean() * mu
            baseline = self.baseline[0]
        elif self.rl_baseline == "value":
            # Pass inputs to Value Net
            if embeds is not None:
                value_inp = torch.cat([torch.cat(e, 0).view(1,len(e),-1) for e in embeds], 0)
                value_outp = self.value_net(value_inp, transitions)
            else:
                value_outp = self.value_net(sentences, transitions)

            # Estimate Reward
            value_prob = value_outp

            # Save MSE Loss using Reward as target
            self.value_loss = nn.MSELoss()(value_prob, to_gpu(Variable(rewards, volatile=not self.training)))

            baseline = value_prob.data.cpu()
        elif self.rl_baseline == "greedy":
            # Pass inputs to Greedy Max
            greedy_outp = self.run_greedy(sentences, transitions)

            # Estimate Reward
            logits = F.softmax(output).data.cpu()
            target = torch.from_numpy(y_batch).long()
            greedy_rewards = self.build_reward(logits, target)

            baseline = greedy_rewards
        else:
            raise NotImplementedError

        return baseline

    def reinforce(self, advantage):
        t_preds, t_logits, t_given, t_mask = self.spinn.get_statistics()

        # TODO: Many of these ops are on the cpu. Might be worth shifting to GPU.
        if self.use_sentence_pair:
            # Handles the case of SNLI where each reward is used for two sentences.
            advantage = torch.cat([advantage, advantage], 0)

        # Expand advantage.
        if not self.spinn.use_skips:
            advantage = advantage.index_select(0, torch.from_numpy(t_mask).long())
        else:
            raise NotImplementedError

        actions = torch.from_numpy(t_preds).long().view(-1, 1)
        action_mask = torch.zeros(t_logits.size()).scatter_(1, actions, 1.0)
        action_mask = to_gpu(Variable(action_mask, volatile=not self.training))
        log_p_action = torch.sum(t_logits * action_mask, 1)

        # source: https://github.com/miyosuda/async_deep_reinforce/issues/1
        if self.rl_entropy:
            # TODO: Taking exp of a log is not the best way to get the initial probability...
            entropy = -1. * (t_logits * torch.exp(t_logits)).sum(1)
            self.avg_entropy = entropy.sum().data[0] / float(entropy.size(0))
        else:
            entropy = 0.0

        policy_loss = -1. * torch.sum(log_p_action * to_gpu(Variable(advantage, volatile=log_p_action.volatile)) + entropy * self.rl_entropy_beta)
        policy_loss /= log_p_action.size(0)
        policy_loss *= self.rl_weight

        return policy_loss

    def output_hook(self, output, sentences, transitions, y_batch=None, embeds=None):
        if not self.training:
            return

        logits = F.softmax(output).data.cpu()
        target = torch.from_numpy(y_batch).long()

        # Get Reward.
        rewards = self.build_reward(logits, target)

        # Get Baseline.
        baseline = self.build_baseline(output, rewards, sentences, transitions, y_batch, embeds)

        # Calculate advantage.
        advantage = rewards - baseline

        # Whiten advantage. Note: Might only make sense when using value net.
        # source: https://rllab.readthedocs.io/en/latest/user/implement_algo_basic.html#normalizing-the-returns
        if self.rl_whiten:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Assign REINFORCE output.
        self.policy_loss = self.reinforce(advantage)

        # Cache values for logging.
        self.norm_rewards = rewards.norm()
        self.norm_baseline = baseline.norm() if hasattr(baseline, 'norm') else baseline
        self.norm_advantage = advantage.norm()


class SentencePairModel(RLBaseModel):

    def build_example(self, sentences, transitions):
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
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def run_spinn(self, example, use_internal_parser=False, validate_transitions=True):
        state_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            example, use_internal_parser, validate_transitions)
        batch_size = len(state_both) / 2
        h_premise = get_h(torch.cat(state_both[:batch_size], 0), self.hidden_dim)
        h_hypothesis = get_h(torch.cat(state_both[batch_size:], 0), self.hidden_dim)
        return [h_premise, h_hypothesis], transition_acc, transition_loss


class SentenceModel(RLBaseModel):

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def run_spinn(self, example, use_internal_parser=False, validate_transitions=True):
        state, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            example, use_internal_parser, validate_transitions)
        h = get_h(torch.cat(state, 0), self.hidden_dim)
        return [h], transition_acc, transition_loss
