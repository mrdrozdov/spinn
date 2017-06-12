"""
logging.py

Log format convenience methods for training spinn.

"""

import numpy as np
from spinn.util.blocks import flatten
from spinn.util.misc import time_per_token
from spinn.data import T_SHIFT, T_REDUCE, T_SKIP


class InspectModel(object):
    '''Examines what kind of SPINN model we are dealing with.'''

    def __init__(self, model):
        self.has_spinn = hasattr(model, 'spinn')
        self.has_transition_loss = hasattr(
            model, 'transition_loss') and model.transition_loss is not None
        self.has_invalid = self.has_spinn and hasattr(model.spinn, 'invalid')
        self.has_policy = self.has_spinn and hasattr(model, 'policy_loss')
        self.has_value = self.has_spinn and hasattr(model, 'value_loss')
        self.has_epsilon = self.has_spinn and hasattr(model.spinn, "epsilon")
        self.has_temperature = self.has_spinn and hasattr(
            model.spinn, "temperature")


def inspect(model):
    return InspectModel(model)


def train_accumulate(model, data_manager, A, batch):

    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch
    im = inspect(model)

    # Accumulate stats for transition accuracy.
    if im.has_transition_loss:
        preds = [m["t_preds"]
                 for m in model.spinn.memories if m.get('t_preds', None) is not None]
        truth = [m["t_given"]
                 for m in model.spinn.memories if m.get('t_given', None) is not None]
        A.add('preds', preds)
        A.add('truth', truth)

    if im.has_invalid:
        A.add('invalid', model.spinn.invalid)


def train_rl_accumulate(model, data_manager, A, batch):

    im = inspect(model)

    if im.has_policy:
        A.add('policy_cost', model.policy_loss.data[0])

    if im.has_value:
        A.add('value_cost', model.value_loss.data[0])

    A.add('adv_mean', model.stats['mean'])
    A.add('adv_mean_magnitude', model.stats['mean_magnitude'])
    A.add('adv_var', model.stats['var'])
    A.add('adv_var_magnitude', model.stats['var_magnitude'])


def stats(model, optimizer, A, step, log_entry):
    im = inspect(model)

    if im.has_transition_loss:
        all_preds = np.array(flatten(A.get('preds')))
        all_truth = np.array(flatten(A.get('truth')))
        avg_trans_acc = (all_preds == all_truth).sum() / \
            float(all_truth.shape[0])

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))

    log_entry.step = step
    log_entry.class_accuracy = A.get_avg('class_acc')
    log_entry.cross_entropy_cost = A.get_avg('xent_cost')  # not actual mean
    log_entry.l2_cost = A.get_avg('l2_cost')  # not actual mean
    log_entry.learning_rate = optimizer.lr
    log_entry.time_per_token_seconds = time_metric

    total_cost = log_entry.l2_cost + log_entry.cross_entropy_cost
    if im.has_transition_loss:
        log_entry.transition_accuracy = avg_trans_acc
        log_entry.transition_cost = model.transition_loss.data[0]
        if model.optimize_transition_loss:
            total_cost += log_entry.transition_cost
    if im.has_invalid:
        log_entry.invalid = A.get_avg('invalid')

    adv_mean = np.array(A.get('adv_mean'), dtype=np.float32)
    adv_mean_magnitude = np.array(
        A.get('adv_mean_magnitude'), dtype=np.float32)
    adv_var = np.array(A.get('adv_var'), dtype=np.float32)
    adv_var_magnitude = np.array(A.get('adv_var_magnitude'), dtype=np.float32)

    if im.has_policy:
        log_entry.policy_cost = A.get_avg('policy_cost')
        total_cost += log_entry.policy_cost
    if im.has_value:
        log_entry.value_cost = A.get_avg('value_cost')
        total_cost += log_entry.value_cost

    def get_mean(x):
        val = x.mean()
        if isinstance(val, float):
            return val
        else:
            return float(val)

    if len(adv_mean) > 0:
        log_entry.mean_adv_mean = get_mean(adv_mean)
    if len(adv_mean_magnitude) > 0:
        log_entry.mean_adv_mean_magnitude = get_mean(adv_mean_magnitude)
    if len(adv_var) > 0:
        log_entry.mean_adv_var = get_mean(adv_var)
    if len(adv_var_magnitude) > 0:
        log_entry.mean_adv_var_magnitude = get_mean(adv_var_magnitude)

    if im.has_epsilon:
        log_entry.epsilon = model.spinn.epsilon
    if im.has_temperature:
        log_entry.temperature = model.spinn.temperature

    log_entry.total_cost = total_cost
    return log_entry


def eval_accumulate(model, data_manager, A, batch):

    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch

    im = inspect(model)

    # Accumulate stats for transition accuracy.
    if im.has_transition_loss:
        preds = [m["t_preds"]
                 for m in model.spinn.memories if m.get('t_preds', None) is not None]
        truth = [m["t_given"]
                 for m in model.spinn.memories if m.get('t_given', None) is not None]
        A.add('preds', preds)
        A.add('truth', truth)

    if im.has_invalid:
        A.add('invalid', model.spinn.invalid)


def eval_stats(model, A, eval_data):
    im = inspect(model)

    class_correct = A.get('class_correct')
    class_total = A.get('class_total')
    class_acc = sum(class_correct) / float(sum(class_total))
    eval_data.eval_class_accuracy = class_acc

    if im.has_transition_loss:
        all_preds = np.array(flatten(A.get('preds')))
        all_truth = np.array(flatten(A.get('truth')))
        avg_trans_acc = (all_preds == all_truth).sum() / \
            float(all_truth.shape[0])
        eval_data.eval_transition_accuracy = avg_trans_acc

    if im.has_invalid:
        eval_data.invalid = A.get_avg('invalid')

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))
    eval_data.time_per_token_seconds = time_metric

    return eval_data

def train_format(log_entry, extra=False, rl=False):
    stats_str = "Step: {step}"

    # Accuracy Component.
    stats_str += " Acc: {class_acc:.5f} {transition_acc:.5f}"

    # Cost Component.
    stats_str += " Cost: {total_loss:.5f} {xent_loss:.5f} {transition_loss:.5f} {l2_loss:.5f}"
    if log_entry.HasField('policy_cost'):
        stats_str += " p{policy_cost:.5f}"
    if log_entry.HasField('value_cost'):
        stats_str += " v{value_cost:.5f}"

    # Time Component.
    stats_str += " Time: {time:.5f}"

    # Extra Component.
    if extra:
        stats_str += "\nTrain Extra:"
        stats_str += " lr{learning_rate:.7f}"
        if log_entry.HasField('invalid'):
            stats_str += " inv{invalid:.3f}"

    # RL Component.
    if rl:
        stats_str += "\nTrain RL:"
        stats_str += " am{mean_adv_mean:.5f}"
        stats_str += " amm{mean_adv_mean_magnitude:.5f}"
        stats_str += " av{mean_adv_var:.5f}"
        stats_str += " avm{mean_adv_var_magnitude:.5f}"
        stats_str += " t{temperature:.3f}"
        stats_str += " eps{epsilon:.7f}"

    return stats_str

def eval_format(evaluation, extra=False):
    eval_str = "Step: {step} Eval acc: {class_acc:.5f} {transition_acc:.5f} {filename} Time: {time:.5f}"

    if extra:
        eval_str += "\nEval Extra:"
        if evaluation.HasField('invalid'):
            eval_str += " inv{invalid:.3f}"

    return eval_str

def log_formatter(log_entry, extra=False, rl=False):
    """Defines the log string to print to std error."""
    args = {
        'step': log_entry.step,
        'class_acc': log_entry.class_accuracy,
        'transition_acc': log_entry.transition_accuracy,
        'total_loss': log_entry.total_cost,
        'xent_loss': log_entry.cross_entropy_cost,
        'transition_loss': log_entry.transition_cost,
        'l2_loss': log_entry.l2_cost,
        'policy_cost': log_entry.policy_cost,
        'value_cost': log_entry.value_cost,
        'time': log_entry.time_per_token_seconds,
        'learning_rate': log_entry.learning_rate,
        'invalid': log_entry.invalid,
        'mean_adv_mean': log_entry.mean_adv_mean,
        'mean_adv_mean_magnitude': log_entry.mean_adv_mean_magnitude,
        'mean_adv_var': log_entry.mean_adv_var,
        'mean_adv_var_magnitude': log_entry.mean_adv_var_magnitude,
        'epsilon': log_entry.epsilon,
        'temperature': log_entry.temperature,
    }

    log_str = train_format(log_entry, extra, rl).format(**args)
    if len(log_entry.evaluation) > 0:
        for evaluation in log_entry.evaluation:
            eval_args = {
                'step': log_entry.step,
                'class_acc': evaluation.eval_class_accuracy,
                'transition_acc': evaluation.eval_transition_accuracy,
                'filename': evaluation.filename,
                'time': evaluation.time_per_token_seconds,
                'invalid': evaluation.invalid,
            }
            log_str += '\n' + eval_format(evaluation, extra).format(**eval_args)

    return log_str

def create_log_formatter(extra, rl):
    def fmt(log_entry):
        return log_formatter(log_entry, extra, rl)
    return fmt

