"""
logging.py

Log format convenience methods for training spinn.

"""

import numpy as np
from spinn.util.blocks import flatten
from spinn.util.misc import time_per_token


def train_accumulate(model, data_manager, A, batch):

    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch

    has_spinn = hasattr(model, 'spinn')
    has_transition_loss = hasattr(model, 'transition_loss') and model.transition_loss is not None
    has_invalid = has_spinn and hasattr(model.spinn, 'invalid')

    # Accumulate stats for transition accuracy.
    if has_transition_loss:
        preds = [m["t_preds"] for m in model.spinn.memories if 't_preds' in m]
        truth = [m["t_given"] for m in model.spinn.memories if 't_given' in m]
        A.add('preds', preds)
        A.add('truth', truth)

    if has_invalid:
        A.add('invalid', model.spinn.invalid)


def train_rl_accumulate(model, data_manager, A, batch):
    has_policy = hasattr(model, 'policy_loss')
    has_value = hasattr(model, 'value_loss')

    if has_policy:
        A.add('policy_loss', model.policy_loss.data[0])

    if has_value:
        A.add('value_loss', model.value_loss.data[0])

    A.add('adv_mean', model.stats['mean'])
    A.add('adv_mean_magnitude', model.stats['mean_magnitude'])
    A.add('adv_var', model.stats['var'])
    A.add('adv_var_magnitude', model.stats['var_magnitude'])


def train_metrics(M, stats_args, step):
    metric_stats = ['class_acc', 'total_loss', 'transition_acc', 'transition_loss']
    for key in metric_stats:
        M.write(key, stats_args[key], step)


def train_rl_metrics(M, stats_args, step):
    stats_rl_args_keys = ['policy_loss', 'value_loss',
                          'mean_adv_mean', 'mean_adv_mean_magnitude',
                          'mean_adv_var', 'mean_adv_var_magnitude']
    for key in stats_rl_args_keys:
        M.write(key, stats_args[key], step)


def train_stats(model, optimizer, A, step):

    has_spinn = hasattr(model, 'spinn')
    has_transition_loss = hasattr(model, 'transition_loss') and model.transition_loss is not None
    has_invalid = has_spinn and hasattr(model.spinn, 'invalid')

    if has_transition_loss:
        all_preds = np.array(flatten(A.get('preds')))
        all_truth = np.array(flatten(A.get('truth')))
        avg_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))

    ret = dict(
        step=step,
        class_acc=A.get_avg('class_acc'),
        transition_acc=avg_trans_acc if has_transition_loss else 0.0,
        xent_loss=A.get_avg('xent_loss'),  # not actual mean
        transition_loss=model.transition_loss.data[0] if has_transition_loss else 0.0,
        total_loss=A.get_avg('total_loss'),
        auxiliary_loss=A.get_avg('auxiliary_loss'),
        l2_loss=A.get_avg('l2_loss'),  # not actual mean
        invalid=A.get_avg('invalid') if has_invalid else 0.0,
        learning_rate=optimizer.lr,
        time=time_metric,
    )

    return ret


def train_rl_stats(model, data_manager, A, batch):
    has_policy = hasattr(model, 'policy_loss')
    has_value = hasattr(model, 'value_loss')

    adv_mean = np.array(A.get('adv_mean'), dtype=np.float32)
    adv_mean_magnitude = np.array(A.get('adv_mean_magnitude'), dtype=np.float32)
    adv_var = np.array(A.get('adv_var'), dtype=np.float32)
    adv_var_magnitude = np.array(A.get('adv_var_magnitude'), dtype=np.float32)

    ret = dict(
        policy_loss=A.get_avg('policy_loss') if has_policy else 0.0,
        value_loss=A.get_avg('value_loss') if has_value else 0.0,
        mean_adv_mean=adv_mean.mean(),
        mean_adv_mean_magnitude=adv_mean_magnitude.mean(),
        mean_adv_var=adv_var.mean(),
        mean_adv_var_magnitude=adv_var_magnitude.mean(),
        epsilon=model.spinn.epsilon,
        temperature=model.spinn.temperature,
    )

    return ret


def train_format(model):

    has_spinn = hasattr(model, 'spinn')

    stats_str = "Step: {step}"

    # Accuracy Component.
    stats_str += " Acc: {class_acc:.5f} {transition_acc:.5f}"

    # Cost Component.
    stats_str += " Cost: {total_loss:.5f} {xent_loss:.5f} {transition_loss:.5f} {l2_loss:.5f}"
    if has_spinn and hasattr(model, 'policy_loss'):
        stats_str += " p{policy_loss:.5f}"
    if has_spinn and hasattr(model, 'value_loss'):
        stats_str += " v{value_loss:.5f}"

    # Time Component.
    stats_str += " Time: {time:.5f}"

    return stats_str


def train_extra_format(model):

    # Extra Component.
    extra_str = "Train Extra:"
    extra_str += " lr{learning_rate:.7f}"
    if hasattr(model, "spinn") and hasattr(model.spinn, "invalid"):
        extra_str += " inv{invalid:.3f}"

    return extra_str


def train_rl_format(model):

    # Extra Component.
    extra_str = "Train RL:"
    extra_str += " am{mean_adv_mean:.5f}"
    extra_str += " amm{mean_adv_mean_magnitude:.5f}"
    extra_str += " av{mean_adv_var:.5f}"
    extra_str += " avm{mean_adv_var_magnitude:.5f}"
    extra_str += " t{temperature:.3f}"
    extra_str += " eps{epsilon:.7f}"

    return extra_str


def eval_accumulate(model, data_manager, A, batch):

    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch

    has_spinn = hasattr(model, 'spinn')
    has_transition_loss = hasattr(model, 'transition_loss') and model.transition_loss is not None
    has_invalid = has_spinn and hasattr(model.spinn, 'invalid')

    # Accumulate stats for transition accuracy.
    if has_transition_loss:
        preds = [m["t_preds"] for m in model.spinn.memories if 't_preds' in m]
        truth = [m["t_given"] for m in model.spinn.memories if 't_given' in m]
        A.add('preds', preds)
        A.add('truth', truth)

    if has_invalid:
        A.add('invalid', model.spinn.invalid)


def eval_format(model):
    eval_str = "Step: {step} Eval acc: {class_acc:.5f} {transition_acc:.5f} {filename} Time: {time:.5f}"

    return eval_str


def eval_extra_format(model):
    extra_str = "Eval Extra:"
    if hasattr(model, 'spinn'):
        extra_str += " inv{invalid:.3f}"

    return extra_str


def eval_metrics(M, stats_args, step):
    metric_stats = ['class_acc', 'transition_acc']
    for key in metric_stats:
        M.write("eval_" + key, stats_args[key], step)


def eval_rl_metrics(M, stats_args, step):
    pass


def eval_stats(model, A, step):

    has_spinn = hasattr(model, 'spinn')
    has_transition_loss = hasattr(model, 'transition_loss') and model.transition_loss is not None
    has_invalid = has_spinn and hasattr(model.spinn, 'invalid')

    class_correct = A.get('class_correct')
    class_total = A.get('class_total')
    class_acc = sum(class_correct) / float(sum(class_total))

    if has_transition_loss:
        all_preds = np.array(flatten(A.get('preds')))
        all_truth = np.array(flatten(A.get('truth')))
        avg_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))

    ret = dict(
        step=step,
        class_acc=class_acc,
        transition_acc=avg_trans_acc if has_transition_loss else 0.0,
        invalid=A.get_avg('invalid') if has_invalid else 0.0,
        time=time_metric,
    )

    return ret
