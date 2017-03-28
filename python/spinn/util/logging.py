"""
logging.py

Log format convenience methods for training spinn.

"""

import numpy as np
from spinn.util.blocks import flatten
from spinn.util.misc import time_per_token
from spinn.data import T_SHIFT, T_REDUCE, T_SKIP, T_STRUCT


def train_accumulate(model, data_manager, A, batch):

    X_batch, transitions_batch, y_batch, num_transitions_batch, spans, train_ids = batch

    has_spinn = hasattr(model, 'spinn')
    has_transition_loss = hasattr(model, 'transition_loss') and model.transition_loss is not None
    has_invalid = has_spinn and hasattr(model.spinn, 'invalid')
    has_policy = has_spinn and hasattr(model, 'policy_loss')
    has_value = has_spinn and hasattr(model, 'value_loss')
    has_rae = has_spinn and hasattr(model.spinn, 'rae_loss')
    has_leaf = has_spinn and hasattr(model.spinn, 'leaf_loss')
    has_gen = has_spinn and hasattr(model.spinn, 'gen_loss')
    has_entropy = hasattr(model, 'avg_entropy')

    # Accumulate stats for transition accuracy.
    if has_transition_loss:
        preds = [m["t_preds"] for m in model.spinn.memories if m.get('t_preds', None) is not None]
        truth = [m["t_given"] for m in model.spinn.memories if m.get('t_given', None) is not None]
        A.add('preds', preds)
        A.add('truth', truth)

    # Accumulate stats for leaf prediction accuracy.
    if has_leaf:
        A.add('leaf_acc', model.spinn.leaf_acc)

    # Accumulate stats for word prediction accuracy.
    if has_gen:
        A.add('gen_acc', model.spinn.gen_acc)

    if has_entropy:
        A.add('entropy', model.avg_entropy)

    if has_invalid:
        A.add('invalid', model.spinn.invalid)

    if has_transition_loss and hasattr(data_manager, 'spans'):
        transitions_per_example = model.spinn.get_transitions_per_example().tolist()
        for t_idx, span in enumerate(spans):
            t_spans = data_manager.spans(transitions_per_example[t_idx])
            target = set([node.span for node in span if node.tag == 'struct'])
            actual = set([node.span for node in t_spans])
            common = set.intersection(target, actual)

            A.add('n_struct_target', len(target))
            A.add('n_struct_common', len(common))


def train_rl_accumulate(model, data_manager, A, batch):
    A.add('adv_mean', model.stats['mean'])
    A.add('adv_mean_magnitude', model.stats['mean_magnitude'])
    A.add('adv_var', model.stats['var'])
    A.add('adv_var_magnitude', model.stats['var_magnitude'])


def train_stats(model, optimizer, A, step):

    has_spinn = hasattr(model, 'spinn')
    has_transition_loss = hasattr(model, 'transition_loss') and model.transition_loss is not None
    has_invalid = has_spinn and hasattr(model.spinn, 'invalid')
    has_policy = has_spinn and hasattr(model, 'policy_loss')
    has_value = has_spinn and hasattr(model, 'value_loss')
    has_rae = has_spinn and hasattr(model.spinn, 'rae_loss')
    has_leaf = has_spinn and hasattr(model.spinn, 'leaf_loss')
    has_gen = has_spinn and hasattr(model.spinn, 'gen_loss')
    has_epsilon = has_spinn and hasattr(model.spinn, "epsilon")
    has_entropy = hasattr(model, 'avg_entropy')

    if has_transition_loss:
        all_preds = np.array(flatten(A.get('preds')))
        all_truth = np.array(flatten(A.get('truth')))
        avg_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))

    n_struct_common = A.get('n_struct_common')
    n_struct_target = A.get('n_struct_target')
    if len(n_struct_target) > 0:
        struct = sum(n_struct_common) / float(sum(n_struct_target))
    else:
        struct = 0.0

    ret = dict(
        step=step,
        class_acc=A.get_avg('class_acc'),
        transition_acc=avg_trans_acc if has_transition_loss else 0.0,
        xent_cost=A.get_avg('xent_cost'), # not actual mean
        transition_cost=model.transition_loss.data[0] if has_transition_loss else 0.0,
        l2_cost=A.get_avg('l2_cost'), # not actual mean
        policy_cost=model.policy_loss.data[0] if has_policy else 0.0,
        value_cost=model.value_loss.data[0] if has_value else 0.0,
        invalid=A.get_avg('invalid') if has_invalid else 0.0,
        epsilon=model.spinn.epsilon if has_epsilon else 0.0,
        avg_entropy=A.get('avg_entropy') if has_entropy else 0.0,
        rae_cost=model.spinn.rae_loss.data[0] if has_rae else 0.0,
        leaf_acc=A.get_avg('leaf_acc') if has_leaf else 0.0,
        leaf_cost=model.spinn.leaf_loss.data[0] if has_leaf else 0.0,
        gen_acc=A.get_avg('gen_acc') if has_gen else 0.0,
        gen_cost=model.spinn.gen_loss.data[0] if has_gen else 0.0,
        struct=struct,
        learning_rate=optimizer.lr,
        time=time_metric,
    )

    total_cost = 0.0
    for key in ret.keys():
        if key == 'transition_cost' and has_transition_loss and not model.optimize_transition_loss:
            pass
        elif 'cost' in key:
            total_cost += ret[key]
    ret['total_cost'] = total_cost

    return ret


def train_rl_stats(model, data_manager, A, batch):
    adv_mean = np.array(A.get('adv_mean'), dtype=np.float32)
    adv_mean_magnitude = np.array(A.get('adv_mean_magnitude'), dtype=np.float32)
    adv_var = np.array(A.get('adv_var'), dtype=np.float32)
    adv_var_magnitude = np.array(A.get('adv_var_magnitude'), dtype=np.float32)

    ret = dict(
        mean_adv_mean=adv_mean.mean(),
        mean_adv_mean_magnitude=adv_mean_magnitude.mean(),
        mean_adv_var=adv_var.mean(),
        mean_adv_var_magnitude=adv_var_magnitude.mean(),
        var_adv_mean=adv_mean.var(),
        var_adv_mean_magnitude=adv_mean_magnitude.var(),
        var_adv_var=adv_var.var(),
        var_adv_var_magnitude=adv_var_magnitude.var()
        )

    return ret


def train_format(model):

    has_spinn = hasattr(model, 'spinn')

    stats_str = "Step: {step}"

    # Accuracy Component.
    stats_str += " Acc: {class_acc:.5f} {transition_acc:.5f}"
    if has_spinn and hasattr(model.spinn, 'leaf_loss'):
        stats_str += " leaf{leaf_acc:.5f}"
    if has_spinn and hasattr(model.spinn, 'gen_loss'):
        stats_str += " gen{gen_acc:.5f}"

    # Cost Component.
    stats_str += " Cost: {total_cost:.5f} {xent_cost:.5f} {transition_cost:.5f} {l2_cost:.5f}"
    if has_spinn and hasattr(model, 'policy_loss'):
        stats_str += " p{policy_cost:.5f}"
    if has_spinn and hasattr(model, 'value_loss'):
        stats_str += " v{value_cost:.5f}"
    # if hasattr(model, 'avg_entropy'):
    #     stats_str += " e{avg_entropy:.5f}"
    if has_spinn and hasattr(model.spinn, 'rae_loss'):
        stats_str += " rae{rae_cost:.5f}"
    if has_spinn and hasattr(model.spinn, 'leaf_loss'):
        stats_str += " leaf{leaf_cost:.5f}"
    if has_spinn and hasattr(model.spinn, 'gen_loss'):
        stats_str += " gen{gen_cost:.5f}"

    # Time Component.
    stats_str += " Time: {time:.5f}"

    return stats_str


def train_extra_format(model):

    # Extra Component.
    extra_str = "Train Extra:"
    extra_str += " lr{learning_rate:.7f}"
    if hasattr(model, "spinn") and hasattr(model.spinn, "epsilon"):
        extra_str += " eps{epsilon:.7f}"
    if hasattr(model, "spinn") and hasattr(model.spinn, "invalid"):
        extra_str += " inv{invalid:.3f}"
    if hasattr(model, "spinn"):
        extra_str += " sub{struct:.3f}"

    return extra_str


def train_rl_format(model):

    # Extra Component.
    extra_str = "Train RL:"
    extra_str += " am{mean_adv_mean:.5f}"
    extra_str += " amm{mean_adv_mean_magnitude:.5f}"
    extra_str += " av{mean_adv_var:.5f}"
    extra_str += " avm{mean_adv_var_magnitude:.5f}"

    extra_str += " "
    extra_str += "(am{var_adv_mean:.5f}"
    extra_str += " amm{var_adv_mean_magnitude:.5f}"
    extra_str += " av{var_adv_var:.5f}"
    extra_str += " avm{var_adv_var_magnitude:.5f}"
    extra_str += ")"

    return extra_str


def eval_accumulate(model, data_manager, A, batch):

    X_batch, transitions_batch, y_batch, num_transitions_batch, spans, train_ids = batch

    has_spinn = hasattr(model, 'spinn')
    has_transition_loss = hasattr(model, 'transition_loss') and model.transition_loss is not None
    has_invalid = has_spinn and hasattr(model.spinn, 'invalid')
    has_policy = has_spinn and hasattr(model, 'policy_loss')
    has_value = has_spinn and hasattr(model, 'value_loss')
    has_rae = has_spinn and hasattr(model.spinn, 'rae_loss')
    has_leaf = has_spinn and hasattr(model.spinn, 'leaf_loss')
    has_gen = has_spinn and hasattr(model.spinn, 'gen_loss')
    has_entropy = hasattr(model, 'avg_entropy')

    # Accumulate stats for transition accuracy.
    if has_transition_loss:
        preds = [m["t_preds"] for m in model.spinn.memories if m.get('t_preds', None) is not None]
        truth = [m["t_given"] for m in model.spinn.memories if m.get('t_given', None) is not None]
        A.add('preds', preds)
        A.add('truth', truth)

    # Accumulate stats for leaf prediction accuracy.
    if has_leaf:
        A.add('leaf_acc', model.spinn.leaf_acc)

    # Accumulate stats for word prediction accuracy.
    if has_gen:
        A.add('gen_acc', model.spinn.gen_acc)

    if has_entropy:
        A.add('entropy', model.avg_entropy)

    if has_invalid:
        A.add('invalid', model.spinn.invalid)

    if has_transition_loss and hasattr(data_manager, 'spans'):
        transitions_per_example = model.spinn.get_transitions_per_example().tolist()
        for t_idx, span in enumerate(spans):
            t_spans = data_manager.spans(transitions_per_example[t_idx])
            target = set([node.span for node in span if node.tag == 'struct'])
            actual = set([node.span for node in t_spans])
            common = set.intersection(target, actual)

            A.add('n_struct_target', len(target))
            A.add('n_struct_common', len(common))


def eval_format(model):
    eval_str = "Step: {step} Eval acc: {class_acc:.5f} {transition_acc:.5f} {filename} Time: {time:.5f}"

    return eval_str


def eval_extra_format(model):
    extra_str = "Eval Extra:"
    if hasattr(model, 'spinn'):
        extra_str += " inv{invalid:.3f}"
    if hasattr(model, "spinn"):
        extra_str += " sub{struct:.3f}"

    return extra_str


def eval_stats(model, A, step):

    has_spinn = hasattr(model, 'spinn')
    has_transition_loss = hasattr(model, 'transition_loss') and model.transition_loss is not None
    has_invalid = has_spinn and hasattr(model.spinn, 'invalid')
    has_policy = has_spinn and hasattr(model, 'policy_loss')
    has_value = has_spinn and hasattr(model, 'value_loss')
    has_rae = has_spinn and hasattr(model.spinn, 'rae_loss')
    has_leaf = has_spinn and hasattr(model.spinn, 'leaf_loss')
    has_gen = has_spinn and hasattr(model.spinn, 'gen_loss')
    has_epsilon = has_spinn and hasattr(model.spinn, "epsilon")
    has_entropy = hasattr(model, 'avg_entropy')

    class_correct = A.get('class_correct')
    class_total = A.get('class_total')
    class_acc = sum(class_correct) / float(sum(class_total))

    if has_transition_loss:
        all_preds = np.array(flatten(A.get('preds')))
        all_truth = np.array(flatten(A.get('truth')))
        avg_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))

    n_struct_common = A.get('n_struct_common')
    n_struct_target = A.get('n_struct_target')
    if len(n_struct_target) > 0:
        struct = sum(n_struct_common) / float(sum(n_struct_target))
    else:
        struct = 0.0

    ret = dict(
        step=step,
        class_acc=class_acc,
        transition_acc=avg_trans_acc if has_transition_loss else 0.0,
        # xent_cost=A.get_avg('xent_cost'), # not actual mean
        # transition_cost=model.transition_loss.data[0] if has_transition_loss else 0.0,
        # policy_cost=model.policy_loss.data[0] if has_policy else 0.0,
        # value_cost=model.value_loss.data[0] if has_value else 0.0,
        invalid=A.get_avg('invalid') if has_invalid else 0.0,
        # epsilon=model.spinn.epsilon if has_epsilon else 0.0,
        # avg_entropy=A.get('avg_entropy') if has_entropy else 0.0,
        # rae_cost=model.spinn.rae_loss.data[0] if has_rae else 0.0,
        # leaf_acc=A.get_avg('leaf_acc') if has_leaf else 0.0,
        # leaf_cost=model.spinn.leaf_loss.data[0] if has_leaf else 0.0,
        # gen_acc=A.get_avg('gen_acc') if has_gen else 0.0,
        # gen_cost=model.spinn.gen_loss.data[0] if has_gen else 0.0,
        struct=struct,
        time=time_metric,
    )

    return ret
