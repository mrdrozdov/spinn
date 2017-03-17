"""
logging.py

Log format convenience methods for training spinn.

"""


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
    if has_spinn and hasattr(model.spinn, 'policy_loss'):
        stats_str += " p{policy_cost:.5f}"
    if has_spinn and hasattr(model.spinn, 'value_loss'):
        stats_str += " v{value_cost:.5f}"
    if hasattr(model, 'avg_entropy'):
        stats_str += " e{avg_entropy:.5f}"
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
    extra_str += " lr={learning_rate:.7f}"
    if hasattr(model, "spinn") and hasattr(model.spinn, "epsilon"):
        extra_str += " eps={epsilon:.7f}"

    return extra_str
