def auxiliary_loss(model):

    has_spinn = hasattr(model, 'spinn')
    has_policy = has_spinn and hasattr(model, 'policy_loss')
    has_value = has_spinn and hasattr(model, 'value_loss')
    has_rae = has_spinn and hasattr(model.spinn, 'rae_loss')
    has_leaf = has_spinn and hasattr(model.spinn, 'leaf_loss')
    has_gen = has_spinn and hasattr(model.spinn, 'gen_loss')

    # Optionally calculate transition loss/accuracy.
    policy_loss = model.policy_loss if has_policy else None
    value_loss = model.value_loss if has_value else None
    rae_loss = model.spinn.rae_loss if has_rae else None
    leaf_loss = model.spinn.leaf_loss if has_leaf else None
    gen_loss = model.spinn.gen_loss if has_gen else None

    total_loss = 0.0
    if has_policy:
        total_loss += model.policy_loss
    if has_value:
        total_loss += model.value_loss
    if has_rae:
        total_loss += model.spinn.rae_loss
    if has_leaf:
        total_loss += model.spinn.leaf_loss
    if has_gen:
        total_loss += model.spinn.gen_loss

    return total_loss