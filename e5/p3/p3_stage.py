"""P3: reference CE for updates 1..150, unchanged joint objective thereafter."""
from e5.p1_schedule import preflight as baseline_preflight

METHOD = dict(variant="reference_warmstart_v1", reference_updates=150,
              joint_updates=600, total_updates=750, explain_loss_factor=1.0,
              prefix_mask=False, operand_weight=1.0,
              reference_target="existing masked-question CoT + answer CE",
              scheduler="continuous frozen v1 schedule; no restart at update 151",
              inactive_parameters="grad=None; no AdamW decay or optimizer state update")


def stage(completed_updates):
    if type(completed_updates) is not int or not 0 <= completed_updates < 750:
        raise ValueError("Expected completed_updates in [0, 749]")
    return "reference_only" if completed_updates < 150 else "joint"


def preflight(init=None, require_weights=False):
    result = baseline_preflight(init, require_weights)
    result["variant"] = METHOD.copy()
    return result


def training_forward(model, batch, completed_updates):
    """Do not backpropagate the detached diagnostic ref_ce_loss from CODI.forward."""
    if stage(completed_updates) == "joint":
        return model(**batch, step=completed_updates, step_ratio=completed_updates / 750)
    output = model.codi(input_ids=batch["ref_input_ids"],
                        attention_mask=batch["ref_attention_mask"])
    logits = output.logits
    loss = model.loss_fct(logits[:, :-1].reshape(-1, logits.size(-1)),
                          batch["ref_labels"][:, 1:].reshape(-1)) * model.ref_loss_factor
    # Null means not computed, not a measured zero loss.
    return dict(loss=loss, ref_ce_loss=loss.detach(), ce_loss=None,
                distill_loss=None, explain_loss=None)
