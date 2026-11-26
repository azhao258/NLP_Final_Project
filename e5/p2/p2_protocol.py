"""P2 reuses the frozen training/data checks, but never the P1 schedule."""
from e5.p1_schedule import preflight as baseline_preflight

METHOD = dict(variant='first_operand_weight_v1', operand_weight=2.0,
              other_token_weight=1.0, explain_loss_factor=1.0, prefix_mask=False,
              normalization='sum weighted token CE / sum valid token weights per slot; mean over effective slots',
              symbolic_policy='x=5 and X/2=0.5X have no first numeric operand: all valid tokens retain weight 1',
              span_policy='first signed decimal numeric literal per <<equation>>, including each equation in merged tail; overlapping BPE tokens')


def preflight(init=None, require_weights=False):
    result = baseline_preflight(init, require_weights)
    result['variant'] = METHOD.copy()
    return result


def explain_factor(completed_updates):
    return 1.0
