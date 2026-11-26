"""P4 contextual end-of-step alignment, without changing the frozen CODI source."""
import math
import torch
import torch.nn.functional as F
from e5.p1_schedule import preflight as baseline_preflight

METHOD = dict(variant="step_alignment_v1", alignment_factor=0.1,
    alignment_loss="mean(1-cosine), FP32, valid slots only",
    student="projected z0..z4; z4 is readout-only",
    teacher="existing no-grad reference pass, last layer at closing >> token",
    tail="first four steps individually; slot 4 uses final end of merged tail",
    teacher_stop_gradient=True, extra_reference_forward=False,
    prefix_mask=False, operand_weight=1.0, explain_loss_factor=1.0,
    reference_warmstart=False, total_updates=750,
    status="exploratory: strong-reference prerequisite not established")


def preflight(init=None, require_weights=False):
    result = baseline_preflight(init, require_weights)
    result["variant"] = METHOD.copy()
    return result


def step_spans(ids, slots=5, start_ids=(16791, 9959), end_id=4211,
               stop_ids=(50256, 50257)):
    """Match frozen get_steps token scanning and tail merge; keep source offsets."""
    if slots < 1:
        raise ValueError("slots must be positive")
    ids = list(ids)
    end = next((i for i, x in enumerate(ids) if x in stop_ids), len(ids))
    spans, i = [], 0
    while i < end:
        if ids[i] in start_ids:
            j = next((j for j in range(i + 1, end) if ids[j] == end_id), None)
            if j is not None:
                spans.append((i, j))
                i = j + 1
                continue
        i += 1
    groups = [[span] for span in spans[:slots]]
    if len(spans) > slots:
        groups[-1] = spans[slots - 1:]
    return groups + [[] for _ in range(slots - len(groups))]


def batch_positions(batch, tokenizer, slots=5):
    """Refuse accidental question/answer alignment; PAD slots use -1."""
    positions = []
    prefix_len = len(tokenizer.encode("The answer is:"))
    for ids, labels, answer_pos in zip(batch["ref_input_ids"].tolist(),
            batch["ref_labels"].tolist(), batch["ref_answer_position"].tolist()):
        groups = step_spans(ids, slots, stop_ids=(tokenizer.eos_token_id, tokenizer.pad_token_id))
        cot_end = answer_pos - prefix_len
        for group in groups:
            for start, end in group:
                if end >= cot_end or any(x == -100 for x in labels[start:end + 1]):
                    raise ValueError("Equation span outside supervised CoT")
        positions.append([group[-1][1] if group else -1 for group in groups])
    return torch.tensor(positions, dtype=torch.long, device=batch["ref_input_ids"].device)


def alignment_loss(student, teacher, positions):
    """No gradient enters teacher; zero-valid cases remain differentiable."""
    if student.ndim != 3 or positions.shape != student.shape[:2]:
        raise ValueError("Expected student [B,S,H] and positions [B,S]")
    if teacher.ndim != 3 or teacher.shape[0] != student.shape[0] or teacher.shape[2] != student.shape[2]:
        raise ValueError("Teacher/student hidden dimensions differ")
    valid = positions >= 0
    if bool((positions >= teacher.shape[1]).any()) or bool((positions < -1).any()):
        raise ValueError("Reference position out of bounds")
    if not bool(valid.any()):
        return student.float().sum() * 0
    target = teacher.detach().gather(1, positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, teacher.shape[-1]))
    return (1 - F.cosine_similarity(student[valid].float(), target[valid].float(), dim=-1, eps=1e-8)).mean()


def training_forward(model, batch, completed_updates, factor=0.1):
    """Reuse baseline forward/RNG exactly; hooks are removed even on exceptions."""
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("Alignment factor must be finite and nonnegative")
    if not model.use_prj or model.num_latent != 4:
        raise ValueError("P4 requires the frozen projector and four latent iterations")
    positions = batch_positions(batch, model.tokenizer, model.num_latent + 1)
    latents, reference = [], []

    def capture_latent(module, args, output):
        latents.append(output)

    def capture_reference(module, args, kwargs, output):
        # Exact tensor identity selects the two reference passes, not the question.
        if kwargs.get("input_ids") is batch["ref_input_ids"] and not torch.is_grad_enabled():
            reference.append(output.hidden_states[-1].detach())

    handles = [model.prj.register_forward_hook(capture_latent),
               model.codi.register_forward_hook(capture_reference, with_kwargs=True)]
    try:
        result = model(**batch, step=completed_updates, step_ratio=completed_updates / 750)
        if len(latents) != 5 or len(reference) != 1:
            raise RuntimeError("Frozen forward capture mismatch: expected 5 latents and one no-grad teacher")
        raw = alignment_loss(torch.cat(latents, dim=1), reference[0], positions)
        base = result["loss"]
        # A zero coefficient is an exact baseline parity path (including gradients).
        result["loss"] = base + factor * raw if factor else base
        result.update(base_loss=base.detach(), alignment_loss=raw.detach(),
                      weighted_alignment_loss=(factor * raw).detach(),
                      alignment_valid_slots=(positions >= 0).sum().detach())
        return result
    finally:
        for handle in handles:
            handle.remove()
