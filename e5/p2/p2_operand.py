"""P2: weight the complete first numeric operand in every <<equation>>.

The frozen forward is unchanged. A decoder call arms the next CE call only;
answer and reference CE retain their original implementations and reductions.
"""
from functools import lru_cache
import re
import torch
import torch.nn.functional as F


def gpt2_byte_decoder():
    # Fixed GPT-2 reversible byte alphabet; independent of transformers version.
    visible = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    mapping = {chr(b):b for b in visible}
    hidden = [b for b in range(256) if b not in visible]
    mapping.update({chr(256+i):b for i,b in enumerate(hidden)})
    return mapping

EQUATION = re.compile(r"<<([^<>]*)>>")
FIRST = re.compile(r"\s*\(*\s*([+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+))")


def operand_spans(text):
    spans = []
    equations = list(EQUATION.finditer(text))
    for eq in equations:
        match = FIRST.match(eq.group(1))
        # Two frozen training equations start with x/X, not a numeric operand.
        # Keep them uniformly weighted; never upweight the RHS or denominator.
        if re.match(r'\s*[xX]\s*(?:=|/)', eq.group(1)):
            continue
        if match is None or '=' not in eq.group(1):
            raise ValueError(f"Unsupported equation: {eq.group(0)!r}")
        spans.append((eq.start(1) + match.start(1), eq.start(1) + match.end(1)))
    if not equations:
        raise ValueError(f"No complete explanation equation: {text!r}")
    return spans


class OperandWeight:
    def __init__(self, tokenizer, factor=2.0):
        if factor < 1:
            raise ValueError('Operand factor must be >= 1')
        self.tokenizer, self.factor = tokenizer, factor
        self.byte_decoder = gpt2_byte_decoder()
        self.handles = []
        self.active = self.pending = False
        self.calls = 0

    @lru_cache(maxsize=16384)
    def token_weights(self, ids):
        # Reconstruct bytes before decoding: a Unicode symbol can span BPE tokens.
        pieces = []
        for i in ids:
            if i == -100:
                pieces.append(b'')
            elif i in self.tokenizer.all_special_ids:
                pieces.append(self.tokenizer.decode([i], clean_up_tokenization_spaces=False).encode('utf8'))
            else:
                pieces.append(bytes(self.byte_decoder[c] for c in self.tokenizer.convert_ids_to_tokens(i)))
        text = b''.join(pieces).decode('utf8')
        decoded = self.tokenizer.decode([i for i in ids if i != -100], clean_up_tokenization_spaces=False)
        if text != decoded:
            raise ValueError('Token-by-token text reconstruction failed')
        spans = [(len(text[:a].encode('utf8')),len(text[:b].encode('utf8'))) for a,b in operand_spans(text)]
        weights, pos = [], 0
        for token, piece in zip(ids, pieces):
            end = pos + len(piece)
            hit = any(pos < b and end > a for a, b in spans)
            weights.append(0.0 if token == -100 else self.factor if hit else 1.0)
            pos = end
        return tuple(weights)

    def weighted_ce(self, logits, labels):
        weights = torch.tensor(self.token_weights(tuple(labels.detach().cpu().tolist())), device=logits.device, dtype=torch.float32)
        losses = F.cross_entropy(logits.float(), labels, ignore_index=-100, reduction='none')
        return (losses * weights).sum() / weights.sum()

    def attach(self, model):
        if self.handles or getattr(model.model_args, 'decoder_path', None):
            raise ValueError('Requires unattached frozen GPT-2 configuration')

        def begin(module, args, kwargs):
            self.active = module.training
            self.pending = False
            if self.active and kwargs['encoder_input_ids'].shape[0] != 1:
                raise ValueError('P2 preserves frozen microbatch size 1')

        def decoder(module, args, kwargs):
            if self.active:
                if self.pending:
                    raise RuntimeError('Previous explanation CE was not consumed')
                self.pending = True

        def loss(module, args, output):
            if self.active and self.pending:
                self.pending = False
                self.calls += 1
                return self.weighted_ce(*args)

        def finish(module, args, output):
            pending = self.pending
            self.active = self.pending = False
            if output is not None and pending:
                raise RuntimeError('Unconsumed explanation loss')

        self.handles = [model.register_forward_pre_hook(begin, with_kwargs=True),
                        model.decoder.register_forward_pre_hook(decoder, with_kwargs=True),
                        model.loss_fct.register_forward_hook(loss),
                        model.register_forward_hook(finish, always_call=True)]
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = self.pending = False
