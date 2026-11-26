"""Training-only masking at the auxiliary decoder input; upstream files unchanged.

The reasoner embedding hook records token IDs, not activations. Immediately before
the auxiliary decoder call the official forward has embedded that step's text.
The decoder pre-hook masks a new inputs_embeds tensor, leaving the original IDs,
labels, attention mask, latent prefix and all reasoner computations untouched.
Hooks are gated by an outer CODI training forward, so standalone generation is inert.
"""
import torch


class PrefixMask:
    def __init__(self, probability, seed, protected_ids):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        self.probability = probability
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.protected_ids = tuple(sorted(set(protected_ids)))
        self.active = False
        self.last_ids = None
        self.handles = []
        self.calls = 0

    def state_dict(self):
        return {"generator": self.generator.get_state().clone(), "probability": self.probability,
                "protected_ids": self.protected_ids}

    def load_state_dict(self, state):
        if state["probability"] != self.probability or tuple(state["protected_ids"]) != self.protected_ids:
            raise ValueError("Masking policy differs from checkpoint")
        self.generator.set_state(state["generator"].cpu())

    def corrupt(self, embeddings, ids):
        if self.probability == 0:
            return embeddings
        if embeddings.shape[:2] != (ids.shape[0], ids.shape[1] + 1):
            raise ValueError("Expected one latent prefix followed by the recorded text tokens")
        eligible = torch.ones_like(ids, dtype=torch.bool)
        for value in self.protected_ids:
            eligible &= ids != value
        # A private CPU generator avoids consuming the reasoner/decoder dropout RNG.
        selected = (torch.rand(ids.shape, generator=self.generator) < self.probability).to(ids.device)
        selected &= eligible
        text = embeddings[:, 1:].masked_fill(selected.unsqueeze(-1), 0)
        return torch.cat((embeddings[:, :1], text), dim=1)

    def attach(self, model):
        if self.handles:
            raise ValueError("Mask already attached")
        if getattr(model.model_args, "decoder_path", None):
            raise ValueError("E5 only supports the frozen GPT-2 decoder configuration")
        embedding = model.get_embd(model.codi, model.model_name)

        def begin(module, args, kwargs):
            self.active = module.training and self.probability > 0
            self.last_ids = None

        def finish(module, args, output):
            self.active = False
            self.last_ids = None

        def remember(module, args, output):
            if self.active:
                self.last_ids = args[0].detach()

        def mask(module, args, kwargs):
            if not self.active:
                return None
            if self.last_ids is None or "inputs_embeds" not in kwargs:
                raise ValueError("Decoder input no longer matches the audited official forward")
            changed = dict(kwargs)
            changed["inputs_embeds"] = self.corrupt(kwargs["inputs_embeds"], self.last_ids)
            self.last_ids = None
            self.calls += 1
            return args, changed

        self.handles = [model.register_forward_pre_hook(begin, with_kwargs=True),
                        model.register_forward_hook(finish, always_call=True),
                        embedding.register_forward_hook(remember),
                        model.decoder.register_forward_pre_hook(mask, with_kwargs=True)]
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = False
        self.last_ids = None


def attach_prefix_mask(model, method):
    # IDs from the frozen GPT-2 get_steps call: opening variants and closing marker.
    protected = list(model.tokenizer.all_special_ids) + [16791, 9959, 4211, model.bot_id, model.eot_id]
    return PrefixMask(method["probability"], method["mask_seed"], protected).attach(model)
