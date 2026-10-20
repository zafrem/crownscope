"""Scoring language models for the model-probability signals (SRS FR-3.6 `ppl`; later `burst` and others).

A `LanguageModel` is anything with `name`, `version` and `token_logprobs(text, max_tokens, max_chunks)`, which
returns the natural-log probability of each scored token given the tokens before it. A pack names a provider and
a model (a Hugging Face model name or, for an install with no network, a local directory); code may also pass a
model to `Detector(pack, language_model=...)`. Nothing is loaded unless a pack enables a signal that needs one,
and a model that cannot be loaded makes the signal `unavailable` with the reason (FR-3.12), never a crash.

Long text is scored in consecutive chunks of at most `max_tokens` tokens, at most `max_chunks` of them, so the
cost per document is bounded. Each chunk is scored on its own, started with the model's BOS token when it has
one (so the first real token is scored too) and otherwise without the first token, which has no context.
"""
from __future__ import annotations

from typing import Protocol


class LanguageModel(Protocol):
    name: str
    version: str  # library versions, recorded in results for reproducibility (NFR-9)

    def token_logprobs(self, text: str, max_tokens: int, max_chunks: int) -> list[float]: ...


class TransformersLM:
    """A Hugging Face causal language model (the `model` extra: transformers and torch)."""

    CHARS_PER_TOKEN_CAP = 8  # text beyond max_tokens * max_chunks * this is never tokenized

    def __init__(self, name: str, local_only: bool = False, device: str | None = None, dtype: str | None = None):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        kwargs = {"local_files_only": True} if local_only else {}
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        model_kwargs = dict(kwargs)
        if dtype:
            # transformers 4.56 renamed `torch_dtype` to `dtype` and warns about the old name from then on
            major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
            model_kwargs["dtype" if major_minor >= (4, 56) else "torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs).eval()
        if device:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.name = name
        self.version = f"transformers {transformers.__version__}, torch {torch.__version__}"
        self.bos_id = self.tokenizer.bos_token_id
        self.context = getattr(self.model.config, "max_position_embeddings", None)

    def token_logprobs(self, text: str, max_tokens: int, max_chunks: int) -> list[float]:
        torch = self._torch
        limit = min(max_tokens, self.context) if self.context else max_tokens
        room = limit - (1 if self.bos_id is not None else 0)
        text = text[: max_tokens * max_chunks * self.CHARS_PER_TOKEN_CAP]
        ids = self.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        out: list[float] = []
        for c in range(max_chunks):
            chunk = ids[c * room:(c + 1) * room]
            seq = ([self.bos_id] if self.bos_id is not None else []) + chunk
            if len(seq) < 2:
                break
            batch = torch.tensor([seq], device=self.device)
            with torch.no_grad():
                logits = self.model(batch).logits[0, :-1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            out += logprobs.gather(1, batch[0, 1:, None])[:, 0].tolist()
        return out


def load_language_model(spec: dict | None) -> tuple[LanguageModel | None, str | None]:
    """(model, None) or (None, reason)."""
    if not spec or not spec.get("name"):
        return None, "no scoring language model configured (set options.ppl.model.name)"
    provider = spec.get("provider", "transformers")
    if provider != "transformers":
        return None, f"unknown language-model provider {provider!r}"
    try:
        return TransformersLM(spec["name"], spec.get("local_only", False), spec.get("device"), spec.get("dtype")), None
    except ImportError as e:
        return None, f"transformers/torch are not usable ({e}); install the `model` extra"
    except Exception as e:  # missing files, no network in local_only mode, out of memory ...
        return None, f"could not load language model {spec['name']!r}: {e}"
