"""Scoring language models for the model-probability signals (SRS FR-3.6 `ppl`, FR-3.7 `burst`, FR-3.8 `fast_detect`).

A `LanguageModel` is anything with `name`, `version` and `token_logprobs(text, max_tokens, max_chunks)`, which
returns the natural-log probability of each scored token given the tokens before it. `burst` also needs
`scored_tokens(...)`, the same with each token's (char_start, char_end) in the text, to group tokens by sentence. `fast_detect` needs
`token_curvature(...)`: per token the log-probability, its expected value and its variance under the model's own
next-token distribution. A pack names a provider and
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

    # optional: scored_tokens(...) for burst and token_curvature(...) for fast_detect, described above


class TransformersLM:
    """A Hugging Face causal language model (the `model` extra: transformers and torch)."""

    ROWS = 32  # positions whose full next-token distribution is held at once
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
        self._last = None

    def _score(self, text: str, max_tokens: int, max_chunks: int) -> list[tuple]:
        """(char_start, char_end, logprob, expected logprob, variance of logprob) per scored token.

        The expected value and variance are over the model's own next-token distribution (what `fast_detect`
        needs). The spans are None for a slow tokenizer. The last result is kept, so `ppl`, `burst` and
        `fast_detect` on one document cost one forward pass.
        """
        key = (text, max_tokens, max_chunks, self.bos_id, self.context)
        if self._last and self._last[0] == key:
            return self._last[1]
        torch = self._torch
        limit = min(max_tokens, self.context) if self.context else max_tokens
        room = limit - (1 if self.bos_id is not None else 0)
        text = text[: max_tokens * max_chunks * self.CHARS_PER_TOKEN_CAP]
        fast = getattr(self.tokenizer, "is_fast", False)
        enc = self.tokenizer(text, add_special_tokens=False, verbose=False, return_offsets_mapping=fast)
        ids, spans = enc["input_ids"], (enc["offset_mapping"] if fast else None)
        out: list[tuple] = []
        for c in range(max_chunks):
            chunk = ids[c * room:(c + 1) * room]
            seq = ([self.bos_id] if self.bos_id is not None else []) + chunk
            if len(seq) < 2:
                break
            batch = torch.tensor([seq], device=self.device)
            with torch.no_grad():
                logits = self.model(batch).logits[0, :-1].float()
                lp, mu, var = [], [], []
                for r in range(0, len(logits), self.ROWS):  # bounded memory: vocabulary-sized rows, a few at a time
                    logp = torch.log_softmax(logits[r:r + self.ROWS], dim=-1)
                    p = logp.exp()
                    m = (p * logp).sum(-1)
                    lp.append(logp.gather(1, batch[0, 1 + r:1 + r + len(logp), None])[:, 0])
                    mu.append(m)
                    var.append(((p * logp * logp).sum(-1) - m * m).clamp(min=0))
                lp, mu, var = torch.cat(lp), torch.cat(mu), torch.cat(var)
            # without a BOS the chunk's first token has no context and is not scored
            where = spans[c * room:(c + 1) * room][(0 if self.bos_id is not None else 1):] if fast else None
            for i, row in enumerate(zip(lp.tolist(), mu.tolist(), var.tolist())):
                out.append((*(where[i] if where else (None, None)), *row))
        self._last = (key, out)
        return out

    def token_logprobs(self, text: str, max_tokens: int, max_chunks: int) -> list[float]:
        return [t[2] for t in self._score(text, max_tokens, max_chunks)]

    def scored_tokens(self, text: str, max_tokens: int, max_chunks: int) -> list[tuple[int, int, float]]:
        """Like `token_logprobs`, with each token's character span in `text` (needs a fast tokenizer)."""
        out = self._score(text, max_tokens, max_chunks)
        if out and out[0][0] is None:
            raise RuntimeError("this tokenizer cannot give character offsets (a fast tokenizer is needed)")
        return [t[:3] for t in out]

    def token_curvature(self, text: str, max_tokens: int, max_chunks: int) -> list[tuple[float, float, float]]:
        """(logprob, expected logprob, variance of logprob) per scored token, for `fast_detect`."""
        return [t[2:] for t in self._score(text, max_tokens, max_chunks)]


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
