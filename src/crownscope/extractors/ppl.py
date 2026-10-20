"""`ppl` signal (SRS FR-3.6): mean token log-probability, as perplexity, under a scoring language model.

Machine-generated text tends to be more predictable to a language model than human writing, so low perplexity
raises suspicion. The score is where the perplexity falls between the pack's `human` and `machine` reference
values, on a log scale. Those values depend on the model and the language, so the packs' numbers are
placeholders to be fitted (the model's scale differs between, say, a 2B and a 7B model).

Stateless, so it works in Single Mode. Length-gated by the pipeline: never computed below the pack's
`min_length` (FR-2.2). Records the scored token count in the result (FR-1.3) and the model in `Result.models`.

Pack inputs, options.ppl:
  model{provider, name, local_only, device, dtype}, max_tokens, max_chunks, min_tokens,
  perplexity{human, machine}, full_confidence_tokens, max_confidence
"""
from __future__ import annotations

import math

from ..result import Evidence, Signal
from . import ramp, register, unavailable


def score_from_logprobs(logprobs: list[float], options: dict) -> tuple[float, float, float]:
    """(score, confidence, perplexity)."""
    mean = sum(logprobs) / len(logprobs)
    perplexity = math.exp(-mean)
    ref = options["perplexity"]
    # a lower perplexity is more machine-like, so the "human" value maps to 0 and the "machine" value to 1
    score = ramp(math.log(perplexity), math.log(ref["human"]), math.log(ref["machine"]))
    confidence = options["max_confidence"] * min(1.0, len(logprobs) / options["full_confidence_tokens"])
    return score, confidence, perplexity


class _Ppl:
    name = "ppl"
    requires_model = True  # so the pipeline skips it below min_length (FR-2.2)
    min_length = 500
    relational = False

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.ppl")
        lm = getattr(ctx, "language_model", None)
        if lm is None:
            return unavailable(self.name, getattr(ctx, "language_model_error", None) or "no scoring language model available")
        try:
            logprobs = lm.token_logprobs(text, options["max_tokens"], options["max_chunks"])
        except Exception as e:  # a model error must not take the document, or the batch, down (NFR-10)
            return unavailable(self.name, f"scoring failed: {e}")
        if len(logprobs) < options["min_tokens"]:
            return unavailable(self.name, f"too few tokens to score: {len(logprobs)} < {options['min_tokens']}")
        if not all(math.isfinite(x) and x <= 1e-6 for x in logprobs):
            return unavailable(self.name, "the model returned invalid log-probabilities")

        score, confidence, perplexity = score_from_logprobs(logprobs, options)
        ctx.cache["tokens"] = len(logprobs)
        ctx.cache.setdefault("models", {})["ppl"] = {"name": lm.name, "version": lm.version}
        evidence = [Evidence(self.name, "number", f"perplexity={perplexity:.2f}"),
                    Evidence(self.name, "number", f"mean_logprob={sum(logprobs) / len(logprobs):.3f}"),
                    Evidence(self.name, "number", f"tokens={len(logprobs)}")]
        return Signal(self.name, score=score, confidence=confidence), evidence


register(_Ppl())
