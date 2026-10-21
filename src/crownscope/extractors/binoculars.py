"""`binoculars` signal (SRS FR-3.9): the ratio of one model's perplexity to the cross-perplexity between two models
(Hans et al., 2024), following the reference implementation.

Two models with the same tokenizer read the text. The numerator is the *performer's* mean negative log-probability
of the actual tokens (its log-perplexity). The denominator is the mean, over tokens, of the cross-entropy of the
*observer's* next-token distribution against the performer's log-probabilities: how surprising the performer finds
what the observer expects. Text that the models find equally unsurprising and predictable gives a low ratio; a
person's text is surprising to both in a way the denominator does not absorb, so the ratio stays higher. The
denominator is what a plain perplexity lacks: it normalises away how hard the *topic* is to predict, which is why
the method needs no per-domain reference for the perplexity itself.

    score = mean_t(-log p_performer(x_t)) / mean_t(-sum_v p_observer(v) log p_performer(v))

A lower ratio is more machine-like. The signal is where it falls between the pack's `human` (higher) and
`machine` (lower) reference values. Those depend on the model pair and the language, so the packs' numbers are
placeholders to be fitted. The paper's pair is a base model as observer and its instruction-tuned sibling as
performer.

The observer is the detector's shared scoring model (the one `ppl`, `burst` and `fast_detect` use, or this
signal's own `model` when it is the only one); the performer is `options.binoculars.performer`. Stateless, so it
works in Single Mode; it reads each document with both models, so it costs two forward passes per chunk.
Length-gated by the pipeline (500 characters, FR-2.2).

Pack inputs, options.binoculars:
  model{...} (observer; only if no other model signal is enabled), performer{provider, name, local_only, device, dtype},
  max_tokens, max_chunks, min_tokens, ratio{human, machine}, full_confidence_tokens, max_confidence
"""
from __future__ import annotations

import math

from ..result import Evidence, Signal
from . import ramp, register, unavailable


def ratio_from_cross(pairs: list[tuple[float, float]]) -> float | None:
    """mean(performer nll) / mean(cross-entropy); None when the cross-entropy is not positive."""
    cross = sum(c for _, c in pairs)
    if cross <= 0:
        return None
    return sum(n for n, _ in pairs) / cross


def score_from_ratio(ratio: float, n_tokens: int, options: dict) -> tuple[float, float]:
    """(score, confidence): the `human` ratio maps to 0 and the lower `machine` ratio to 1."""
    ref = options["ratio"]
    score = ramp(ratio, ref["human"], ref["machine"])
    confidence = options["max_confidence"] * min(1.0, n_tokens / options["full_confidence_tokens"])
    return score, confidence


class _Binoculars:
    name = "binoculars"
    requires_model = True  # so the pipeline skips it below min_length (FR-2.2)
    min_length = 500
    relational = False

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.binoculars")
        observer = getattr(ctx, "language_model", None)
        if observer is None:
            return unavailable(self.name, getattr(ctx, "language_model_error", None) or "no scoring language model available")
        performer = getattr(ctx, "performer", None)
        if performer is None:
            return unavailable(self.name, getattr(ctx, "performer_error", None) or "no performer language model available")
        if not hasattr(observer, "cross_scores"):
            return unavailable(self.name, "the scoring model cannot be read together with a second one (it needs cross_scores)")
        try:
            pairs = ctx.memo(("lm_cross", options["max_tokens"], options["max_chunks"]),
                             lambda: observer.cross_scores(performer, text, options["max_tokens"], options["max_chunks"]))
        except Exception as e:  # a model error must not take the document, or the batch, down (NFR-10)
            return unavailable(self.name, f"scoring failed: {e}")
        if len(pairs) < options["min_tokens"]:
            return unavailable(self.name, f"too few tokens to score: {len(pairs)} < {options['min_tokens']}")
        if not all(math.isfinite(x) and x >= -1e-6 for row in pairs for x in row):
            return unavailable(self.name, "the models returned invalid log-probabilities")
        ratio = ratio_from_cross(pairs)
        if ratio is None:
            return unavailable(self.name, "the cross-perplexity is not positive")

        score, confidence = score_from_ratio(ratio, len(pairs), options)
        ctx.cache["tokens"] = len(pairs)
        models = ctx.cache.setdefault("models", {})
        models["binoculars"] = {"observer": {"name": observer.name, "version": observer.version},
                                "performer": {"name": performer.name, "version": performer.version}}
        evidence = [Evidence(self.name, "number", f"ratio={ratio:.3f}"),
                    Evidence(self.name, "number", f"tokens={len(pairs)}")]
        return Signal(self.name, score=score, confidence=confidence), evidence


register(_Binoculars())
