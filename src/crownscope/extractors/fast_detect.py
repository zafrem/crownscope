"""`fast_detect` signal (SRS FR-3.8): conditional probability curvature, after Fast-DetectGPT (Bao et al., 2023).

For each token the scoring model has a distribution over what could come next. Text a model would write tends to
sit above the average of that distribution (its tokens are higher-probability than a random draw would be); human
text sits nearer to or below it. The criterion is how far the document's log-probability is above its expected
value, in standard deviations:

    criterion = sum_t (log p(x_t) - E[log p]_t) / sqrt(sum_t Var[log p]_t)

with the expectation and variance taken over the model's own next-token distribution (the analytic form, so no
sampling). Here one model plays both roles (scoring and sampling); the paper's strongest results use a pair of
models. A higher criterion is more machine-like; the score is where it falls between the pack's `human` and
`machine` reference values. Those depend on the model and the language, so the packs' numbers are placeholders
to be fitted.

Stateless, so it works in Single Mode. It shares the model pass with `ppl` and `burst`. Length-gated by the
pipeline (500 characters, FR-2.2).

Pack inputs, options.fast_detect:
  model{provider, name, local_only, device, dtype} (only if neither ppl nor burst is enabled; then theirs is used),
  max_tokens, max_chunks, min_tokens, criterion{human, machine}, full_confidence_tokens, max_confidence
"""
from __future__ import annotations

import math

from ..result import Evidence, Signal
from . import ramp, register, unavailable


def criterion_from_curvature(stats: list[tuple[float, float, float]]) -> float | None:
    """(sum of logprob - expected logprob) / sqrt(sum of variance); None when there is no variance to divide by."""
    total_var = sum(v for _, _, v in stats)
    if total_var <= 0:
        return None
    return sum(lp - mu for lp, mu, _ in stats) / math.sqrt(total_var)


def score_from_criterion(criterion: float, n_tokens: int, options: dict) -> tuple[float, float]:
    """(score, confidence): the `human` criterion maps to 0 and the `machine` one to 1."""
    ref = options["criterion"]
    score = ramp(criterion, ref["human"], ref["machine"])
    confidence = options["max_confidence"] * min(1.0, n_tokens / options["full_confidence_tokens"])
    return score, confidence


class _FastDetect:
    name = "fast_detect"
    requires_model = True  # so the pipeline skips it below min_length (FR-2.2)
    min_length = 500
    relational = False

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.fast_detect")
        lm = getattr(ctx, "language_model", None)
        if lm is None:
            return unavailable(self.name, getattr(ctx, "language_model_error", None) or "no scoring language model available")
        if not hasattr(lm, "token_curvature"):
            return unavailable(self.name, "the scoring model does not give next-token distributions (it needs token_curvature)")
        try:
            stats = ctx.memo(("lm_curvature", options["max_tokens"], options["max_chunks"]),
                             lambda: lm.token_curvature(text, options["max_tokens"], options["max_chunks"]))
        except Exception as e:  # a model error must not take the document, or the batch, down (NFR-10)
            return unavailable(self.name, f"scoring failed: {e}")
        if len(stats) < options["min_tokens"]:
            return unavailable(self.name, f"too few tokens to score: {len(stats)} < {options['min_tokens']}")
        if not all(math.isfinite(x) for row in stats for x in row) or any(lp > 1e-6 or v < 0 for lp, _, v in stats):
            return unavailable(self.name, "the model returned invalid log-probabilities")
        criterion = criterion_from_curvature(stats)
        if criterion is None:
            return unavailable(self.name, "the model's next-token distributions have no variance")

        score, confidence = score_from_criterion(criterion, len(stats), options)
        ctx.cache["tokens"] = len(stats)
        ctx.cache.setdefault("models", {})["fast_detect"] = {"name": lm.name, "version": lm.version}
        evidence = [Evidence(self.name, "number", f"criterion={criterion:.3f}"),
                    Evidence(self.name, "number", f"tokens={len(stats)}")]
        return Signal(self.name, score=score, confidence=confidence), evidence


register(_FastDetect())
