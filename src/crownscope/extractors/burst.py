"""`burst` signal (SRS FR-3.7): how much sentence-level perplexity varies across the document.

People write unevenly: a plain sentence, then a surprising one. Machine text tends to stay at one level of
predictability. So low variation raises suspicion. The statistic is the standard deviation, over sentences, of each
sentence's mean token log-probability (in nats); the score is where it falls between the pack's `human` (high) and
`machine` (low) reference values. Those values depend on the model and the language, so the packs' numbers are
placeholders to be fitted.

Stateless, so it works in Single Mode. It shares the model scoring with `ppl` (the document is scored once when
both are enabled). Length-gated by the pipeline (500 characters, FR-2.2) and by `min_sentences` (10, FR-3.7).

Only sentences scored in full count: a sentence cut off at the end of the scored window is dropped, and so is any
sentence with fewer than `min_sentence_tokens` scored tokens, whose mean would be mostly noise. A sentence that
runs over a chunk boundary is scored in two parts with no shared context; both parts count toward the sentence.

Pack inputs, options.burst:
  model{provider, name, local_only, device, dtype} (only if `ppl` is not enabled; then ppl's model is used),
  max_tokens, max_chunks, min_sentences, min_sentence_tokens,
  spread{human, machine}, full_confidence_sentences, max_confidence
"""
from __future__ import annotations

import math
import re
import statistics

from ..result import Evidence, Signal
from . import ramp, register, unavailable
from .ppl import scored_tokens

SENTENCE = re.compile(r"[^\n]+?(?:[.!?…]+(?=\s|$)|[。！？]+[」』）)”’\"']*|(?=\n)|$)")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences (split like `style`: after . ! ? … and full-width 。！？ even without a space, and at line breaks)."""
    spans = []
    for m in SENTENCE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))
    return spans


def sentence_means(tokens: list[tuple], text: str, options: dict) -> list[tuple[int, int, float, int]]:
    """(start, end, mean log-probability, token count) for each sentence that was scored in full and has enough tokens."""
    if not tokens:
        return []
    scored_to = max(e for _, e, _ in tokens)
    spans = sentence_spans(text)
    if spans and scored_to < spans[-1][1]:  # the scoring window ended inside the last sentence(s)
        spans = [sp for sp in spans if sp[1] <= scored_to]
    out, i = [], 0
    for s, e in spans:
        while i < len(tokens) and tokens[i][0] < s:
            i += 1
        j = i
        while j < len(tokens) and tokens[j][0] < e:
            j += 1
        lps = [lp for _, _, lp in tokens[i:j]]
        i = j
        if len(lps) >= options["min_sentence_tokens"]:
            out.append((s, e, sum(lps) / len(lps), len(lps)))
    return out


def score_from_spread(spread: float, n_sentences: int, options: dict) -> tuple[float, float]:
    """(score, confidence). A smaller spread is more machine-like, so `human` (large) maps to 0 and `machine` to 1."""
    ref = options["spread"]
    score = ramp(spread, ref["human"], ref["machine"])
    confidence = options["max_confidence"] * min(1.0, n_sentences / options["full_confidence_sentences"])
    return score, confidence


class _Burst:
    name = "burst"
    requires_model = True  # so the pipeline skips it below min_length (FR-2.2)
    min_length = 500
    relational = False

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.burst")
        lm = getattr(ctx, "language_model", None)
        if lm is None:
            return unavailable(self.name, getattr(ctx, "language_model_error", None) or "no scoring language model available")
        try:
            tokens = scored_tokens(ctx, lm, text, options)
        except Exception as e:  # a model error must not take the document, or the batch, down (NFR-10)
            return unavailable(self.name, f"scoring failed: {e}")
        if not all(math.isfinite(lp) and lp <= 1e-6 for _, _, lp in tokens):
            return unavailable(self.name, "the model returned invalid log-probabilities")
        if any(s is None for s, _, _ in tokens):
            return unavailable(self.name, "the scoring model gives no token positions (it needs scored_tokens)")
        sents = sentence_means(tokens, text, options)
        if len(sents) < options["min_sentences"]:
            return unavailable(self.name, f"too few scored sentences: {len(sents)} < {options['min_sentences']}")

        means = [m for _, _, m, _ in sents]
        spread = statistics.pstdev(means)
        score, confidence = score_from_spread(spread, len(sents), options)
        ctx.cache["tokens"] = len(tokens)
        ctx.cache.setdefault("models", {})["burst"] = {"name": lm.name, "version": lm.version}
        evidence = [Evidence(self.name, "number", f"spread={spread:.3f}"),
                    Evidence(self.name, "number", f"sentences={len(sents)}"),
                    Evidence(self.name, "number", f"mean_logprob={statistics.fmean(means):.3f}")]
        return Signal(self.name, score=score, confidence=confidence), evidence


register(_Burst())
