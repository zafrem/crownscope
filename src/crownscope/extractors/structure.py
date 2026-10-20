"""`structure` signal (SRS FR-3.4): how often this document's layout repeats in the window.

Structure fingerprint = hash of heading count, paragraph count, and the opening and closing pattern
(first/last words of the first/last sentence, digits masked). Templated bulk output repeats it.

Pack inputs, options.structure:
  opening_tokens, closing_tokens, paragraph_bucket, repeats{lo, hi}, max_confidence,
  reference_store_size, evidence_cap
"""
from __future__ import annotations

import hashlib
import re

from ..result import Evidence, Signal
from . import ramp, register, unavailable

WORD = re.compile(r"\w+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。])\s+|\n+")
HEADING = re.compile(r"^(?:#{1,6}\s|\*\*[^*]+\*\*$|[■▶●◆]\s)")


def _shape(sentence: str, n: int, tail: bool = False) -> str:
    words = [re.sub(r"\d", "0", w.lower()) for w in WORD.findall(sentence)]
    return " ".join(words[-n:] if tail else words[:n])


def fingerprint(text: str, options: dict) -> str | None:
    if not options:
        return None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if not lines or not sentences:
        return None
    headings = sum(1 for ln in lines if HEADING.match(ln))
    paragraphs = (len(lines) - headings) // options["paragraph_bucket"]
    shape = (headings, paragraphs,
             _shape(sentences[0], options["opening_tokens"]),
             _shape(sentences[-1], options["closing_tokens"], tail=True))
    return hashlib.blake2b(repr(shape).encode("utf-8"), digest_size=8).hexdigest()


def score_from_count(count: int, store_size: int, options: dict) -> tuple[float, float]:
    r = options["repeats"]
    full = options["max_confidence"]
    confidence = full if count else full * min(1.0, store_size / options["reference_store_size"])
    return ramp(count, r["lo"], r["hi"]), confidence


def find_repeats(ctx, fp: str) -> list[str]:
    """Documents in the window with the same structure fingerprint, looked up once per document."""
    return ctx.memo("structure_repeats", lambda: ctx.store.same_structure(fp, exclude=ctx.doc_id))


class _Structure:
    name = "structure"
    requires_model = False
    min_length = 0
    relational = True

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.structure")
        if ctx is None or ctx.store is None:
            return unavailable(self.name, "requires Cumulative Mode (no store)")
        fp = ctx.memo("structure", lambda: fingerprint(text, options))
        if fp is None:
            return unavailable(self.name, "no text to fingerprint")

        same = find_repeats(ctx, fp)
        score, confidence = score_from_count(len(same), ctx.store.size, options)

        evidence = []
        if same:
            evidence.append(Evidence(self.name, "number", f"structure_repeats={len(same)} window={ctx.store.size}"))
            evidence += [Evidence(self.name, "similar_document", doc_id) for doc_id in same[:options["evidence_cap"]]]
        return Signal(self.name, score=score, confidence=confidence), evidence


register(_Structure())
