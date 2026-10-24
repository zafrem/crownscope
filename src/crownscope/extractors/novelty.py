"""`novelty` signal (SRS FR-3.10): how much of a document is already present in other documents in the window,
and whether it carries any facts of its own.

Two parts, both measured against the window store:
  reuse  share of the document's content sentences that also occur in some other document. `near_dup` asks
         whether the whole document is a copy; this asks how much of it is borrowed, which also catches
         documents stitched together from sentences used elsewhere.
  facts  document-specific facts per 1000 characters. A fact is a span matched by the pack's concrete-detail
         rules (numbers with units, usage periods, proper nouns); it is document-specific when no other
         document in the window mentions it. Bulk text typically has none.

The score is suspicion, as for every signal: high means little that is new. Only hashes are kept, never text.
Reuse is lexical by default. With `semantic.enabled` and an embedding model (the SRS's "optional model"), a
sentence also counts as reused when another document has a sentence that means the same thing, which catches
paraphrased bulk content. A sentence reused in both ways counts once. Without a model the signal falls back to
the lexical reading and says so in its `reason`; it never becomes unavailable for that.

Pack inputs, options.novelty:
  min_unit_chars, min_units, min_chars, full_confidence_chars, reuse{lo, hi}, facts{rules, lo, hi},
  mix{reuse, facts}, max_confidence, reference_store_size, evidence_cap,
  semantic{enabled, model, min_similarity, max_sentences, lsh}
  lists.concrete supplies the regexes named by facts.rules.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import NamedTuple

from ..embedding import normalize as unit_vector, quantize
from ..result import Evidence, Signal
from . import ramp, register, unavailable
from .pattern import _rx

WORD = re.compile(r"\w+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])(?![」』）)”’\"'])\s*|\n+")


def _hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def features(text: str, lists: dict, options: dict):
    """(unit hashes, {hash: span of its first occurrence}, fact hashes), all in a deterministic order."""
    spans: dict[int, tuple[int, int] | None] = {}
    cursor = 0
    for piece in SENTENCE_SPLIT.split(text):
        piece = piece.strip()
        if not piece:
            continue
        i = text.find(piece, cursor)
        if i >= 0:
            cursor = i + len(piece)
        norm = " ".join(WORD.findall(piece)).lower()
        if len(norm) < options["min_unit_chars"]:
            continue  # greetings and one-word sentences are not content
        spans.setdefault(_hash(norm), (i, i + len(piece)) if i >= 0 else None)
    rules = {r["name"]: r["regex"] for r in lists.get("concrete", [])}
    facts = set()
    for name in options["facts"]["rules"]:
        for m in _rx(rules[name]).finditer(text):
            facts.add(_hash(re.sub(r"\s+", "", m.group(0)).lower()))
    return tuple(spans), spans, tuple(sorted(facts))


def score_from_state(n_units: int, shared: int, unique_facts: int, chars: int, store_size: int, o: dict):
    """(score, confidence) from counts, or None when neither part can be measured. Used by re-grading too."""
    parts = {}
    if n_units >= o["min_units"]:
        parts["reuse"] = ramp(shared / n_units, o["reuse"]["lo"], o["reuse"]["hi"])
    if chars >= o["min_chars"]:
        parts["facts"] = ramp(unique_facts * 1000 / chars, o["facts"]["lo"], o["facts"]["hi"])
    if not parts:
        return None
    total = sum(o["mix"][k] for k in parts)
    score = sum(o["mix"][k] * v for k, v in parts.items()) / total if total else 0.0
    # How sure we can be: longer text says more, and "nothing reused" only means something if the window is populated.
    populated = 1.0 if shared else min(1.0, store_size / o["reference_store_size"])
    confidence = o["max_confidence"] * min(1.0, chars / o["full_confidence_chars"]) * (populated if "reuse" in parts else 1.0)
    return score, confidence


class State(NamedTuple):
    """Everything known about one document's novelty, computed once and shared by the extractor and the session."""
    units: tuple
    spans: dict
    facts: tuple
    lex: object  # NoveltyScan
    sem: object  # SemanticScan or None
    vectors: list  # int8 vectors of the first sentences (empty without the semantic part)
    mask: int  # bit i: sentence i is reused (lexically or semantically)
    marks: dict  # doc_id -> mask of that document's sentences this one now confirms as shared
    holders: Counter  # documents the reused sentences come from
    semantic_error: str | None  # why the semantic part was skipped, if it was wanted


def _semantic_wanted(options: dict) -> bool:
    return bool(options.get("semantic", {}).get("enabled"))


def sentence_vectors(ctx, text: str, units: tuple, spans: dict, sem: dict) -> list[bytes]:
    """int8 vectors of the first `max_sentences` content sentences, in document order."""
    chosen = list(units)[:sem["max_sentences"]]
    texts = [text[spans[h][0]:spans[h][1]] for h in chosen]
    vecs = [unit_vector(v) for v in ctx.embedder.embed(texts)]
    if len(vecs) != len(texts) or any(len(v) != ctx.embedder.dim for v in vecs):
        raise ValueError(f"embedder returned {len(vecs)} vectors for {len(texts)} sentences")
    return [quantize(v) for v in vecs]


def get_state(ctx, text: str, lists: dict, options: dict) -> State:
    def compute():
        units, spans, facts = features(text, lists, options)
        lex = ctx.store.novelty_scan(units, facts, ctx.doc_id)
        vectors, sem_scan, error = [], None, None
        if _semantic_wanted(options) and units:
            if ctx.embedder is None:
                error = ctx.embedder_error or "no embedding model available"
            else:
                try:
                    vectors = sentence_vectors(ctx, text, units, spans, options["semantic"])
                    sem_scan = ctx.store.semantic_scan(vectors, options["semantic"]["min_similarity"], ctx.doc_id)
                except Exception as e:  # a model error must not take the document down (NFR-10)
                    vectors, sem_scan, error = [], None, f"embedding failed: {e}"
        mask = 0
        for i, h in enumerate(units):
            if h in lex.shared_hashes or (sem_scan and i < len(sem_scan.hits) and sem_scan.hits[i]):
                mask |= 1 << i
        marks = dict(lex.marks)
        for doc_id, m in (sem_scan.marks if sem_scan else {}).items():
            marks[doc_id] = marks.get(doc_id, 0) | m
        holders = Counter(lex.holders)
        if sem_scan:
            holders.update(sem_scan.holders)
        return State(units, spans, facts, lex, sem_scan, vectors, mask, marks, holders, error)
    return ctx.memo("novelty_state", compute)


class _Novelty:
    name = "novelty"
    requires_model = False
    min_length = 0
    relational = True

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.novelty")
        if ctx is None or ctx.store is None:
            return unavailable(self.name, "requires Cumulative Mode (no store)")
        st = get_state(ctx, text, lists, options)
        shared = st.mask.bit_count()
        got = score_from_state(len(st.units), shared, st.lex.unique, len(text), ctx.store.size, options)
        if got is None:
            return unavailable(self.name, f"too little text: needs {options['min_units']} content sentences "
                                          f"or {options['min_chars']} characters")
        evidence = []
        if st.units and shared:
            evidence.append(Evidence(self.name, "number", f"reused_sentences={shared}/{len(st.units)}"))
            shown = [h for i, h in enumerate(st.units) if st.mask >> i & 1][:options["evidence_cap"]]
            evidence += [Evidence(self.name, "fragment", text[st.spans[h][0]:st.spans[h][1]], st.spans[h])
                         for h in shown if st.spans.get(h)]
            evidence += [Evidence(self.name, "similar_document", d)
                         for d, _ in st.holders.most_common(options["evidence_cap"])]
        if len(text) >= options["min_chars"]:
            evidence.append(Evidence(self.name, "number", f"unique_facts={st.lex.unique}/{len(st.facts)}"))
        reason = f"semantic reuse skipped, lexical only: {st.semantic_error}" if st.semantic_error else None
        return Signal(self.name, score=got[0], confidence=got[1], reason=reason), evidence


register(_Novelty())
