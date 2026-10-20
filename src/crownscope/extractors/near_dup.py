"""`near_dup` signal (SRS FR-3.3): near-duplicate documents in the window, by MinHash similarity.

Compares against the whole-stream window, so splitting uploads across accounts does not hide a cluster
(FR-5.7). Evidence lists the similar document IDs (FR-3.13).

Pack inputs, options.near_dup:
  shingle_size, min_chars, permutations, bands, min_similarity, matches{lo, hi},
  max_confidence, reference_store_size, evidence_cap
"""
from __future__ import annotations

import hashlib
import re
import struct
from functools import lru_cache

from ..result import Evidence, Signal
from . import ramp, register, unavailable

_NON_WORD = re.compile(r"\W+")
MASK = (1 << 64) - 1


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


@lru_cache(maxsize=8)
def _params(permutations: int) -> tuple[tuple[int, int], ...]:
    """Deterministic (a, b) pairs for the hash family h_i(x) = (a_i * x + b_i) mod 2^64 (NFR-9)."""
    return tuple((_h64(b"a%d" % i) | 1, _h64(b"b%d" % i)) for i in range(permutations))


def minhash(text: str, shingle_size: int, permutations: int) -> bytes:
    """MinHash signature over character shingles (whitespace and punctuation removed; robust for Korean)."""
    flat = _NON_WORD.sub("", text).lower()
    shingles = {_h64(flat[i:i + shingle_size].encode("utf-8")) for i in range(max(1, len(flat) - shingle_size + 1))}
    slots = [min(((a * h + b) & MASK) >> 32 for h in shingles) for a, b in _params(permutations)]
    return struct.pack(f">{permutations}I", *slots)


def fingerprint(text: str, options: dict) -> bytes | None:
    if not options or len(_NON_WORD.sub("", text)) < options["min_chars"]:
        return None
    return minhash(text, options["shingle_size"], options["permutations"])


def score_from_count(count: int, store_size: int, options: dict) -> tuple[float, float]:
    """(score, confidence) for a document with `count` near-duplicates in a window of `store_size`.

    With matches the evidence is direct; with none, "unique" only means something if the window is populated.
    """
    m = options["matches"]
    full = options["max_confidence"]
    confidence = full if count else full * min(1.0, store_size / options["reference_store_size"])
    return ramp(count, m["lo"], m["hi"]), confidence


def find_matches(ctx, fp: bytes) -> list[tuple[str, float]]:
    """Near-duplicates of this document in the window, looked up once per document."""
    return ctx.memo("near_matches", lambda: ctx.store.near_duplicates(fp, exclude=ctx.doc_id))


class _NearDup:
    name = "near_dup"
    requires_model = False
    min_length = 0
    relational = True

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.near_dup")
        if ctx is None or ctx.store is None:
            return unavailable(self.name, "requires Cumulative Mode (no store)")
        fp = ctx.memo("minhash", lambda: fingerprint(text, options))
        if fp is None:
            return unavailable(self.name, "text too short to fingerprint")

        matches = find_matches(ctx, fp)
        score, confidence = score_from_count(len(matches), ctx.store.size, options)

        evidence = []
        if matches:
            evidence.append(Evidence(self.name, "number", f"near_duplicates={len(matches)}"))
            evidence += [Evidence(self.name, "similar_document", doc_id)
                         for doc_id, _ in matches[:options["evidence_cap"]]]
            evidence.append(Evidence(self.name, "number", f"max_similarity={matches[0][1]:.2f}"))
        return Signal(self.name, score=score, confidence=confidence), evidence


register(_NearDup())
