"""`embed_sim` signal (SRS FR-3.5): semantic similarity to other documents in the window.

Finds documents whose sentence embedding is close (cosine >= min_similarity) to this one, which catches
paraphrased bulk content that shares few words and so escapes `near_dup`. Reports how many such documents
there are (cluster size) and how close they are (density). Needs the `embed` extra or an injected embedder;
without one the signal is `unavailable` with the reason, never zero (FR-3.12).

Pack inputs, options.embed_sim:
  model{provider, name, local_only}, min_chars, chunk_chars, max_chunks, min_similarity, lsh{tables, bits, seed},
  size{lo, hi}, density{lo, hi}, mix{size, density}, max_confidence, reference_store_size, evidence_cap
"""
from __future__ import annotations

from ..embedding import embed_document, pack_vector, unpack_vector
from ..result import Evidence, Signal
from . import ramp, register, unavailable


def score_from_stats(count: int, sim_sum: float, store_size: int, options: dict) -> tuple[float, float]:
    """(score, confidence) from the number of semantic neighbours and the sum of their similarities.

    score = mix.size * ramp(count) + mix.density * ramp(mean similarity); zero with no neighbours.
    """
    full = options["max_confidence"]
    if not count:
        return 0.0, full * min(1.0, store_size / options["reference_store_size"])
    mix, size, dens = options["mix"], options["size"], options["density"]
    total = mix["size"] + mix["density"]
    score = (mix["size"] * ramp(count, size["lo"], size["hi"])
             + mix["density"] * ramp(sim_sum / count, dens["lo"], dens["hi"])) / total
    return score, full


def get_embedding(ctx, text: str, options: dict):
    """This document's embedding, computed once; None if no model or the model failed (reason in the context)."""
    def compute():
        try:
            vec = embed_document(ctx.embedder, text, options["chunk_chars"], options["max_chunks"])
            # Round to the float32 that is stored, so the similarity of two documents is the same number
            # whichever of them arrived first (otherwise re-grading and arrival order differ by ~1e-8).
            return unpack_vector(pack_vector(vec))
        except Exception as e:  # a model error must not take the document, or the batch, down (NFR-10)
            ctx.cache["embedding_error"] = f"embedding failed: {e}"
            return None
    return ctx.memo("embedding", compute)


def embeddable(ctx, text: str, options: dict) -> bool:
    return bool(options) and ctx.embedder is not None and len(text) >= options["min_chars"]


def find_neighbors(ctx, vec, options: dict) -> list[tuple[str, float]]:
    return ctx.memo("embed_matches", lambda: ctx.store.near_embeddings(vec, options["min_similarity"], ctx.doc_id))


class _EmbedSim:
    name = "embed_sim"
    requires_model = False  # a different model from the length-gated probability signals: no 500-character rule
    min_length = 0
    relational = True

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.embed_sim")
        if ctx is None or ctx.store is None:
            return unavailable(self.name, "requires Cumulative Mode (no store)")
        if ctx.embedder is None:
            return unavailable(self.name, ctx.embedder_error or "no embedding model available")
        if len(text) < options["min_chars"]:
            return unavailable(self.name, "text too short to embed")
        vec = get_embedding(ctx, text, options)
        if vec is None:
            return unavailable(self.name, ctx.cache.get("embedding_error", "embedding failed"))

        neighbors = find_neighbors(ctx, vec, options)
        sims = [s for _, s in neighbors]
        score, confidence = score_from_stats(len(sims), sum(sims), ctx.store.size, options)
        evidence = []
        if neighbors:
            evidence.append(Evidence(self.name, "number", f"semantic_neighbors={len(neighbors)}"))
            evidence.append(Evidence(self.name, "number", f"mean_similarity={sum(sims) / len(sims):.2f}"))
            evidence += [Evidence(self.name, "similar_document", d) for d, _ in neighbors[:options["evidence_cap"]]]
        return Signal(self.name, score=score, confidence=confidence), evidence


register(_EmbedSim())
