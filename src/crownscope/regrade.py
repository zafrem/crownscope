"""Re-grading earlier cluster members (SRS FR-5.6, supports AC-1).

A document is scored when it arrives, before the documents that will turn out to be its duplicates
exist, so the first members of a bulk cluster get low grades. When a new document matches earlier ones
(near-duplicate, same layout, or semantically close), each of those is re-graded with its updated counts.

No text is kept, so a re-grade rebuilds the score from what the store holds: the non-relational signals
computed at arrival (`Entry.signals`) plus the relational signals recomputed from the counts. It goes
through the same `Detector.aggregate` as scoring, so both always apply the same rules.

Counts only grow while a document is in the window: a document's grade is raised when later documents
confirm it, and is not lowered when those documents leave the window.
"""
from __future__ import annotations

from collections import defaultdict

from .extractors import embed_sim, near_dup, novelty, structure
from .result import Signal

RELATIONAL = ("near_dup", "structure", "embed_sim", "novelty")


def _available(entry, name: str) -> bool:
    if name == "novelty":
        return True  # whether it can be measured depends on the amount of text; see novelty.score_from_state
    return {"near_dup": entry.minhash, "structure": entry.structure, "embed_sim": entry.embedding}[name] is not None


def _relational_score(entry, name: str, store_size: int, options: dict):
    """(score, confidence) of a relational signal from the entry's current counts, or None if unavailable."""
    if not options or not _available(entry, name):
        return None
    if name == "near_dup":
        return near_dup.score_from_count(entry.matches, store_size, options)
    if name == "structure":
        return structure.score_from_count(entry.repeats, store_size, options)
    if name == "embed_sim":
        return embed_sim.score_from_stats(entry.similar, entry.similar_sum, store_size, options)
    return novelty.score_from_state(len(entry.units), entry.shared_units, entry.unique_facts, entry.chars or 0,
                                    store_size, options)


def _signals(pack, entry, store_size: int) -> list[Signal]:
    """The entry's signals as they stand now."""
    surface = {name: (score, conf) for name, score, conf in entry.signals}
    out = []
    for name in pack.signals:
        got = (_relational_score(entry, name, store_size, pack.options.get(name)) if name in RELATIONAL
               else surface.get(name))
        out.append(Signal(name, score=got[0], confidence=got[1]) if got
                   else Signal(name, status="unavailable", reason="not available for this document"))
    return out


def _update_evidence(entry, signal: str, numbers: dict, new_id: str, cap: int, keep_max: str | None = None,
                     similarity: float | None = None) -> None:
    """Refresh the relational evidence of `entry` for `signal` after a new document `new_id` confirmed it.

    `numbers` maps evidence keys to their new text (e.g. {"near_duplicates": "near_duplicates=5"}).
    `keep_max` names a numeric item that keeps the highest value seen (max_similarity).
    """
    keep, best = [], similarity
    for ev in entry.evidence:
        sig, kind, value = ev
        key = value.split("=", 1)[0]
        if sig == signal and kind == "number" and key in numbers:
            continue
        if sig == signal and kind == "number" and keep_max and key == keep_max:
            best = max(best or 0.0, float(value.split("=", 1)[1]))
            continue
        keep.append(ev)
    ids = [v for s, k, v in keep if s == signal and k == "similar_document"]
    if new_id not in ids and len(ids) < cap:
        keep.append((signal, "similar_document", new_id))
    keep += [(signal, "number", text) for text in numbers.values()]
    if keep_max and best is not None:
        keep.append((signal, "number", f"{keep_max}={best:.2f}"))
    entry.evidence = tuple(keep)


def regrade_members(detector, store, new_id: str, matches: list, repeats: list, semantic: list = (),
                    scan=None) -> list[dict]:
    """Raise the counts of every document that `new_id` matched and re-grade those whose score changes.

    `matches` is [(doc_id, similarity)] of near-duplicates, `repeats` the doc_ids sharing the structure
    fingerprint, `semantic` [(doc_id, cosine)] of semantic neighbours, `scan` the new document's novelty state
    (lexical and semantic marks). Returns one {id, from, to} per document whose grade or score changed.
    """
    pack = detector.pack
    delta = defaultdict(lambda: {"near_dup": 0, "structure": 0, "embed_sim": 0, "marks": 0, "common": 0})
    sims = {"near_dup": {}, "embed_sim": {}}
    for doc_id, sim in matches:
        delta[doc_id]["near_dup"] += 1
        sims["near_dup"][doc_id] = sim
    for doc_id in repeats:
        delta[doc_id]["structure"] += 1
    for doc_id, sim in semantic:
        delta[doc_id]["embed_sim"] += 1
        sims["embed_sim"][doc_id] = sim
    if scan is not None:
        for doc_id, mask in scan.marks.items():
            delta[doc_id]["marks"] |= mask
        for doc_id, n in scan.lex.newly_common.items():
            delta[doc_id]["common"] += n

    changes = []
    for doc_id, d in delta.items():
        if doc_id not in store or doc_id == new_id:
            continue  # evicted meanwhile
        e = store.get(doc_id)
        before = {n: _relational_score(e, n, store.size, pack.options.get(n)) for n in RELATIONAL}
        e.matches += d["near_dup"]
        e.repeats += d["structure"]
        e.similar += d["embed_sim"]
        e.similar_sum += sims["embed_sim"].get(doc_id, 0.0)
        new_bits = d["marks"] & ~e.shared_mask
        e.shared_mask |= d["marks"]
        e.unique_facts -= d["common"]
        if e.signals is None:
            continue  # saved before re-grading existed
        after = {n: _relational_score(e, n, store.size, pack.options.get(n)) for n in RELATIONAL}
        if before == after:
            continue  # saturated: the extra match changes nothing
        if d["near_dup"] and "near_dup" in pack.signals:
            _update_evidence(e, "near_dup", {"near_duplicates": f"near_duplicates={e.matches}"}, new_id,
                             pack.options["near_dup"]["evidence_cap"], "max_similarity", sims["near_dup"].get(doc_id))
        if d["structure"] and "structure" in pack.signals:
            _update_evidence(e, "structure", {"structure_repeats": f"structure_repeats={e.repeats} window={store.size}"},
                             new_id, pack.options["structure"]["evidence_cap"])
        if d["embed_sim"] and "embed_sim" in pack.signals:
            _update_evidence(e, "embed_sim", {"semantic_neighbors": f"semantic_neighbors={e.similar}",
                                              "mean_similarity": f"mean_similarity={e.similar_sum / e.similar:.2f}"},
                             new_id, pack.options["embed_sim"]["evidence_cap"])
        if (new_bits or d["common"]) and "novelty" in pack.signals:
            _update_evidence(e, "novelty", {"reused_sentences": f"reused_sentences={e.shared_units}/{len(e.units)}",
                                            "unique_facts": f"unique_facts={e.unique_facts}/{len(e.facts)}"},
                             new_id, pack.options["novelty"]["evidence_cap"])
        grade, score, _ = detector.aggregate(_signals(pack, e, store.size), e.chars, e.chars < pack.min_length)
        if (grade, score) != (e.grade, e.score):
            changes.append({"id": doc_id, "from": {"grade": e.grade, "score": e.score},
                            "to": {"grade": grade, "score": score}})
            e.grade, e.score = grade, score
    return changes
