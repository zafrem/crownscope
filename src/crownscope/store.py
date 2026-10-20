"""Window store for Cumulative Mode (SRS 3.1, FR-5.1, 5.3, 5.10, NFR-3, NFR-4).

Holds fingerprints only: never the document text (hash-only storage, FR-5.10). Near-duplicate lookup
uses MinHash signatures with banded LSH, so a query never compares against every stored document:
a signature is `permutations` 32-bit minima, cut into `bands` bands; two documents become candidates if
any band matches exactly, and candidates are then verified by their estimated Jaccard similarity.
With 64 permutations and 16 bands, a pair at similarity 0.6 is found about 89% of the time and
an unrelated pair about 0.01%.
"""
from __future__ import annotations

import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, NamedTuple

from .embedding import EmbeddingIndex, SentenceIndex, unpack_vector

SLOT_BYTES = 4  # one 32-bit minimum per permutation


class NoveltyScan(NamedTuple):
    """What the window says about one document's sentences and facts (see Store.novelty_scan)."""
    shared: int  # content sentences that some other document also has
    unique: int  # facts no other document has
    shared_hashes: frozenset
    holders: Counter  # other documents holding shared sentences, by how many
    marks: dict  # doc_id -> bitmask of its sentences whose only other copy is this document's (now shared)
    newly_common: Counter  # documents whose unique fact this document now repeats


class SemanticScan(NamedTuple):
    """Sentence-level semantic lookup for one document (see Store.semantic_scan)."""
    hits: list  # per sentence of this document: does some other document have a close sentence?
    marks: dict  # doc_id -> bitmask of its sentences that are close to one of this document's
    holders: Counter


@dataclass
class _Cluster:
    """Connected group of near-duplicates. Time bounds are cached and recomputed only after an eviction."""
    founder_seq: int
    members: set
    first: float
    last: float
    dirty: bool = False


@dataclass
class Entry:
    doc_id: str
    ts: float
    minhash: bytes | None = None
    structure: str | None = None
    author: str | None = None
    source: str | None = None
    product: str | None = None
    cluster: str | None = None  # set by Store.add when the entry has a MinHash
    # Per-document result kept for reports (FR-5.1, FR-5.5); evicted with the entry.
    grade: str | None = None
    score: float | None = None
    evidence: tuple = ()  # ((signal, type, value), ...)
    # What a re-grade needs without the text (regrade.py). `signals` is None for entries saved before
    # these fields existed; those cannot be re-graded.
    chars: int | None = None
    signals: tuple | None = None  # ((name, score, confidence), ...) of the non-relational signals that ran
    matches: int = 0  # near-duplicates seen so far (at arrival and since)
    repeats: int = 0  # documents sharing the structure fingerprint seen so far
    # embed_sim: the unit vector (little-endian float32) and running neighbour statistics
    embedding: bytes | None = None
    similar: int = 0  # semantic neighbours seen so far
    similar_sum: float = 0.0  # sum of their cosine similarities
    # novelty: hashes of the document's content sentences and facts (never text), and how they stand in the window
    units: tuple = ()
    facts: tuple = ()
    shared_mask: int = 0  # bit i set: sentence i is present in another document (lexically or semantically), kept as a
    #                       high-water mark like the counts above. A mask, so two kinds of hit on one sentence count once.
    unique_facts: int = 0  # facts no other document mentions
    # semantic novelty: int8 vectors of the first sentences, back to back, `sentence_dim` bytes each
    sentence_vectors: bytes | None = None
    sentence_dim: int = 0

    @property
    def shared_units(self) -> int:
        return self.shared_mask.bit_count()


def parse_timestamp(value) -> float | None:
    """Epoch seconds (a number, or a string such as a CSV cell holding one), or an ISO-8601 string
    (a trailing Z is accepted). A bare number is always epoch seconds, never a year or a date."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        pass
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


class Store:
    def __init__(self, window: dict | None = None, permutations: int = 64, bands: int = 16,
                 min_similarity: float = 0.6, clock: Callable[[], float] = time.time,
                 embed_lsh: dict | None = None, sentence_lsh: dict | None = None):
        if permutations % bands:
            raise ValueError("bands must divide permutations")
        self.window = window or {}
        self.min_similarity = min_similarity
        self.clock = clock
        self._slots = permutations
        self._bands = bands
        self._band_bytes = permutations // bands * SLOT_BYTES
        self._entries: OrderedDict[str, Entry] = OrderedDict()
        self._buckets: dict[tuple[int, bytes], set[str]] = defaultdict(set)
        self._structures: dict[str, set[str]] = defaultdict(set)
        self._clusters: dict[str, _Cluster] = {}
        self._seq = 0
        # Semantic neighbour index (FR-3.5); LSH settings come from the pack, e.g. {"tables": 12, "bits": 8, "seed": 0}
        self._units: dict[int, set[str]] = defaultdict(set)  # sentence hash -> documents containing it
        self._facts: dict[int, set[str]] = defaultdict(set)  # fact hash -> documents mentioning it
        self._sentence_dim = 0  # bytes of one sentence vector, fixed by the first one
        self._sentences = SentenceIndex(self._sentence_vector, **(sentence_lsh or {}))
        self._embed_bytes = 0  # size of one stored embedding, fixed by the first one
        self._embeddings = EmbeddingIndex(lambda doc_id: self._entries[doc_id].embedding, **(embed_lsh or {}))
        self.on_evict: Callable[["Entry"], None] | None = None  # called for window evictions, not replacements

    def get(self, doc_id: str) -> "Entry":
        return self._entries[doc_id]

    def entries(self) -> list["Entry"]:
        return list(self._entries.values())

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._entries

    @property
    def size(self) -> int:
        return len(self._entries)

    def now(self, meta: dict | None = None) -> float:
        ts = parse_timestamp((meta or {}).get("timestamp"))
        return self.clock() if ts is None else ts

    def _keys(self, sig: bytes):
        w = self._band_bytes
        return [(i, sig[i * w:(i + 1) * w]) for i in range(self._bands)]

    def similarity(self, a: bytes, b: bytes) -> float:
        """Estimated Jaccard similarity: share of permutations whose minimum agrees."""
        same = sum(x == y for x, y in zip(memoryview(a).cast("I"), memoryview(b).cast("I")))
        return same / self._slots

    def add(self, entry: Entry, neighbors: tuple[str, ...] | list[str] = ()) -> None:
        """Store an entry. `neighbors` are the near-duplicates found for it; their clusters merge with its own."""
        if entry.doc_id in self._entries:
            self.remove(entry.doc_id)
        self._entries[entry.doc_id] = entry
        if entry.minhash is not None:
            self._join_cluster(entry, [n for n in neighbors if n in self._entries])
        self._index(entry)

    def _sentence_vector(self, key: str) -> bytes:
        doc_id, i = key.rsplit("#", 1)
        e = self._entries[doc_id]
        return e.sentence_vectors[int(i) * e.sentence_dim:(int(i) + 1) * e.sentence_dim]

    def _index(self, entry: Entry) -> None:
        if entry.sentence_vectors:
            d = entry.sentence_dim
            if d < 1 or len(entry.sentence_vectors) % d or (self._sentence_dim and d != self._sentence_dim):
                raise ValueError(f"{entry.doc_id}: sentence vectors of {d} bytes do not match the "
                                 f"{self._sentence_dim} bytes of the others (a different model?)")
            self._sentence_dim = d
            for i in range(len(entry.sentence_vectors) // d):
                self._sentences.add(f"{entry.doc_id}#{i}", entry.sentence_vectors[i * d:(i + 1) * d])
        for u in entry.units:
            self._units[u].add(entry.doc_id)
        for f in entry.facts:
            self._facts[f].add(entry.doc_id)
        if entry.embedding is not None:
            if len(entry.embedding) % 4 or (self._embed_bytes and len(entry.embedding) != self._embed_bytes):
                raise ValueError(f"{entry.doc_id}: embedding size {len(entry.embedding)} bytes does not match the "
                                 f"{self._embed_bytes} bytes of the others (a different model?)")
            self._embed_bytes = len(entry.embedding)
            self._embeddings.add(entry.doc_id, entry.embedding)
        if entry.minhash is not None:
            for key in self._keys(entry.minhash):
                self._buckets[key].add(entry.doc_id)
        if entry.structure is not None:
            self._structures[entry.structure].add(entry.doc_id)

    def cluster_state(self) -> dict:
        """What the entries alone cannot tell: cluster founding order and time bounds (for saving)."""
        out = {}
        for label in self._clusters:
            self.cluster_stats(label)  # refreshes cached bounds after evictions
            c = self._clusters[label]
            out[label] = {"founder_seq": c.founder_seq, "first": c.first, "last": c.last}
        return out

    @property
    def seq(self) -> int:
        return self._seq

    def restore(self, entries: list[Entry], clusters: dict[str, dict], seq: int) -> None:
        """Load saved state into an empty store, in the original insertion order. LSH buckets are rebuilt
        from the signatures, so the band layout may differ from the one the state was saved with."""
        if self._entries:
            raise ValueError("restore needs an empty store")
        for e in entries:
            if e.minhash is not None and len(e.minhash) != self._slots * SLOT_BYTES:
                raise ValueError(f"{e.doc_id}: signature length does not match {self._slots} permutations")
            self._entries[e.doc_id] = e
            self._index(e)
        members: dict[str, set[str]] = defaultdict(set)
        for e in entries:
            if e.cluster is not None:
                members[e.cluster].add(e.doc_id)
        if set(members) != set(clusters):
            raise ValueError("clusters do not match the entries that refer to them")
        for label, c in clusters.items():
            self._clusters[label] = _Cluster(c["founder_seq"], members[label], c["first"], c["last"])
        self._seq = seq

    def _join_cluster(self, entry: Entry, neighbors: list[str]) -> None:
        self._seq += 1
        labels = {self._entries[n].cluster for n in neighbors if self._entries[n].cluster}
        if not labels:
            entry.cluster = entry.doc_id
            self._clusters[entry.doc_id] = _Cluster(self._seq, {entry.doc_id}, entry.ts, entry.ts)
            return
        keep = min(labels, key=lambda lb: (self._clusters[lb].founder_seq, lb))  # oldest cluster survives
        target = self._clusters[keep]
        for other in labels - {keep}:
            c = self._clusters.pop(other)
            for member in c.members:
                self._entries[member].cluster = keep
            target.members |= c.members
            target.first, target.last = min(target.first, c.first), max(target.last, c.last)
            target.dirty = target.dirty or c.dirty
        entry.cluster = keep
        target.members.add(entry.doc_id)
        target.first, target.last = min(target.first, entry.ts), max(target.last, entry.ts)

    def cluster_stats(self, label: str, min_span_hours: float | None = None) -> dict:
        """Size and arrival rate of a cluster inside the window.

        Arrival rate = (size - 1) intervals / span in hours, with the span floored at `min_span_hours`
        so that a burst landing in one instant gives a large finite rate. None for a single document.
        """
        c = self._clusters[label]
        if c.dirty:
            ts = [self._entries[m].ts for m in c.members]
            c.first, c.last, c.dirty = min(ts), max(ts), False
        size, span = len(c.members), (c.last - c.first) / 3600
        rate = None
        if size > 1 and min_span_hours:
            rate = (size - 1) / max(span, min_span_hours)
        return {"id": label, "size": size, "arrival_rate_per_hour": rate,
                "span_hours": span, "first_seen": c.first, "last_seen": c.last}

    def remove(self, doc_id: str) -> None:
        entry = self._entries.pop(doc_id)
        for index, hashes in ((self._units, entry.units), (self._facts, entry.facts)):
            for h in hashes:
                holders = index[h]
                holders.discard(doc_id)
                if not holders:
                    del index[h]
        if entry.sentence_vectors:
            for i in range(len(entry.sentence_vectors) // entry.sentence_dim):
                self._sentences.remove(f"{doc_id}#{i}")
            if not len(self._sentences):
                self._sentence_dim = 0
        if entry.embedding is not None:
            self._embeddings.remove(doc_id)
            if not len(self._embeddings):
                self._embed_bytes = 0
        if entry.cluster is not None:
            c = self._clusters[entry.cluster]
            c.members.discard(doc_id)
            if not c.members:
                del self._clusters[entry.cluster]
            elif entry.ts in (c.first, c.last):
                c.dirty = True
        if entry.minhash is not None:
            for key in self._keys(entry.minhash):
                bucket = self._buckets[key]
                bucket.discard(doc_id)
                if not bucket:
                    del self._buckets[key]
        if entry.structure is not None:
            ids = self._structures[entry.structure]
            ids.discard(doc_id)
            if not ids:
                del self._structures[entry.structure]

    def expire(self, now: float) -> None:
        """Evict by count and by age (NFR-4). Oldest-inserted first."""
        count, hours = self.window.get("count"), self.window.get("hours")
        while self._entries:
            oldest = next(iter(self._entries.values()))
            too_many = count is not None and len(self._entries) > count
            too_old = hours is not None and now - oldest.ts > hours * 3600
            if not (too_many or too_old):
                break
            self.remove(oldest.doc_id)
            if self.on_evict:
                self.on_evict(oldest)

    def near_duplicates(self, minhash: bytes, exclude: str | None = None) -> list[tuple[str, float]]:
        """(doc_id, estimated similarity) of stored documents at or above `min_similarity`, most similar first."""
        candidates: set[str] = set()
        for key in self._keys(minhash):
            candidates |= self._buckets.get(key, set())
        found = []
        for doc_id in candidates - {exclude}:
            sim = self.similarity(minhash, self._entries[doc_id].minhash)
            if sim >= self.min_similarity:
                found.append((doc_id, sim))
        return sorted(found, key=lambda x: (-x[1], x[0]))

    def cluster_sizes(self) -> dict[str, int]:
        return {label: len(c.members) for label, c in self._clusters.items()}

    def novelty_scan(self, units, facts, exclude: str | None = None) -> NoveltyScan:
        """Look a document's sentence and fact hashes up in the window. Nothing is compared pairwise."""
        shared_hashes, holders, marks = set(), Counter(), {}
        for u in units:
            others = self._units.get(u, set()) - {exclude}
            if others:
                shared_hashes.add(u)
                holders.update(others)
                if len(others) == 1:  # with two or more holders the sentence was already marked shared in each
                    only = next(iter(others))
                    marks[only] = marks.get(only, 0) | (1 << self._entries[only].units.index(u))
        unique, newly_common = 0, Counter()
        for f in facts:
            others = self._facts.get(f, set()) - {exclude}
            if not others:
                unique += 1
            elif len(others) == 1:
                newly_common[next(iter(others))] += 1
        return NoveltyScan(len(shared_hashes), unique, frozenset(shared_hashes), holders, marks, newly_common)

    def semantic_scan(self, vectors: list[bytes], min_similarity: float, exclude: str | None = None) -> SemanticScan:
        """For each int8 sentence vector, look for close sentences in other documents (LSH, then exact check)."""
        hits, marks, holders = [], {}, Counter()
        for q in vectors:
            found = [(key, sim) for key, sim in self._sentences.query(q, min_similarity)
                     if key.rsplit("#", 1)[0] != exclude]
            hits.append(bool(found))
            for key, _ in found:
                doc_id, i = key.rsplit("#", 1)
                marks[doc_id] = marks.get(doc_id, 0) | (1 << int(i))
                holders[doc_id] += 1
        return SemanticScan(hits, marks, holders)

    def rebuild_lexical_masks(self) -> None:
        """Recompute which sentences are shared from the current window. For sessions saved before masks existed,
        which kept only a count; evicted documents' earlier hits cannot be recovered."""
        for e in self._entries.values():
            e.shared_mask = sum(1 << i for i, u in enumerate(e.units) if len(self._units[u]) > 1)

    def near_embeddings(self, vec, min_similarity: float, exclude: str | None = None) -> list[tuple[str, float]]:
        """(doc_id, cosine) of stored documents semantically close to `vec`, most similar first."""
        return self._embeddings.query(vec, min_similarity, exclude)

    def same_structure(self, fingerprint: str, exclude: str | None = None) -> list[str]:
        return sorted(self._structures.get(fingerprint, set()) - {exclude})

    def structure_counts(self) -> Counter:
        return Counter({fp: len(ids) for fp, ids in self._structures.items()})
