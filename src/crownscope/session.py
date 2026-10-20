"""Cumulative Mode session (SRS 3.1 steps 3-4, FR-5.1 to 5.3, 5.10).

`add()` runs the Single Mode pipeline with the window store available, then records the document's
fingerprints. The document text itself is never kept.
"""
from __future__ import annotations

from . import extractors
from .extractors import Context, embed_sim, near_dup, novelty, structure
from .baseline import build_baseline
from .embedding import pack_vector
from .normalize import normalize
from .regrade import _signals, regrade_members
from .report import build_report
from .result import Evidence, Result
from .store import Entry, Store


class Session:
    def __init__(self, detector, window: dict | None = None, clock=None, hash_only: bool = False):
        """`hash_only` keeps no document text: evidence quoting the text (phrases, fragments) is dropped
        from the per-document record, leaving feature names and document IDs (FR-5.10, NFR-7)."""
        self.detector = detector
        self.hash_only = hash_only
        pack = detector.pack
        kwargs = {"clock": clock} if clock else {}
        nd = pack.options.get("near_dup", {})
        if nd:
            kwargs.update(permutations=nd["permutations"], bands=nd["bands"], min_similarity=nd["min_similarity"])
        es = pack.options.get("embed_sim", {})
        if es and "embed_sim" in pack.signals:
            kwargs["embed_lsh"] = es["lsh"]
        nsem = pack.options.get("novelty", {}).get("semantic", {})
        if nsem.get("enabled") and "novelty" in pack.signals:
            kwargs["sentence_lsh"] = nsem["lsh"]
        self.store = Store(window if window is not None else pack.window, **kwargs)
        self._n = 0
        # Called as on_evict(doc_id, Result) with the final result of a document the window drops.
        self.on_evict = None
        self.store.on_evict = self._evicted

    def _evicted(self, entry: Entry) -> None:
        if self.on_evict:
            self.on_evict(entry.doc_id, self._result_for(entry))

    def _result_for(self, e: Entry) -> Result:
        """A document's result as it stands now: relational signals and grade reflect later arrivals."""
        pack = self.detector.pack
        try:
            cluster = self.store.cluster_stats(e.cluster, pack.cluster.get("min_span_hours")) if e.cluster else None
        except KeyError:  # its cluster was dropped with the window
            cluster = None
        common = dict(mode="cumulative", length={"chars": e.chars}, cluster=cluster, pack=pack.meta,
                      evidence=[Evidence(*ev) for ev in e.evidence])
        if e.signals is None:  # saved before re-grading data existed: grade and score only
            return Result(grade=e.grade, score=e.score, signals=[], **common)
        signals = _signals(pack, e, self.store.size)
        grade, score, confidence = self.detector.aggregate(signals, e.chars, e.chars < pack.min_length)
        return Result(grade=grade, score=score, confidence=confidence, signals=signals, **common)

    def current_results(self) -> dict[str, Result]:
        """Current result of every document still in the window, keyed by ID, in arrival order."""
        return {e.doc_id: self._result_for(e) for e in self.store.entries()}

    def add(self, doc, meta: dict | None = None):
        if isinstance(doc, dict):
            doc, meta = doc["text"], {**doc.get("meta", {}), **(meta or {})}
        meta = dict(meta or {})
        self._n += 1
        doc_id = str(meta.get("id") or f"doc_{self._n}")
        ts = self.store.now(meta)
        self.store.expire(ts)

        ctx = Context(store=self.store, doc_id=doc_id, meta=meta, embedder=self.detector.embedder,
                      embedder_error=self.detector.embedder_error, language_model=self.detector.language_model,
                      language_model_error=self.detector.language_model_error)
        result = self.detector._analyze(doc, meta, ctx)

        opts = self.detector.pack.options
        text = normalize(doc)
        minhash = ctx.memo("minhash", lambda: near_dup.fingerprint(text, opts.get("near_dup", {})))
        sfp = ctx.memo("structure", lambda: structure.fingerprint(text, opts.get("structure", {})))
        matches = near_dup.find_matches(ctx, minhash) if minhash is not None else []
        repeats = structure.find_repeats(ctx, sfp) if sfp is not None else []
        es = opts.get("embed_sim", {})
        vec = (embed_sim.get_embedding(ctx, text, es)
               if "embed_sim" in self.detector.pack.signals and embed_sim.embeddable(ctx, text, es) else None)
        semantic = embed_sim.find_neighbors(ctx, vec, es) if vec is not None else []
        nv = opts.get("novelty", {})
        if "novelty" in self.detector.pack.signals and nv:
            scan = novelty.get_state(ctx, text, self.detector.pack.lists, nv)
            units, facts = scan.units, scan.facts
        else:
            units = facts = ()
            scan = None
        entry = Entry(
            doc_id=doc_id, ts=ts, minhash=minhash, structure=sfp,
            author=meta.get("author"), source=meta.get("source"), product=meta.get("product"),
            grade=result.grade, score=result.score,
            evidence=tuple((e.signal, e.type, e.value) for e in result.evidence
                           if not self.hash_only or e.type in ("number", "similar_document")),
            chars=result.length["chars"],
            signals=tuple((s.name, s.score, s.confidence) for s in result.signals
                          if s.status == "ok" and not extractors.REGISTRY[s.name].relational),
            matches=len(matches), repeats=len(repeats),
            embedding=pack_vector(vec) if vec is not None else None,
            similar=len(semantic), similar_sum=sum(s for _, s in semantic),
            units=units, facts=facts, shared_mask=scan.mask if scan else 0, unique_facts=scan.lex.unique if scan else 0,
            sentence_vectors=b"".join(scan.vectors) if scan and scan.vectors else None,
            sentence_dim=len(scan.vectors[0]) if scan and scan.vectors else 0,
        )
        if vec is not None:
            result.models["embed_sim"] = {"name": ctx.embedder.name, "dim": ctx.embedder.dim}
        if scan is not None and scan.vectors:
            result.models["novelty"] = {"name": ctx.embedder.name, "dim": ctx.embedder.dim}
        self.store.add(entry, [d for d, _ in matches])
        self.store.expire(ts)  # evict first, so the cluster reports only what is still inside the window
        if entry.doc_id in self.store and entry.cluster is not None:
            result.cluster = self.store.cluster_stats(entry.cluster, self.detector.pack.cluster.get("min_span_hours"))
        # Earlier documents this one confirms as duplicates may now deserve a higher grade (FR-5.6).
        result.regraded = regrade_members(self.detector, self.store, doc_id, matches, repeats, semantic, scan)
        return result

    def record(self, doc_id: str) -> dict | None:
        """The document's current grade and score, which may have been raised since `add` returned."""
        if doc_id not in self.store:
            return None
        e = self.store.get(doc_id)
        return {"id": doc_id, "grade": e.grade, "score": e.score, "cluster": e.cluster}

    def report(self, by: str = "all", flag_from: str = "medium", top_evidence: int = 5) -> dict:
        """Grouped report over the current window. `by`: "author" | "source" | "product" | "all" (IR-4)."""
        return build_report(self.store.entries(), by, flag_from, top_evidence)

    def save(self, path, include_evidence: bool = False) -> None:
        """Persist the store (IR-5). Evidence that quotes document text is left out unless `include_evidence`."""
        from .persist import save_session
        save_session(self, path, include_evidence)

    def baseline(self, by: str = "author") -> dict:
        """Compare each author (or source, product) with the rest of the population in the window (FR-5.8)."""
        if by not in ("author", "source", "product"):
            raise ValueError(f"by must be one of author, source, product, got {by!r}")
        pack = self.detector.pack
        return build_baseline(self.store.entries(), self.store.cluster_sizes(), self.store.structure_counts(),
                              pack.baseline, pack.cluster.get("min_span_hours", 0.05), by)
