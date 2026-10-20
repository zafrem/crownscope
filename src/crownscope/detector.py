"""Single Mode pipeline (SRS 3.1, 4.4): normalize, route by length, run extractors, aggregate."""
from __future__ import annotations

from pathlib import Path

from . import extractors
from .extractors import Context
from .embedding import load_embedder
from .languagemodel import load_language_model
from .normalize import normalize
from .pack import LM_SIGNALS, Pack, load_pack
from .result import Result, Signal


def _band(length: int) -> str:
    return "short" if length < 1000 else "medium" if length < 2000 else "long"


class Detector:
    def __init__(self, pack: str | Path | Pack = "blog", embedder=None, language_model=None):
        """`embedder` supplies the embed_sim model directly; otherwise the pack's options.embed_sim.model is
        loaded the first time it is needed (nothing is loaded unless the pack enables embed_sim)."""
        self.pack = pack if isinstance(pack, Pack) else load_pack(pack)
        self._embedder = embedder
        self._embedder_error = None
        self._embedder_resolved = embedder is not None
        self._lm = language_model
        self._lm_error = None
        self._lm_resolved = language_model is not None

    def _resolve_embedder(self) -> None:
        if not self._embedder_resolved:
            self._embedder_resolved = True
            nsem = self.pack.options.get("novelty", {}).get("semantic", {})
            if "embed_sim" in self.pack.signals:
                spec = self.pack.options.get("embed_sim", {}).get("model")
            elif "novelty" in self.pack.signals and nsem.get("enabled"):
                spec = nsem.get("model")
            else:
                return  # nothing in this pack uses a model
            self._embedder, self._embedder_error = load_embedder(spec)

    @property
    def language_model(self):
        """The scoring model for `ppl`, `burst` and `fast_detect`, loaded on first use and only if the pack enables a signal that needs it."""
        if not self._lm_resolved:
            self._lm_resolved = True
            # the model signals share one model: the first enabled of LM_SIGNALS names it (the pack check makes them agree)
            owner = next((s for s in LM_SIGNALS if s in self.pack.signals), None)
            if owner:
                self._lm, self._lm_error = load_language_model(self.pack.options.get(owner, {}).get("model"))
        return self._lm

    @property
    def language_model_error(self) -> str | None:
        self.language_model
        return self._lm_error

    @property
    def embedder(self):
        self._resolve_embedder()
        return self._embedder

    @property
    def embedder_error(self) -> str | None:
        self._resolve_embedder()
        return self._embedder_error

    def analyze(self, text: str, meta: dict | None = None) -> Result:
        """Single Mode: stateless, no relational signals (FR-4.1). Model signals (ppl) run here."""
        return self._analyze(text, meta, ctx=Context(language_model=self.language_model,
                                                     language_model_error=self.language_model_error))

    def session(self, window: dict | None = None, hash_only: bool = False):
        """Open Cumulative Mode. `window` overrides the pack's {count, hours} (FR-5.3);
        `hash_only` keeps no document text in the store (FR-5.10)."""
        from .session import Session
        return Session(self, window, hash_only=hash_only)

    def load_session(self, path, window: dict | None = None, hash_only: bool | None = None):
        """Restore a session saved with `Session.save`, using this detector's pack (IR-5)."""
        from .persist import load_session
        return load_session(self, path, window, hash_only)

    def _analyze(self, text: str, meta: dict | None, ctx: Context | None) -> Result:
        normalized = normalize(text or "")
        length = {"chars": len(normalized)}
        short = len(normalized) < self.pack.min_length
        # FR-2.2/2.3: below the threshold, no model signals; the grade stays insufficient_length
        # unless a relational signal applies (SRS 5.2).
        return self._run(normalized, length, short=short, ctx=ctx)

    def analyze_many(self, docs):
        for doc in docs:
            yield self.analyze(doc) if isinstance(doc, str) else self.analyze(**doc)

    def aggregate(self, signals: list[Signal], chars: int, short: bool) -> tuple[str, float | None, float]:
        """Combine signals into (grade, score, confidence) with the pack's weights (FR-6.1 to 6.4).

        Sets each signal's weight and contribution. Shared by scoring and by re-grading, so both
        always apply the same rules.
        """
        weights = self.pack.weights[_band(chars)]
        for s in signals:
            s.weight = weights.get(s.name)
        usable = [s for s in signals if s.status == "ok" and s.weight]
        total = sum(s.weight for s in usable)
        score = confidence = None
        if total:
            for s in usable:
                s.contribution = s.score * s.weight / total
            score = sum(s.contribution for s in usable)
            confidence = sum(s.confidence * s.weight for s in usable) / total
        t = self.pack.thresholds
        # A short document keeps insufficient_length (and no score, SRS 5.2) unless a relational
        # signal fired, e.g. it belongs to a cluster of near-duplicates (FR-5.6).
        relational_hit = any(
            s.score and extractors.REGISTRY[s.name].relational for s in usable if s.name in extractors.REGISTRY
        )
        if short and not relational_hit:
            return "insufficient_length", None, confidence or 0.0
        grade = "low" if score is None or score < t["medium"] else "medium" if score < t["high"] else "high"
        return grade, score, confidence or 0.0

    def _run(self, text, length, short: bool, ctx: Context | None) -> Result:
        signals, evidence = [], []
        for name in self.pack.signals:
            ext = extractors.REGISTRY.get(name)
            if ext is None:
                sig, ev = extractors.unavailable(name, "extractor not installed")
            elif ext.requires_model and short:
                sig, ev = extractors.unavailable(name, "text below minimum length")
            else:
                sig, ev = ext.run(text, self.pack.lists, self.pack.options.get(name, {}), ctx)
            signals.append(sig)
            evidence.extend(ev)

        grade, score, confidence = self.aggregate(signals, length["chars"], short)
        if ctx is not None and "tokens" in ctx.cache:  # FR-1.3: token count when a model scored the text
            length = {**length, "tokens": ctx.cache["tokens"]}
        return Result(grade=grade, score=score, confidence=confidence, signals=signals,
                      evidence=evidence, length=length, pack=self.pack.meta,
                      models=dict(ctx.cache.get("models", {})) if ctx is not None else {},
                      mode="cumulative" if ctx and ctx.store else "single")
