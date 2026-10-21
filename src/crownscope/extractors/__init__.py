"""Extractor interface and registry (SRS FR-3.11, IR-7).

An extractor reads its thresholds, phrase lists and weights from the pack (`options`, `lists`);
nothing tunable is hard-coded (FR-7.3). Evidence locations are offsets into the normalized text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

from ..result import Evidence, Signal

if TYPE_CHECKING:
    from ..store import Store


@dataclass
class Context:
    """Per-document context. `store` is None in Single Mode; relational extractors then report unavailable."""
    store: "Store | None" = None
    doc_id: str | None = None
    meta: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    embedder: object = None  # sentence embedder for embed_sim, None if not configured or not loadable
    embedder_error: str | None = None  # why there is none
    language_model: object = None  # scoring language model for ppl, None if not configured or not loadable
    language_model_error: str | None = None  # why there is none
    performer: object = None  # second language model for binoculars, None if not configured or not loadable
    performer_error: str | None = None  # why there is none

    def memo(self, key: str, fn: Callable):
        """Compute a value once per document (fingerprints are shared between extractor and session)."""
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]


class Extractor(Protocol):
    name: str
    requires_model: bool
    min_length: int
    relational: bool  # needs the window store (Cumulative Mode)

    def run(self, text: str, lists: dict, options: dict,
            ctx: Context | None = None) -> tuple[Signal, list[Evidence]]: ...


REGISTRY: dict[str, Extractor] = {}


def register(extractor: Extractor) -> Extractor:
    REGISTRY[extractor.name] = extractor
    return extractor


def unavailable(name: str, reason: str) -> tuple[Signal, list[Evidence]]:
    """A signal that cannot run is excluded from aggregation, never scored as zero (FR-3.12)."""
    return Signal(name=name, status="unavailable", reason=reason), []


def ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0..1 suspicion between `lo` (0) and `hi` (1). `lo > hi` inverts the direction."""
    if lo == hi:
        return 0.0
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


from . import binoculars, burst, embed_sim, fast_detect, near_dup, novelty, pattern, ppl, structure, style  # noqa: E402,F401  (register built-ins)
