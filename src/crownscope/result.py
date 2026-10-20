"""Result schema (SRS 5.2). All fields are JSON-serializable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

GRADES = ("low", "medium", "high", "insufficient_length")


@dataclass
class Signal:
    name: str
    status: str = "ok"  # ok | unavailable
    score: float | None = None
    confidence: float | None = None
    weight: float | None = None
    contribution: float | None = None
    reason: str | None = None  # set when unavailable


@dataclass
class Evidence:
    signal: str
    type: str  # phrase | fragment | similar_document | number
    value: str
    location: tuple[int, int] | None = None


@dataclass
class Result:
    grade: str
    mode: str = "single"
    score: float | None = None
    confidence: float = 0.0
    signals: list[Signal] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    length: dict = field(default_factory=dict)
    cluster: dict | None = None
    pack: dict = field(default_factory=dict)
    # Cumulative Mode: earlier documents whose grade changed because of this one (see regrade.py)
    regraded: list = field(default_factory=list)
    # Models that produced signals for this result, for reproducibility (NFR-9): {signal: {name, dim}}
    models: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
