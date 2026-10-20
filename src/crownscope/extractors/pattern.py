"""`pattern` signal (SRS FR-3.1): URLs, contact details, business names, promotional keywords and
sponsorship/advertising disclosure phrases. Rule based; every rule comes from the pack.

Pack inputs:
  lists.patterns     [{category, name, regex}]   e.g. url, phone
  lists.businesses   [str]                       business names
  lists.promotional  [str]                       promotional keywords
  lists.disclosure   [str]                       sponsorship / ad disclosure phrases
  options.pattern    category_weights, per_category_cap, confidence{base, per_category, max}
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..result import Evidence, Signal
from . import register, unavailable

PHRASE_LISTS = {"businesses": "business", "promotional": "promotional", "disclosure": "disclosure"}


@lru_cache(maxsize=256)
def _rx(pattern: str, flags: int = 0) -> re.Pattern:
    return re.compile(pattern, flags)


class _Pattern:
    name = "pattern"
    requires_model = False
    min_length = 0
    relational = False

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.pattern")

        hits: dict[str, list[Evidence]] = {}

        def add(category, kind, value, span):
            hits.setdefault(category, []).append(Evidence(self.name, kind, value, span))

        for rule in lists.get("patterns", []):
            for m in _rx(rule["regex"]).finditer(text):
                add(rule["category"], "phrase", m.group(0), (m.start(), m.end()))
        for key, category in PHRASE_LISTS.items():
            for phrase in lists.get(key, []):
                for m in _rx(re.escape(phrase), re.IGNORECASE).finditer(text):
                    add(category, "phrase", m.group(0), (m.start(), m.end()))

        weights, cap = options["category_weights"], options["per_category_cap"]
        score = min(1.0, sum(weights.get(c, 0) * min(len(ev), cap) for c, ev in hits.items()))
        conf = options["confidence"]
        confidence = min(conf["max"], conf["base"] + conf["per_category"] * len(hits))
        evidence = [e for ev in hits.values() for e in ev]
        return Signal(self.name, score=score, confidence=confidence), evidence


register(_Pattern())
