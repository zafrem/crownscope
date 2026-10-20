"""`style` signal (SRS FR-3.2): model-free surface statistics of the writing.

Each feature is measured, mapped to a 0..1 suspicion with a pack-defined ramp (lo -> 0, hi -> 1),
and combined by pack-defined weights. A feature that cannot be measured (too few sentences or words)
is skipped, not counted as zero. Confidence grows with length (the signal is weighted down for short text).

Pack inputs:
  lists.boilerplate  {opening: [str], closing: [str]}
  lists.endings      {formal, polite, plain}  regexes matching sentence endings
  lists.concrete     [{name, regex}]           usage period, units, comparison, drawbacks, proper nouns
  lists.translationese [str]
  options.style      full_confidence_chars, max_confidence, evidence_min, ngram, mattr_window,
                     features {name: {weight, lo, hi}}
"""
from __future__ import annotations

import re
import statistics
from collections import Counter

from ..result import Evidence, Signal
from . import ramp, register, unavailable
from .pattern import _rx

WORD = re.compile(r"\w+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。])\s+|\n+")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐❤]")
LIST_LINE = re.compile(r"^\s*(?:[-*•▶●■◆✔✅]|\d+[.)])\s+|^#{1,6}\s", re.M)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def _strip(s: str) -> str:
    return " ".join(WORD.findall(s))


def _first_span(text: str, fragment: str) -> tuple[int, int] | None:
    i = text.find(fragment)
    return (i, i + len(fragment)) if i >= 0 else None


def _density(count: int, chars: int) -> float:
    return count * 1000 / max(chars, 1)


class _Style:
    name = "style"
    requires_model = False
    min_length = 0
    relational = False

    def run(self, text, lists, options, ctx=None):
        if not options:
            return unavailable(self.name, "pack has no options.style")
        tokens = WORD.findall(text)
        sentences = _sentences(text)
        if len(tokens) < 3:
            return unavailable(self.name, "too little text for style statistics")

        measured: dict[str, tuple[float, list[Evidence]]] = {}
        for feature, fn in (
            ("sentence_repeat", self._sentence_repeat),
            ("ngram_repeat", self._ngram_repeat),
            ("lexical_diversity", self._lexical_diversity),
            ("sentence_length_cv", self._sentence_length_cv),
            ("opening_skew", self._opening_skew),
            ("boilerplate", self._boilerplate),
            ("formatting", self._formatting),
            ("concreteness", self._concreteness),
            ("ending_mix", self._ending_mix),
            ("translationese", self._translationese),
        ):
            cfg = options["features"].get(feature)
            if not cfg:
                continue
            out = fn(text, tokens, sentences, lists, options)
            if out is not None:
                measured[feature] = out

        if not measured:
            return unavailable(self.name, "no style feature could be measured")

        feats = options["features"]
        total = sum(feats[f]["weight"] for f in measured)
        if not total:
            return unavailable(self.name, "all measured features have zero weight")
        score, evidence = 0.0, []
        for f, (value, ev) in measured.items():
            s = ramp(value, feats[f]["lo"], feats[f]["hi"])
            score += s * feats[f]["weight"] / total
            if s >= options["evidence_min"]:
                evidence.append(Evidence(self.name, "number", f"{f}={value:.3f}"))
                evidence.extend(ev)
        confidence = options["max_confidence"] * min(1.0, len(text) / options["full_confidence_chars"])
        return Signal(self.name, score=score, confidence=confidence), evidence

    # Each feature returns (value, evidence) or None when it cannot be measured.

    def _sentence_repeat(self, text, tokens, sentences, lists, o):
        if len(sentences) < 3:
            return None
        norm = [_strip(s) for s in sentences]
        counts = Counter(norm)
        value = 1 - len(counts) / len(norm)
        ev = [Evidence(self.name, "fragment", s, _first_span(text, sentences[norm.index(s)]))
              for s, n in counts.most_common(1) if n > 1]
        return value, ev

    def _ngram_repeat(self, text, tokens, sentences, lists, o):
        n = o["ngram"]
        if len(tokens) < n + 7:
            return None
        grams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(grams)
        value = 1 - len(counts) / len(grams)
        top, c = counts.most_common(1)[0]
        ev = [Evidence(self.name, "fragment", top, _first_span(text, top))] if c > 1 else []
        return value, ev

    def _lexical_diversity(self, text, tokens, sentences, lists, o):
        """Moving-average type/token ratio, so the value does not depend on text length."""
        if len(tokens) < 10:
            return None
        w = min(o["mattr_window"], len(tokens))
        counts: Counter = Counter(tokens[:w])
        ratios = [len(counts) / w]
        for i in range(w, len(tokens)):
            counts[tokens[i]] += 1
            out = tokens[i - w]
            counts[out] -= 1
            if not counts[out]:
                del counts[out]
            ratios.append(len(counts) / w)
        return sum(ratios) / len(ratios), []

    def _sentence_length_cv(self, text, tokens, sentences, lists, o):
        if len(sentences) < 3:
            return None
        lengths = [len(s) for s in sentences]
        return statistics.pstdev(lengths) / statistics.mean(lengths), []

    def _opening_skew(self, text, tokens, sentences, lists, o):
        if len(sentences) < 3:
            return None
        openings = [WORD.findall(s)[0] for s in sentences if WORD.search(s)]
        if len(openings) < 3:
            return None
        top, c = Counter(openings).most_common(1)[0]
        ev = [Evidence(self.name, "phrase", top, _first_span(text, top))] if c > 1 else []
        return c / len(openings), ev

    def _boilerplate(self, text, tokens, sentences, lists, o):
        bp = lists.get("boilerplate")
        if not bp or not sentences:
            return None
        ev, count = [], 0
        for part, sentence in (("opening", sentences[0]), ("closing", sentences[-1])):
            for phrase in bp.get(part, []):
                i = sentence.find(phrase)
                if i >= 0:
                    count += 1
                    ev.append(Evidence(self.name, "phrase", phrase, _first_span(text, phrase)))
                    break
        return float(count), ev

    def _formatting(self, text, tokens, sentences, lists, o):
        marks = len(EMOJI.findall(text)) + text.count("**") // 2 + len(LIST_LINE.findall(text))
        return _density(marks, len(text)), []

    def _concreteness(self, text, tokens, sentences, lists, o):
        """Specific details (numbers with units, usage period, comparisons, drawbacks) suggest a real author."""
        rules = lists.get("concrete")
        if not rules:
            return None
        count = sum(len(_rx(r["regex"]).findall(text)) for r in rules)
        return _density(count, len(text)), []

    def _ending_mix(self, text, tokens, sentences, lists, o):
        """Share of sentences not in the dominant ending register (formal / polite / plain)."""
        endings = lists.get("endings")
        if not endings:
            return None
        kinds = []
        for s in sentences:
            tail = s.rstrip(".!?…~ ")
            for kind in ("formal", "polite", "plain"):  # order matters: formal endings also end in 다
                if kind in endings and _rx(endings[kind]).search(tail):
                    kinds.append(kind)
                    break
        if len(kinds) < 3:
            return None
        return 1 - Counter(kinds).most_common(1)[0][1] / len(kinds), []

    def _translationese(self, text, tokens, sentences, lists, o):
        phrases = lists.get("translationese")
        if not phrases:
            return None
        ev = [Evidence(self.name, "phrase", p, _first_span(text, p)) for p in phrases if p in text]
        count = sum(text.count(p) for p in phrases)
        return _density(count, len(text)), ev


register(_Style())
