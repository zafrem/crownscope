"""Grouped reports over the window (SRS FR-5.5, IR-4).

Per group: document count, flagged count and share, dominant structure fingerprint, dominant cluster,
and the evidence that appears most often among the flagged documents.

"Flagged" means graded `medium` or higher by default (AC-1). Documents graded `insufficient_length`
count in the denominator and are never flagged.
"""
from __future__ import annotations

from collections import Counter

GROUP_FIELDS = ("author", "source", "product")
FLAG_GRADES = {"medium": ("medium", "high"), "high": ("high",)}
# Evidence that names another document is covered by the dominant cluster, not repeated in the ranking.
SKIP_EVIDENCE = {"similar_document"}


def _evidence_key(signal: str, kind: str, value: str) -> tuple[str, str, str]:
    # numeric evidence reads "feature=0.31"; count documents per feature, not per exact value
    return (signal, kind, value.split("=", 1)[0] if kind == "number" else value)


def _dominant(counter: Counter, total: int, name: str) -> dict | None:
    if not counter:
        return None
    key, count = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]
    return {name: key, "count": count, "share": count / total}


def build_report(entries, by: str = "all", flag_from: str = "medium", top_evidence: int = 5) -> dict:
    if by != "all" and by not in GROUP_FIELDS:
        raise ValueError(f"by must be 'all' or one of {GROUP_FIELDS}, got {by!r}")
    if flag_from not in FLAG_GRADES:
        raise ValueError(f"flag_from must be one of {tuple(FLAG_GRADES)}, got {flag_from!r}")
    flagged_grades = FLAG_GRADES[flag_from]

    groups: dict = {}
    for e in entries:
        groups.setdefault("all" if by == "all" else getattr(e, by), []).append(e)

    rows = []
    for key, docs in groups.items():
        flagged = [d for d in docs if d.grade in flagged_grades]
        evidence = Counter()
        for d in flagged:
            evidence.update({_evidence_key(*ev) for ev in d.evidence if ev[1] not in SKIP_EVIDENCE})
        n = len(docs)
        rows.append({
            "key": key,
            "documents": n,
            "flagged": len(flagged),
            "share_flagged": len(flagged) / n,
            "dominant_structure": _dominant(Counter(d.structure for d in docs if d.structure), n, "fingerprint"),
            "dominant_cluster": _dominant(Counter(d.cluster for d in docs if d.cluster), n, "id"),
            "top_evidence": [
                {"signal": s, "type": t, "value": v, "documents": c}
                for (s, t, v), c in sorted(evidence.items(), key=lambda kv: (-kv[1], kv[0]))[:top_evidence]
            ],
        })
    rows.sort(key=lambda r: (-r["flagged"], -r["share_flagged"], str(r["key"])))
    return {"by": by, "flag_from": flag_from, "documents": len(entries), "groups": rows}
