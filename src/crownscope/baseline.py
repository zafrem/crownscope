"""Population baseline for authors (SRS FR-5.8).

An author's output is compared with the other authors in the window, not only with their own history:
someone who always posts at a machine pace is not normal just because they are consistent.

Per author, over the documents still in the window:
  rate_per_hour    (documents - 1) / span in hours, span floored at the pack's cluster.min_span_hours
  clustered_share  share of fingerprinted documents that sit in a near-duplicate cluster of two or more
  templated_share  share of documents whose structure fingerprint appears on at least one other document

Each metric is compared with the population by a robust z-score (median and MAD, so a few extreme
authors cannot distort the baseline) and a percentile. Only the high side counts: posting faster or
more templated than the population is suspicious, the reverse is not. The pack sets the MAD floor per
metric (so a population that is mostly zeros does not make every nonzero value look infinite),
the z cutoff and the minimum population size. Reporting only; it does not change document scores.
"""
from __future__ import annotations

import math
import statistics

METRICS = ("rate_per_hour", "clustered_share", "templated_share")
LOG_METRICS = {"rate_per_hour"}  # heavy-tailed: compared on a log1p scale
MAD_TO_SIGMA = 0.6745  # robust z = 0.6745 * (x - median) / MAD


def _transform(metric: str, v: float) -> float:
    return math.log1p(v) if metric in LOG_METRICS else v


def robust_z(x: float, median: float, mad: float, floor: float) -> float:
    return MAD_TO_SIGMA * (x - median) / max(mad, floor)


def percentile(x: float, population: list[float]) -> float:
    """Midrank percentile in [0, 1]: share of the population below x, counting ties as half."""
    below = sum(v < x for v in population)
    equal = sum(v == x for v in population)
    return (below + equal / 2) / len(population)


def author_metrics(docs, cluster_sizes: dict, structure_counts: dict, min_span_hours: float) -> dict:
    ts = [d.ts for d in docs]
    span = (max(ts) - min(ts)) / 3600
    fingerprinted = [d for d in docs if d.cluster is not None]
    structured = [d for d in docs if d.structure is not None]
    return {
        "rate_per_hour": (len(docs) - 1) / max(span, min_span_hours) if len(docs) > 1 else 0.0,
        "clustered_share": (sum(cluster_sizes[d.cluster] > 1 for d in fingerprinted) / len(fingerprinted)
                            if fingerprinted else None),
        "templated_share": (sum(structure_counts[d.structure] > 1 for d in structured) / len(structured)
                            if structured else None),
    }


def build_baseline(entries, cluster_sizes: dict, structure_counts: dict, options: dict,
                   min_span_hours: float, by: str = "author") -> dict:
    if not options:
        return {"by": by, "status": "not_configured", "population": 0, "baseline": {}, "groups": []}

    groups: dict = {}
    unattributed = 0
    for e in entries:
        key = getattr(e, by)
        if key is None:
            unattributed += 1
        else:
            groups.setdefault(key, []).append(e)

    metrics = {k: author_metrics(docs, cluster_sizes, structure_counts, min_span_hours)
               for k, docs in groups.items() if len(docs) >= options["min_docs"]}
    population = len(metrics)
    status = "ok" if population >= options["min_population"] else "insufficient_population"

    baseline, values = {}, {}
    if status == "ok":
        for m in METRICS:
            vals = [v for g in metrics.values() if (v := g[m]) is not None]
            values[m] = vals
            if vals:
                t = [_transform(m, v) for v in vals]
                med = statistics.median(t)
                baseline[m] = {"median": med, "mad": statistics.median(abs(x - med) for x in t),
                               "population": len(vals)}

    rows = []
    for key, docs in groups.items():
        row = {"key": key, "documents": len(docs)}
        if key not in metrics:
            row["status"] = "too_few_documents"
        elif status != "ok":
            row.update(status=status, metrics=metrics[key])
        else:
            z, pct = {}, {}
            for m, v in metrics[key].items():
                if v is not None and m in baseline:
                    b = baseline[m]
                    z[m] = robust_z(_transform(m, v), b["median"], b["mad"], options["mad_floor"][m])
                    pct[m] = percentile(v, values[m])
            outliers = sorted((m for m, s in z.items() if s >= options["z_threshold"]), key=lambda m: -z[m])
            row.update(status="ok", metrics=metrics[key], z=z, percentile=pct, outliers=outliers,
                       max_z=max(z.values()) if z else None)
        rows.append(row)

    rows.sort(key=lambda r: (-(r.get("max_z") if r.get("max_z") is not None else -math.inf), str(r["key"])))
    return {"by": by, "status": status, "population": population, "unattributed": unattributed,
            "z_threshold": options["z_threshold"], "baseline": baseline, "groups": rows}
