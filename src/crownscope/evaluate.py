"""`crownscope evaluate` (SRS 4.8 FR-8.1 to 8.4, 7.1 to 7.3, IR-10).

Scores a labeled file with a pack and reports how well it separates bulk-generated content from
ordinary human content.

Labels (SRS 7.1): `human` (ordinary content), `templated` (low-quality or templated content) and
`bulk` (bulk-generated). `bulk` is the positive class and `human` the negative; `templated` is
ambiguous, so by default it is reported but left out of precision and recall (--templated-as changes
that). Optional columns: `cluster` (id shared by the members of a bulk cluster) and `hard_negative`
(on human documents: one author's consistent style, repeated product reviews, formal prose).

"Flagged" means graded medium or higher (high with --flag-from high). A document with no score
(grade insufficient_length, no relational signal) is never flagged and ranks at 0 for ROC-AUC.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from . import analyze as A
from .detector import Detector
from .pack import PackError
from .persist import SessionFileError

LABELS = ("human", "templated", "bulk")
TRUE = {"1", "true", "yes", "y", "t"}
FALSE = {"0", "false", "no", "n", "f"}
FLAGGED = {"medium": ("medium", "high"), "high": ("high",)}
BANDS = (("lt_500", "under 500 characters"), ("500_1000", "500 to 1,000"), ("gt_1000", "over 1,000"))
# Signal groups for the ablation report (SRS 7.3)
GROUPS = {
    "pattern": ("pattern",),
    "style": ("style",),
    "relational": ("near_dup", "structure", "embed_sim", "novelty"),
    "model": ("ppl", "burst", "fast_detect", "binoculars"),
}
# Initial acceptance targets from SRS 7.2; placeholders until a baseline is measured.
TARGETS = {"AC-1": 0.80, "AC-2": 0.02, "AC-3": 0.90, "AC-5": 1.0, "AC-6": 1.0, "AC-8": 2.0}


def length_band(chars: int) -> str:
    return "lt_500" if chars < 500 else "500_1000" if chars <= 1000 else "gt_1000"


# ---------------------------------------------------------------- metrics (pure)

def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def roc_auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half). None if a class is empty."""
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mid = (i + 1 + j) / 2  # average of ranks i+1..j
        rank_sum += mid * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def confusion(rows: list[dict], flagged: tuple) -> dict:
    tp = sum(r["positive"] and r["grade"] in flagged for r in rows)
    fp = sum(not r["positive"] and r["grade"] in flagged for r in rows)
    fn = sum(r["positive"] and r["grade"] not in flagged for r in rows)
    tn = sum(not r["positive"] and r["grade"] not in flagged for r in rows)
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None  # nothing predicted, or nothing to find
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": ratio(fp, fp + tn)}


def group_metrics(rows: list[dict]) -> dict:
    """Metrics over rows that have a binary truth (`positive`)."""
    scored = [r for r in rows if r["positive"] is not None]
    return {
        "documents": len(scored),
        "positives": sum(r["positive"] for r in scored),
        "negatives": sum(not r["positive"] for r in scored),
        "roc_auc": roc_auc([r["score"] for r in scored if r["positive"]], [r["score"] for r in scored if not r["positive"]]),
        "at_medium": confusion(scored, FLAGGED["medium"]),
        "at_high": confusion(scored, FLAGGED["high"]),
        "grades_by_label": {lab: dict(Counter(r["grade"] for r in rows if r["label"] == lab))
                            for lab in LABELS if any(r["label"] == lab for r in rows)},
    }


# ---------------------------------------------------------------- reading labels

def _flag(value, doc_id, column) -> bool:
    v = str(value).strip().lower() if value is not None else ""
    if v in TRUE:
        return True
    if v in FALSE or v == "":
        return False
    raise A.InputError(f"document {doc_id!r}: {column} must be true/false, got {value!r}")


def attach_labels(docs: list[dict], args) -> None:
    for d in docs:
        raw = d["extra"].get(args.label_column)
        if raw is None:
            raise A.InputError(f"document {d['id']!r}: no value in the {args.label_column!r} column")
        label = str(raw).strip().lower()
        if label not in LABELS:
            raise A.InputError(f"document {d['id']!r}: label {raw!r} is not one of {', '.join(LABELS)}")
        d["label"] = label
        d["cluster_label"] = d["extra"].get(args.cluster_column)
        d["hard"] = _flag(d["extra"].get(args.hard_negative_column), d["id"], args.hard_negative_column)
        if d["hard"] and label != "human":
            raise A.InputError(f"document {d['id']!r}: {args.hard_negative_column} only applies to human documents")


# ---------------------------------------------------------------- evaluation

def _positive(label: str, templated_as: str):
    if label == "bulk":
        return True
    if label == "human":
        return False
    return {"exclude": None, "positive": True, "negative": False}[templated_as]


def build_rows(docs, results, pack, templated_as) -> list[dict]:
    rows = []
    for d in docs:
        r = results[d["id"]]
        chars = r.length.get("chars", 0)
        rows.append({
            "id": d["id"], "label": d["label"], "positive": _positive(d["label"], templated_as), "hard": d["hard"],
            "score": r.score if r.score is not None else 0.0, "has_score": r.score is not None, "grade": r.grade,
            "chars": chars, "band": length_band(chars), "domain": d["meta"].get("domain") or pack.domain,
            "cluster_label": d["cluster_label"], "cluster": (r.cluster or {}).get("id"),
            "meta": d["meta"], "result": r,
        })
    return rows


def cluster_level(rows: list[dict], flagged: tuple, min_share: float) -> dict:
    """Cluster-level recall and precision (FR-8.2).

    A labeled bulk cluster is detected when at least `min_share` of its members are flagged. A cluster found
    by the session (size >= 2) is predicted when at least `min_share` of its members are flagged, and is right
    when at least `min_share` of its members are labeled bulk.
    """
    truth = defaultdict(list)
    for r in rows:
        if r["label"] == "bulk" and r["cluster_label"]:
            truth[r["cluster_label"]].append(r)
    found = defaultdict(list)
    for r in rows:
        if r["cluster"]:
            found[r["cluster"]].append(r)

    share = lambda members, pred: sum(pred(m) for m in members) / len(members)
    is_flagged = lambda m: m["grade"] in flagged
    detected = [c for c, m in truth.items() if share(m, is_flagged) >= min_share]
    predicted = [c for c, m in found.items() if len(m) >= 2 and share(m, is_flagged) >= min_share]
    correct = [c for c in predicted if share(found[c], lambda m: m["label"] == "bulk") >= min_share]
    return {"labeled_clusters": len(truth), "detected": len(detected), "recall": ratio(len(detected), len(truth)),
            "predicted_clusters": len(predicted), "correct": len(correct),
            "precision": ratio(len(correct), len(predicted)), "min_share": min_share,
            "missed": sorted(set(truth) - set(detected))}


def hard_negatives(rows: list[dict]) -> dict:
    hard = [r for r in rows if r["label"] == "human" and r["hard"]]
    plain = [r for r in rows if r["label"] == "human" and not r["hard"]]
    out = {"hard_documents": len(hard), "ordinary_documents": len(plain)}
    for name, flagged in (("medium", FLAGGED["medium"]), ("high", FLAGGED["high"])):
        h = ratio(sum(r["grade"] in flagged for r in hard), len(hard))
        o = ratio(sum(r["grade"] in flagged for r in plain), len(plain))
        out[f"hard_fpr_{name}"], out[f"ordinary_fpr_{name}"] = h, o
        out[f"ratio_{name}"] = ratio(h, o) if h is not None and o else None
    return out


def evidence_ok(r) -> bool:
    ok = [s for s in r.signals if s.status == "ok"]
    return bool(r.evidence) and (not ok or abs(sum(s.contribution or 0 for s in ok) - r.score) < 1e-9)


def acceptance(rows, clusters, hard, cumulative, flag_from) -> list[dict]:
    """The SRS 7.2 criteria that a labeled run can show; the others need other data (AC-4, AC-7, AC-9)."""
    human = [r for r in rows if r["label"] == "human" and not r["hard"]]
    high = group_metrics(rows)["at_high"]
    shown = [r for r in rows if r["grade"] in ("medium", "high")]
    short = [r for r in rows if r["chars"] < 500]

    def item(ac, text, value, direction, note=None):
        t = TARGETS[ac]
        passed = None if value is None else (value >= t if direction == ">=" else value <= t)
        return {"id": ac, "criterion": text, "target": f"{direction} {t}", "value": value, "passed": passed,
                **({"note": note} if note else {})}

    return [
        item("AC-1", "cluster-level recall, flagged medium or higher", clusters["recall"] if cumulative else None, ">=",
             None if cumulative else "needs --cumulative"),
        item("AC-2", "false-positive rate on ordinary human content at high", ratio(sum(r["grade"] == "high" for r in human), len(human)), "<="),
        item("AC-3", "precision among documents graded high", high["precision"] if cumulative else None, ">=",
             "needs --cumulative" if not cumulative else
             "no document was graded high, so precision is undefined" if high["precision"] is None else None),
        item("AC-5", "documents under 500 characters graded insufficient_length, no model signal",
             None if cumulative or not short else
             sum(r["grade"] == "insufficient_length" and not any(
                 s.name in GROUPS["model"] and s.status == "ok" for s in r["result"].signals) for r in short) / len(short),
             ">=", "Single Mode only" if cumulative else None),
        item("AC-6", "medium/high results carry evidence and contributions that sum to the score",
             sum(evidence_ok(r["result"]) for r in shown) / len(shown) if shown else None, ">="),
        item("AC-8", "hard-negative false-positive rate relative to ordinary human content (at high)",
             hard["ratio_high"] if hard["ordinary_fpr_high"] else None, "<=",
             None if hard["ordinary_fpr_high"] else "ordinary human false-positive rate is 0 or there are no hard negatives"),
    ]


def error_lists(rows, flagged: tuple, top: int) -> dict:
    def brief(r):
        res = r["result"]
        return {"id": r["id"], "label": r["label"], "grade": r["grade"], "score": res.score, "chars": r["chars"],
                "domain": r["domain"], "meta": r["meta"], "cluster": r["cluster"],
                "signals": {s.name: s.score for s in res.signals if s.status == "ok"},
                "evidence": [dataclasses.asdict(e) for e in res.evidence]}
    fp = sorted((r for r in rows if r["positive"] is False and r["grade"] in flagged), key=lambda r: -r["score"])
    fn = sorted((r for r in rows if r["positive"] is True and r["grade"] not in flagged), key=lambda r: r["score"])
    return {"false_positives": [brief(r) for r in fp[:top]], "false_negatives": [brief(r) for r in fn[:top]],
            "false_positive_total": len(fp), "false_negative_total": len(fn)}


def ablation(detector, docs, args, score_fn) -> list[dict]:
    """Metrics with one signal group removed at a time, so each group's value is measured (SRS 7.3)."""
    pack = detector.pack

    def summary(det):
        results = score_fn(det)
        m = group_metrics(build_rows(docs, results, det.pack, args.templated_as))
        return {"roc_auc": m["roc_auc"], **{k: m["at_medium"][k] for k in ("precision", "recall", "f1")}}

    base = summary(detector)
    out = [{"removed": None, "signals": [], **base, "delta_roc_auc": None}]
    for group, names in GROUPS.items():
        present = [s for s in pack.signals if s in names]
        if not present:
            continue
        if group == "relational" and not args.cumulative:
            out.append({"removed": group, "signals": present, "skipped": "relational signals need --cumulative"})
            continue
        s = summary(Detector(dataclasses.replace(pack, signals=[x for x in pack.signals if x not in names])))
        out.append({"removed": group, "signals": present, **s,
                    "delta_roc_auc": None if None in (s["roc_auc"], base["roc_auc"]) else s["roc_auc"] - base["roc_auc"]})
    return out


# ---------------------------------------------------------------- output

def _f(x, d=3):
    return "" if x is None else f"{x:.{d}f}"


def metrics_table(report: dict) -> list[dict]:
    """Flat rows (one per group) for CSV: overall, then each domain and length band."""
    rows = []
    groups = [("overall", "all", report["overall"])]
    groups += [("domain", k, v) for k, v in report["by_domain"].items()]
    groups += [("length_band", k, v) for k, v in report["by_length_band"].items()]
    for kind, name, m in groups:
        row = {"group": kind, "name": name, "documents": m["documents"], "positives": m["positives"],
               "negatives": m["negatives"], "roc_auc": _f(m["roc_auc"])}
        for level in ("medium", "high"):
            c = m[f"at_{level}"]
            row.update({f"{level}_{k}": c[k] for k in ("tp", "fp", "fn", "tn")})
            row.update({f"{level}_{k}": _f(c[k]) for k in ("precision", "recall", "f1", "false_positive_rate")})
        rows.append(row)
    return rows


def errors_table(errors: dict) -> list[dict]:
    rows = []
    for kind, key in (("false_positive", "false_positives"), ("false_negative", "false_negatives")):
        for rank, e in enumerate(errors[key], start=1):
            ev = [f"{x['signal']}:{x['type']}:{x['value']}" for x in e["evidence"]]
            rows.append({"error": kind, "rank": rank, "id": e["id"], "label": e["label"], "grade": e["grade"],
                         "score": _f(e["score"], 4), "chars": e["chars"], "domain": e["domain"],
                         "author": e["meta"].get("author", ""), "cluster": e["cluster"] or "",
                         "signals": ";".join(f"{k}={_f(v, 2)}" for k, v in e["signals"].items()),
                         "evidence": " | ".join(ev[:5]) + (f" | (+{len(ev) - 5} more)" if len(ev) > 5 else "")})
    return rows


def run(args) -> int:
    try:
        if (args.window_count or args.window_hours) and not args.cumulative:
            raise A.InputError("window options need --cumulative")
        if args.input_format in ("dir", "text"):
            raise A.InputError("a labeled file must be CSV, TSV or JSONL")
        detector = Detector(args.pack)
        docs = A.read_documents(args, extra=(args.label_column, args.cluster_column, args.hard_negative_column))
        attach_labels(docs, args)

        ns = SimpleNamespace(window_count=args.window_count, window_hours=args.window_hours,
                             load_session=None, save_session=None, hash_only=False)
        ordered = docs

        def score(det):
            nonlocal ordered
            if args.cumulative:
                results, _, ordered = A._score_cumulative(det, docs, ns)
                return results
            return A._score_single(det, docs)

        results = score(detector)
        rows = build_rows(ordered, results, detector.pack, args.templated_as)
        flagged = FLAGGED[args.flag_from]

        def split(key):
            groups = defaultdict(list)
            for r in rows:
                groups[r[key]].append(r)
            return groups

        clusters = cluster_level(rows, flagged, args.cluster_min_share) if args.cumulative else None
        hard = hard_negatives(rows)
        report = {
            "pack": detector.pack.meta, "mode": "cumulative" if args.cumulative else "single",
            "flag_from": args.flag_from, "templated_as": args.templated_as, "documents": len(rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "unscored": sum(not r["has_score"] for r in rows),
            "overall": group_metrics(rows),
            "by_domain": {k: group_metrics(v) for k, v in sorted(split("domain").items())},
            "by_length_band": {k: group_metrics(split("band")[k]) for k, _ in BANDS if k in split("band")},
            "cluster_level": clusters, "hard_negatives": hard,
            "acceptance": acceptance(rows, clusters or {"recall": None}, hard, args.cumulative, args.flag_from),
            "errors": error_lists(rows, flagged, args.top_errors),
        }
        if args.ablation:
            report["ablation"] = ablation(detector, docs, args, score)
        write_report(args, report)
        return 0
    except (A.InputError, PackError, SessionFileError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"error: {e.filename}: no such file", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0


def write_report(args, report: dict) -> None:
    if args.format == "json":
        f = A._open_out(args.output, False)
        try:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        finally:
            if f is not sys.stdout:
                f.close()
        return
    table = metrics_table(report)
    A._write_csv(args.output, list(table[0]), table)
    target = args.errors_out or (str(Path(args.output).with_suffix("")) + ".errors.csv" if args.output else None)
    errors = errors_table(report["errors"])
    if target:
        A._write_csv(target, ["error", "rank", "id", "label", "grade", "score", "chars", "domain", "author",
                              "cluster", "signals", "evidence"], errors)
    else:
        print("warning: error lists not written; give -o or --errors-out with --format csv", file=sys.stderr)
