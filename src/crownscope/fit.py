"""`crownscope fit`: learn pack weights and thresholds from labeled samples (SRS FR-6.5, NFR-14).

The pack scores a document as a weighted mean of the signals that ran:

    score = sum(w_j * s_j) / sum(w_j)          over the signals available for that document

so the weights are learned in exactly that form. They are non-negative and sum to 1 (a softmax), and a
logistic loss on `sigmoid(a * (score - c))` is minimised, where the slope `a` and offset `c` are only
fitting aids and are not exported. Readers can see what each signal contributes, and the result is an
ordinary pack. A small L2 penalty pulls weights toward uniform (or, per length band, toward the global fit)
so that a few dozen samples cannot produce extreme weights.

Thresholds are then read off the fitted score distribution: `high` is the lowest score that still meets the
precision and false-positive targets (SRS AC-3, AC-2), and `medium` maximises F1 (and never exceeds `high`).

Features are the final per-signal scores after the whole labeled file has been scored (cumulative mode keeps
re-grading in effect), so fit in the mode you deploy. Signals that ran for under half of the documents (for
example relational signals in Single Mode) are not fitted and keep their weights.

Cross-validation holds out whole groups (a bulk cluster, or an author) so that near-duplicates of a training
document never appear in the held-out part.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml

from . import analyze as A
from . import evaluate as E
from .detector import Detector, _band
from .pack import PackError, load_pack, load_pack_data, validate
from .persist import SessionFileError
from .result import Signal

BANDS = ("short", "medium", "long")


# ---------------------------------------------------------------- the model

def softmax(theta: list[float]) -> list[float]:
    m = max(theta)
    e = [math.exp(t - m) for t in theta]
    s = sum(e)
    return [x / s for x in e]


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _softplus(x: float) -> float:
    return math.log1p(math.exp(x)) if x < 30 else x


def mean_score(w: list[float], scores: list[float], mask: list[int]) -> float | None:
    """The pack's aggregation: weighted mean over the signals that are available."""
    den = sum(wj * m for wj, m in zip(w, mask))
    return sum(wj * s * m for wj, s, m in zip(w, scores, mask)) / den if den > 0 else None


def loss_and_grad(params: list[float], X, M, y, cw, l2: float, prior: list[float]):
    """Class-weighted logistic loss of sigmoid(a*(score-c)) with softmax weights, and its gradient.

    params = [theta_1..theta_K, alpha, c] with a = softplus(alpha).
    """
    k = len(prior)
    theta, alpha, c = params[:k], params[k], params[k + 1]
    a = _softplus(alpha)
    w = softmax(theta)
    gw = [0.0] * k
    ga = gc = loss = 0.0
    n = 0
    for xs, ms, yi, wi in zip(X, M, y, cw):
        den = sum(wj * m for wj, m in zip(w, ms))
        if den <= 0:
            continue
        score = sum(wj * s * m for wj, s, m in zip(w, xs, ms)) / den
        p = min(max(_sigmoid(a * (score - c)), 1e-12), 1 - 1e-12)
        loss += -wi * (math.log(p) if yi else math.log(1 - p))
        err = wi * (p - yi)  # d loss / d logit
        dscore = err * a
        for j in range(k):
            if ms[j]:
                gw[j] += dscore * (xs[j] - score) / den
        ga += err * (score - c)
        gc += -err * a
        n += 1
    n = max(n, 1)
    loss /= n
    gw = [g / n for g in gw]
    dot = sum(wj * gj for wj, gj in zip(w, gw))
    gtheta = [w[j] * (gw[j] - dot) for j in range(k)]  # through the softmax
    reg = l2 * sum((t - p0) ** 2 for t, p0 in zip(theta, prior))
    gtheta = [g + 2 * l2 * (t - p0) for g, t, p0 in zip(gtheta, theta, prior)]
    sig = 1 / (1 + math.exp(-alpha))  # d softplus
    return loss + reg, gtheta + [ga / n * sig, gc / n]


def fit_logistic_mean(X, M, y, l2=0.01, prior=None, iterations=400, lr=0.05):
    """Adam on `loss_and_grad`. Returns (weights, a, c, final_loss). Deterministic."""
    k = len(X[0])
    prior = list(prior) if prior is not None else [0.0] * k
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    cw = [len(y) / (2 * n_pos) if yi else len(y) / (2 * n_neg) for yi in y]  # balance the classes

    w0 = softmax(prior)
    s0 = [mean_score(w0, xs, ms) for xs, ms in zip(X, M)]
    pos = [s for s, yi in zip(s0, y) if yi and s is not None]
    neg = [s for s, yi in zip(s0, y) if not yi and s is not None]
    mp, mn = sum(pos) / len(pos), sum(neg) / len(neg)
    a0 = min(30.0, 4 / max(0.05, mp - mn))
    params = prior + [math.log(math.expm1(a0)), (mp + mn) / 2]

    m1 = [0.0] * len(params)
    m2 = [0.0] * len(params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    prev, still = None, 0
    loss = 0.0
    for t in range(1, iterations + 1):
        loss, g = loss_and_grad(params, X, M, y, cw, l2, prior)
        for i, gi in enumerate(g):
            m1[i] = b1 * m1[i] + (1 - b1) * gi
            m2[i] = b2 * m2[i] + (1 - b2) * gi * gi
            params[i] -= lr * (m1[i] / (1 - b1 ** t)) / (math.sqrt(m2[i] / (1 - b2 ** t)) + eps)
        still = still + 1 if prev is not None and abs(prev - loss) < 1e-10 else 0
        if still >= 10:
            break
        prev = loss
    return softmax(params[:k]), _softplus(params[k]), params[k + 1], loss


# ---------------------------------------------------------------- thresholds

def choose_thresholds(scored: list[tuple[float, bool]], target_precision: float, max_fpr: float,
                      unflaggable: tuple[int, int] = (0, 0)) -> dict:
    """Pick (medium, high) from (score, is_positive) pairs. Flagged means score >= threshold.

    `unflaggable` = (positives, negatives) that have no score and so can never be flagged whatever the
    threshold (short documents without a relational hit): they count as misses and correct rejections.

    high: the lowest score where precision >= target_precision and the false-positive rate <= max_fpr.
    medium: the score that maximises F1 (ties go to the higher score); never above high.
    """
    pos = sum(p for _, p in scored) + unflaggable[0]
    neg = len(scored) - sum(p for _, p in scored) + unflaggable[1]
    distinct = sorted({s for s, _ in scored}, reverse=True)
    stats = []  # for each candidate cut (descending): (score, tp, fp)
    tp = fp = 0
    by_score = defaultdict(lambda: [0, 0])
    for s, p in scored:
        by_score[s][0 if p else 1] += 1
    for s in distinct:
        tp += by_score[s][0]
        fp += by_score[s][1]
        stats.append((s, tp, fp))

    def below(s):  # next lower observed score, to place a threshold between observations
        lower = [d for d in distinct if d < s]
        return lower[0] if lower else s - 0.02

    def place(s):
        mid = (s + below(s)) / 2
        r = round(mid, 4)
        return min(1.0, max(0.0, r if below(s) < r <= s else mid))

    notes = []
    high = None
    for s, tp_, fp_ in stats:
        if tp_ and tp_ / (tp_ + fp_) >= target_precision and fp_ / max(neg, 1) <= max_fpr:
            high = s  # keep going down: the lowest feasible cut keeps the most recall
    best = None
    for s, tp_, fp_ in stats:
        f1 = 2 * tp_ / (2 * tp_ + fp_ + (pos - tp_)) if tp_ else 0.0
        if best is None or f1 > best[0] + 1e-12:
            best = (f1, s)
    if high is None:
        notes.append(f"no score reaches precision {target_precision} with false-positive rate <= {max_fpr}; "
                     "`high` is set to 1.0, so nothing is graded high")
        high_t = 1.0
    else:
        high_t = place(high)
    if best is None or best[0] == 0.0:
        notes.append("no threshold gives a positive F1; thresholds were left unchanged")
        return {"medium": None, "high": None, "notes": notes}
    medium_t = min(place(best[1]), high_t)
    if high is not None and medium_t == high_t:
        notes.append("`medium` and `high` coincide: the labeled data separates at a single score, so there is no "
                     "medium band; harder examples (borderline documents) would give it a width")
    return {"medium": medium_t, "high": max(high_t, medium_t), "notes": notes, "best_f1": best[0]}


# ---------------------------------------------------------------- data

def collect(docs, results, pack, templated_as) -> list[dict]:
    rows = []
    for d in docs:
        r = results[d["id"]]
        pos = E._positive(d["label"], templated_as)
        if pos is None:
            continue
        chars = r.length.get("chars", 0)
        rows.append({
            "id": d["id"], "positive": pos, "chars": chars, "band": _band(chars),
            "signals": {s.name: s.score for s in r.signals if s.status == "ok" and s.name in pack.signals},
            "group": d.get("cluster_label") or d["meta"].get("author") or d["id"],
            "label": d["label"], "hard": d["hard"],
        })
    return rows


def matrices(rows, names):
    X = [[row["signals"].get(n, 0.0) for n in names] for row in rows]
    M = [[1 if n in row["signals"] else 0 for n in names] for row in rows]
    return X, M, [1 if row["positive"] else 0 for row in rows]


def deployed(rows, pack) -> list[dict]:
    """Score and grade each row exactly as the pack would, including the short-document rule."""
    det = Detector(pack)
    out = []
    for row in rows:
        sigs = [Signal(n, score=row["signals"][n], confidence=1.0) if n in row["signals"]
                else Signal(n, status="unavailable") for n in pack.signals]
        grade, score, _ = det.aggregate(sigs, row["chars"], row["chars"] < pack.min_length)
        out.append({**row, "grade": grade, "score": score if score is not None else 0.0, "has_score": score is not None})
    return out


def summarize(rows, pack) -> dict:
    m = E.group_metrics(deployed(rows, pack))
    pick = lambda c: {k: c[k] for k in ("precision", "recall", "f1", "false_positive_rate")}
    return {"roc_auc": m["roc_auc"], "at_medium": pick(m["at_medium"]), "at_high": pick(m["at_high"])}


# ---------------------------------------------------------------- fitting a pack

class Options(SimpleNamespace):
    pass


def fit_pack_data(rows, data: dict, opt) -> tuple[dict, dict]:
    """Return (new pack data, details) fitted on `rows`. `data` is the validated pack YAML."""
    names_all = list(data["signals"])
    avail = {n: sum(n in r["signals"] for r in rows) / len(rows) for n in names_all}
    fitted = [n for n in names_all if avail[n] >= opt.min_availability]
    kept = {n: f"available for {avail[n]:.0%} of documents" for n in names_all if n not in fitted}
    if not fitted:
        raise A.InputError("no signal ran for at least half of the labeled documents; nothing to fit")

    def fit_set(subset, prior):
        X, M, y = matrices(subset, fitted)
        w, a, c, loss = fit_logistic_mean(X, M, y, opt.l2, prior, opt.iterations)
        return w, loss

    theta_global = None
    w_global, loss = fit_set(rows, [0.0] * len(fitted))
    theta_global = [math.log(max(x, 1e-9)) for x in w_global]
    source = {}
    weights = {}
    for band in BANDS:
        if band not in data["weights"]:
            continue
        sub = [r for r in rows if r["band"] == band]
        npos, nneg = sum(r["positive"] for r in sub), sum(not r["positive"] for r in sub)
        if opt.bands and npos >= opt.min_band_class and nneg >= opt.min_band_class:
            w, _ = fit_set(sub, theta_global)
            source[band] = f"band ({npos} positive, {nneg} negative)"
        else:
            w = w_global
            source[band] = "global" + (f" (band has {npos} positive, {nneg} negative)" if opt.bands else "")
        w = [x if x >= opt.min_weight else 0.0 for x in w]
        total = sum(w)
        w = [x / total for x in w]
        # signals that were not fitted keep their original weight; the fitted ones share what is left
        original = data["weights"][band]
        reserved = sum(original.get(n, 0.0) for n in kept)
        share = 1.0 - reserved if reserved < 1.0 else 1.0
        weights[band] = {n: (round(w[fitted.index(n)] * share, 4) if n in fitted else original.get(n, 0.0))
                         for n in names_all}

    new = {**data, "weights": weights}
    scored = deployed(rows, load_pack_dict(new))
    th = choose_thresholds([(r["score"], r["positive"]) for r in scored if r["has_score"]],
                           opt.target_precision, opt.max_fpr,
                           unflaggable=(sum(r["positive"] for r in scored if not r["has_score"]),
                                        sum(not r["positive"] for r in scored if not r["has_score"])))
    if th["medium"] is not None:
        new["thresholds"] = {"medium": round(th["medium"], 4), "high": round(th["high"], 4)}
        validate(new)
    return new, {"fitted": fitted, "kept": kept, "source": source, "loss": loss, "threshold_notes": th["notes"]}


def load_pack_dict(data: dict):
    from .pack import Pack
    validate(data)
    return Pack(**{k: data[k] for k in Pack.__dataclass_fields__ if k in data})


# ---------------------------------------------------------------- cross-validation

def assign_folds(rows, k: int, seed: int) -> list[int]:
    """Fold index per row. Whole groups (a bulk cluster, or an author) stay together; each class is spread
    evenly. Deterministic for a given seed."""
    fold = {}
    for cls in (True, False):
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            if r["positive"] is cls:
                groups[r["group"]].append(i)
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        keys.sort(key=lambda g: -len(groups[g]))  # stable: largest groups first, ties in shuffled order
        load = [0] * k
        for g in keys:
            f = load.index(min(load))
            load[f] += len(groups[g])
            for i in groups[g]:
                fold[i] = f
    return [fold[i] for i in range(len(rows))]


def cross_validate(rows, data, opt) -> dict:
    pos_groups = len({r["group"] for r in rows if r["positive"]})
    neg_groups = len({r["group"] for r in rows if not r["positive"]})
    k = min(opt.folds, pos_groups, neg_groups)
    if opt.folds < 2:
        return {"skipped": "disabled with --folds 0"}
    if k < 2:
        return {"skipped": f"needs at least 2 groups per class ({pos_groups} positive, {neg_groups} negative)"}
    folds = assign_folds(rows, k, opt.seed)
    held_new, held_old = [], []
    base = load_pack_dict(data)
    for f in range(k):
        train = [r for r, g in zip(rows, folds) if g != f]
        test = [r for r, g in zip(rows, folds) if g == f]
        if min(sum(r["positive"] for r in train), sum(not r["positive"] for r in train)) < 3:
            return {"skipped": "a training split would have fewer than 3 documents of one class"}
        fitted_data, _ = fit_pack_data(train, data, opt)
        held_new += deployed(test, load_pack_dict(fitted_data))
        held_old += deployed(test, base)
    return {"folds": k, "after": _summary_of(held_new), "before": _summary_of(held_old)}


def _summary_of(scored_rows) -> dict:
    m = E.group_metrics(scored_rows)
    pick = lambda c: {k: c[k] for k in ("precision", "recall", "f1", "false_positive_rate")}
    return {"roc_auc": m["roc_auc"], "at_medium": pick(m["at_medium"]), "at_high": pick(m["at_high"])}


# ---------------------------------------------------------------- command

def cv_warnings(cv: dict) -> list[str]:
    """Held-out results that argue against trusting the fit."""
    if "after" not in cv:
        return []
    out = []
    before, after = cv["before"]["roc_auc"], cv["after"]["roc_auc"]
    if before is not None and after is not None and after < before - 0.01:
        out.append(f"cross-validated ROC-AUC fell from {before:.3f} to {after:.3f}: the labeled set is probably too "
                   "small to fit reliably (the fit improves the training data but not held-out data); add labeled "
                   "documents or raise --l2 before using this pack")
    return out


def bump(version: str) -> str:
    return f"{version}+fit"


def run(args) -> int:
    try:
        if (args.window_count or args.window_hours) and not args.cumulative:
            raise A.InputError("window options need --cumulative")
        out = Path(args.output)
        if out.exists() and not args.force:
            raise A.InputError(f"{out} exists; use --force to overwrite")

        data = load_pack_data(args.pack)
        detector = Detector(args.pack)
        docs = A.read_documents(args, extra=(args.label_column, args.cluster_column, args.hard_negative_column))
        E.attach_labels(docs, args)

        ns = SimpleNamespace(window_count=args.window_count, window_hours=args.window_hours,
                             load_session=None, save_session=None, hash_only=False)
        if args.cumulative:
            results, _, docs = A._score_cumulative(detector, docs, ns)
        else:
            results = A._score_single(detector, docs)
        rows = collect(docs, results, detector.pack, args.templated_as)

        npos, nneg = sum(r["positive"] for r in rows), sum(not r["positive"] for r in rows)
        if min(npos, nneg) < args.min_class:
            raise A.InputError(f"not enough labeled documents: need at least {args.min_class} of each class, "
                               f"found {npos} bulk and {nneg} human")

        opt = Options(l2=args.l2, iterations=args.iterations, min_weight=args.min_weight, bands=not args.no_bands,
                      min_band_class=args.min_band_class, min_availability=0.5, folds=args.folds, seed=args.seed,
                      target_precision=args.target_precision, max_fpr=args.max_fpr)
        new, detail = fit_pack_data(rows, data, opt)
        cv = cross_validate(rows, data, opt)

        warnings = list(detail["threshold_notes"])
        if detail["kept"]:
            warnings.append("not fitted, original weights kept: " + "; ".join(f"{n} ({why})" for n, why in detail["kept"].items())
                            + ("; fit with --cumulative to learn relational signals" if not args.cumulative else ""))
        if "skipped" in cv:
            warnings.append("cross-validation skipped: " + cv["skipped"])
        warnings += cv_warnings(cv)

        raw = Path(args.input).read_bytes()
        new["version"] = args.set_version or bump(str(data["version"]).split("+fit")[0])
        new["fit"] = {
            "method": "logistic regression on the weighted-mean score; non-negative weights summing to 1",
            "base_version": str(data["version"]), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "mode": "cumulative" if args.cumulative else "single", "documents": len(rows),
            "positives": npos, "negatives": nneg, "templated_as": args.templated_as,
            "fitted_signals": detail["fitted"], "l2": args.l2, "iterations": args.iterations,
            "target_precision": args.target_precision, "max_fpr": args.max_fpr,
            "cross_validation_folds": cv.get("folds"),
            "cross_validation_roc_auc": (cv.get("after") or {}).get("roc_auc"),
        }
        validate(new)
        header = ("# Fitted by `crownscope fit` from labeled samples (see the `fit` key for provenance).\n"
                  "# Comments of the original pack are not carried over.\n")
        out.write_text(header + yaml.safe_dump(new, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
        load_pack(out)  # the file we wrote must load

        report = {
            "output": str(out), "pack": {"name": new["name"], "from_version": str(data["version"]), "to_version": new["version"]},
            "mode": new["fit"]["mode"], "documents": len(rows), "positives": npos, "negatives": nneg,
            "signals": {"fitted": detail["fitted"], "kept": detail["kept"]},
            "weights": {"before": data["weights"], "after": new["weights"], "fit_from": detail["source"]},
            "thresholds": {"before": data["thresholds"], "after": new["thresholds"]},
            "metrics_on_labeled_data": {"before": summarize(rows, detector.pack),
                                        "after": summarize(rows, load_pack(out))},
            "cross_validation": cv, "warnings": warnings,
        }
        f = A._open_out(args.report, False)
        try:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        finally:
            if f is not sys.stdout:
                f.close()
        return 0
    except (A.InputError, PackError, SessionFileError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"error: {e.filename}: no such file", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
