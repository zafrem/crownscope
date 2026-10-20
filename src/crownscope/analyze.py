"""`crownscope analyze` (SRS IR-8, IR-9, FR-6.6, FR-8.4).

Reads CSV, JSONL or a directory of .txt files, scores every document, and writes the results ranked by
score (highest first) as JSON or CSV. With --cumulative the documents are fed through one session, in
order, so relational signals, clusters and re-grading apply, and a grouped report is written as well.

Document text is never written: the output carries IDs, metadata, scores, signals and evidence.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from .detector import Detector
from .pack import PackError
from .persist import SessionFileError
from .store import parse_timestamp

META_FIELDS = ("author", "source", "product", "timestamp", "domain")
GRADE_RANK = {"low": 1, "medium": 2, "high": 3}


class InputError(Exception):
    """A problem with the input that the user can fix; reported without a traceback."""


# ---------------------------------------------------------------- reading

def _doc(doc_id, text, meta, where, extra=None):
    if not isinstance(text, str):
        raise InputError(f"{where}: text is missing or not a string")
    if meta.get("timestamp") not in (None, ""):
        try:
            parse_timestamp(meta["timestamp"])
        except ValueError:
            raise InputError(f"{where}: timestamp {meta['timestamp']!r} is not epoch seconds or ISO-8601") from None
    return {"id": doc_id, "text": text, "meta": {k: v for k, v in meta.items() if v not in (None, "")},
            "extra": {k: v for k, v in (extra or {}).items() if v not in (None, "")}}


def _read_csv(path: Path, args, delimiter: str, extra: tuple) -> list[dict]:
    csv.field_size_limit(2**31 - 1)  # documents can be long
    with open(path, newline="", encoding=args.encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if args.text_column not in (reader.fieldnames or []):
            raise InputError(f"{path}: no column named {args.text_column!r} (found: {', '.join(reader.fieldnames or [])})")
        docs = []
        for n, row in enumerate(reader, start=2):  # line 1 is the header
            doc_id = row.get(args.id_column) or f"row_{n - 1}"
            docs.append(_doc(doc_id, row[args.text_column], {k: row.get(k) for k in META_FIELDS}, f"{path} line {n}",
                             {k: row.get(k) for k in extra}))
    return docs


def _read_jsonl(path: Path, args, extra: tuple) -> list[dict]:
    docs = []
    with open(path, encoding=args.encoding) as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise InputError(f"{path} line {n}: invalid JSON ({e.msg})") from None
            if not isinstance(obj, dict):
                raise InputError(f"{path} line {n}: expected a JSON object")
            meta = {**{k: obj.get(k) for k in META_FIELDS}, **(obj.get("meta") or {})}
            docs.append(_doc(str(obj.get(args.id_column) or f"row_{len(docs) + 1}"), obj.get(args.text_column), meta,
                             f"{path} line {n}", {k: obj.get(k) for k in extra}))
    return docs


def _read_dir(path: Path, args) -> list[dict]:
    files = sorted(p for p in path.rglob("*.txt") if p.is_file())
    if not files:
        raise InputError(f"{path}: no .txt files found")
    return [_doc(p.relative_to(path).as_posix(), p.read_text(encoding=args.encoding), {}, str(p)) for p in files]


def read_documents(args, extra: tuple = ()) -> list[dict]:
    """Read documents; `extra` names further columns/keys to carry along (e.g. labels for evaluation)."""
    path = Path(args.input)
    if not path.exists():
        raise InputError(f"{path}: no such file or directory")
    kind = args.input_format
    if kind == "auto":
        kind = ("dir" if path.is_dir() else "jsonl" if path.suffix in (".jsonl", ".ndjson") else
                "csv" if path.suffix == ".csv" else "tsv" if path.suffix == ".tsv" else
                "text" if path.suffix == ".txt" else None)
        if kind is None:
            raise InputError(f"{path}: cannot tell the format from the name; use --input-format")
    try:
        if kind == "dir":
            docs = _read_dir(path, args)
        elif kind == "text":
            docs = [_doc(path.name, path.read_text(encoding=args.encoding), {}, str(path))]
        elif kind == "jsonl":
            docs = _read_jsonl(path, args, extra)
        else:
            docs = _read_csv(path, args, "\t" if kind == "tsv" else ",", extra)
    except UnicodeDecodeError as e:
        raise InputError(f"{path}: not readable as {args.encoding} ({e.reason}); try --encoding cp949") from None
    except (IsADirectoryError, NotADirectoryError) as e:
        raise InputError(f"{path}: {e.strerror}") from None
    if not docs:
        raise InputError(f"{path}: no documents found")
    seen = {}
    for i, d in enumerate(docs, start=1):
        if d["id"] in seen:
            raise InputError(f"duplicate id {d['id']!r} (documents {seen[d['id']]} and {i})")
        seen[d["id"]] = i
    return docs


# ---------------------------------------------------------------- scoring

def _warn_missing_language_model(detector: Detector) -> None:
    if "ppl" in detector.pack.signals and detector.language_model is None:
        print(f"warning: ppl is unavailable: {detector.language_model_error}", file=sys.stderr)


def _score_single(detector: Detector, docs: list[dict]) -> dict:
    _warn_missing_language_model(detector)
    return {d["id"]: detector.analyze(d["text"], d["meta"]) for d in docs}


def _score_cumulative(detector: Detector, docs: list[dict], args):
    window = {k: v for k, v in (("count", args.window_count), ("hours", args.window_hours)) if v is not None} or None
    if args.load_session:
        session = detector.load_session(args.load_session, window=window, hash_only=True if args.hash_only else None)
    else:
        session = detector.session(window=window, hash_only=args.hash_only)
    _warn_missing_language_model(detector)
    if "embed_sim" in detector.pack.signals and detector.embedder is None:
        print(f"warning: embed_sim is unavailable: {detector.embedder_error}", file=sys.stderr)
    if ("novelty" in detector.pack.signals and detector.pack.options.get("novelty", {}).get("semantic", {}).get("enabled")
            and detector.embedder is None and "embed_sim" not in detector.pack.signals):
        print(f"warning: novelty will use lexical reuse only: {detector.embedder_error}", file=sys.stderr)
    results = {}
    session.on_evict = lambda doc_id, result: results.__setitem__(doc_id, result)

    stamped = [d for d in docs if d["meta"].get("timestamp") is not None]
    if len(stamped) == len(docs):  # a stream is processed in time order; sorting is stable for ties
        docs = sorted(docs, key=lambda d: parse_timestamp(d["meta"]["timestamp"]))
    elif (window or {}).get("hours") or (not window and detector.pack.window.get("hours")):
        print("warning: documents without a timestamp are treated as arriving now, so the time window "
              "cannot separate them", file=sys.stderr)

    for d in docs:
        session.add(d["text"], {**d["meta"], "id": d["id"]})
    results.update(session.current_results())  # documents still in the window, as they stand at the end
    if args.save_session:
        session.save(args.save_session)
    return results, session, docs  # docs in processing order: ties in the ranking break the same way


# ---------------------------------------------------------------- ranking and output

def rank(docs: list[dict], results: dict, min_grade: str | None, top: int | None) -> list[dict]:
    """Ranked rows, highest score first. Documents without a score (too short, no relational hit) come last."""
    rows = []
    for order, d in enumerate(docs):
        r = results.get(d["id"])
        if r is None:  # defensive: a document the session never saw
            continue
        rows.append((r.score is None, -(r.score or 0.0), order, d, r))
    rows.sort(key=lambda t: t[:3])
    out = []
    for _, _, _, d, r in rows:
        if min_grade and GRADE_RANK.get(r.grade, 0) < GRADE_RANK[min_grade]:
            continue
        out.append({"rank": len(out) + 1, "id": d["id"], "meta": d["meta"], **r.to_dict()})
    return out[:top] if top else out


def _fmt(x, digits=4):
    return "" if x is None else f"{x:.{digits}f}"


def _csv_row(row: dict, max_evidence: int) -> dict:
    cluster = row.get("cluster") or {}
    meta = row["meta"]
    evidence = [f"{e['signal']}:{e['type']}:{e['value']}" for e in row["evidence"]]
    return {
        "rank": row["rank"], "id": row["id"], "grade": row["grade"], "score": _fmt(row["score"]),
        "confidence": _fmt(row["confidence"]), "chars": row["length"].get("chars", ""),
        "author": meta.get("author", ""), "source": meta.get("source", ""), "product": meta.get("product", ""),
        "cluster_id": cluster.get("id", ""), "cluster_size": cluster.get("size", ""),
        "signals": ";".join(f"{s['name']}={_fmt(s['score'], 2)}" for s in row["signals"] if s["status"] == "ok"),
        "evidence": " | ".join(evidence[:max_evidence]) + (f" | (+{len(evidence) - max_evidence} more)"
                                                           if len(evidence) > max_evidence else ""),
    }


CSV_COLUMNS = ["rank", "id", "grade", "score", "confidence", "chars", "author", "source", "product",
               "cluster_id", "cluster_size", "signals", "evidence"]
REPORT_COLUMNS = ["by", "key", "documents", "flagged", "share_flagged", "dominant_structure",
                  "dominant_structure_share", "dominant_cluster", "dominant_cluster_share", "top_evidence"]


def _report_rows(reports: dict) -> list[dict]:
    rows = []
    for by, rep in reports.items():
        for g in rep["groups"]:
            ds, dc = g["dominant_structure"] or {}, g["dominant_cluster"] or {}
            rows.append({
                "by": by, "key": g["key"], "documents": g["documents"], "flagged": g["flagged"],
                "share_flagged": _fmt(g["share_flagged"], 3),
                "dominant_structure": ds.get("fingerprint", ""), "dominant_structure_share": _fmt(ds.get("share"), 3),
                "dominant_cluster": dc.get("id", ""), "dominant_cluster_share": _fmt(dc.get("share"), 3),
                "top_evidence": " | ".join(f"{e['signal']}:{e['value']} ({e['documents']})" for e in g["top_evidence"]),
            })
    return rows


def _open_out(path: str | None, csv_file: bool):
    if not path:
        return sys.stdout
    # Excel on Korean systems needs the BOM to read UTF-8 CSV correctly
    return open(path, "w", newline="", encoding="utf-8-sig" if csv_file else "utf-8")


def _write_csv(path, columns, rows):
    f = _open_out(path, True)
    try:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    finally:
        if f is not sys.stdout:
            f.close()


def write_output(args, detector, rows, reports) -> None:
    if args.format == "json":
        payload = {"pack": detector.pack.meta, "mode": "cumulative" if args.cumulative else "single",
                   "documents": len(rows), "results": rows}
        if reports is not None:
            payload["reports"] = reports
        f = _open_out(args.output, False)
        try:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        finally:
            if f is not sys.stdout:
                f.close()
        return
    _write_csv(args.output, CSV_COLUMNS, [_csv_row(r, args.max_evidence) for r in rows])
    if reports is not None:
        target = args.report_out or (str(Path(args.output).with_suffix("")) + ".report.csv" if args.output else None)
        if target:
            _write_csv(target, REPORT_COLUMNS, _report_rows(reports))
        else:
            print("warning: grouped report not written; give -o or --report-out with --format csv", file=sys.stderr)


def run(args) -> int:
    try:
        if (args.window_count or args.window_hours or args.hash_only or args.load_session or args.save_session
                or args.report_out) and not args.cumulative:
            raise InputError("window, session and report options need --cumulative")
        detector = Detector(args.pack)
        docs = read_documents(args)
        reports = None
        if args.cumulative:
            results, session, docs = _score_cumulative(detector, docs, args)
            reports = {by: session.report(by=by) for by in args.report_by}
        else:
            results = _score_single(detector, docs)
        write_output(args, detector, rank(docs, results, args.min_grade, args.top), reports)
        return 0
    except (InputError, PackError, SessionFileError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:  # pack name or path, session file
        print(f"error: {e.filename}: no such file", file=sys.stderr)
        return 1
    except BrokenPipeError:  # e.g. piped into `head`
        return 0
