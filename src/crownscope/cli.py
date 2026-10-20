"""Command line (SRS 5.3). `analyze`, `evaluate`, `fit` and `pack validate` work."""
from __future__ import annotations

import argparse
import sys

from .pack import PackError, load_pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crownscope")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="Score documents from CSV, JSONL or a text directory, ranked")
    a.add_argument("input", help="CSV/TSV/JSONL file, a directory of .txt files, or one .txt file")
    a.add_argument("--pack", default="blog", help="bundled pack name (blog, shopping) or a YAML path")
    a.add_argument("--input-format", choices=["auto", "csv", "tsv", "jsonl", "dir", "text"], default="auto")
    a.add_argument("--text-column", default="text", help="column or JSON key holding the text")
    a.add_argument("--id-column", default="id", help="column or JSON key holding the document ID")
    a.add_argument("--encoding", default="utf-8-sig", help="input encoding (use cp949 for Excel-saved Korean CSV)")
    a.add_argument("--format", choices=["json", "csv"], default="json", help="output format")
    a.add_argument("-o", "--output", help="output file (default: standard output)")
    a.add_argument("--top", type=int, help="keep only the N highest-ranked documents")
    a.add_argument("--min-grade", choices=["low", "medium", "high"], help="drop documents graded below this")
    a.add_argument("--max-evidence", type=int, default=5, help="evidence items per row in CSV output")
    c = a.add_argument_group("cumulative mode", "documents are processed as one stream, in timestamp order "
                             "when every document has one")
    c.add_argument("--cumulative", action="store_true", help="relational signals, clusters, re-grading, report")
    c.add_argument("--window-count", type=int, help="keep the most recent N documents (overrides the pack)")
    c.add_argument("--window-hours", type=float, help="keep the last T hours (overrides the pack)")
    c.add_argument("--hash-only", action="store_true", help="keep no document text in the session")
    c.add_argument("--report-by", nargs="+", default=["all"], choices=["all", "author", "source", "product"],
                   help="grouped reports to write")
    c.add_argument("--report-out", help="CSV report file (default: <output>.report.csv)")
    c.add_argument("--load-session", help="continue from a saved session")
    c.add_argument("--save-session", help="save the session afterwards")
    e = sub.add_parser("evaluate", help="Evaluate a pack against a labeled CSV/JSONL file")
    e.add_argument("input", help="labeled CSV/TSV/JSONL: text, label (human|templated|bulk), optional cluster, "
                   "hard_negative, domain, author, source, product, timestamp")
    e.add_argument("--pack", default="blog", help="bundled pack name (blog, shopping) or a YAML path")
    e.add_argument("--input-format", choices=["auto", "csv", "tsv", "jsonl"], default="auto")
    e.add_argument("--text-column", default="text")
    e.add_argument("--id-column", default="id")
    e.add_argument("--label-column", default="label")
    e.add_argument("--cluster-column", default="cluster", help="shared by the members of a labeled bulk cluster")
    e.add_argument("--hard-negative-column", default="hard_negative")
    e.add_argument("--encoding", default="utf-8-sig")
    e.add_argument("--format", choices=["json", "csv"], default="json")
    e.add_argument("-o", "--output", help="output file (default: standard output)")
    e.add_argument("--errors-out", help="CSV error list file (default: <output>.errors.csv)")
    e.add_argument("--flag-from", choices=["medium", "high"], default="medium",
                   help="grade at which a document counts as flagged, for error lists and cluster metrics")
    e.add_argument("--templated-as", choices=["exclude", "positive", "negative"], default="exclude",
                   help="how to treat the ambiguous 'templated' label in precision and recall")
    e.add_argument("--top-errors", type=int, default=10, help="false positives and false negatives to list")
    e.add_argument("--ablation", action="store_true", help="also measure each signal group by removing it")
    ec = e.add_argument_group("cumulative mode")
    ec.add_argument("--cumulative", action="store_true", help="score as one stream; adds cluster-level metrics")
    ec.add_argument("--window-count", type=int)
    ec.add_argument("--window-hours", type=float)
    ec.add_argument("--cluster-min-share", type=float, default=0.5,
                    help="share of members that must be flagged (or labeled bulk) for a cluster to count")
    f = sub.add_parser("fit", help="Learn pack weights and thresholds from a labeled file")
    f.add_argument("input", help="labeled CSV/TSV/JSONL, same columns as `evaluate`")
    f.add_argument("-o", "--output", required=True, help="where to write the fitted pack (YAML)")
    f.add_argument("--pack", default="blog", help="pack to start from: bundled name or YAML path")
    f.add_argument("--report", help="write the fit report here (default: standard output)")
    f.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    f.add_argument("--set-version", help="version for the fitted pack (default: '<old>+fit')")
    f.add_argument("--input-format", choices=["auto", "csv", "tsv", "jsonl"], default="auto")
    f.add_argument("--text-column", default="text")
    f.add_argument("--id-column", default="id")
    f.add_argument("--label-column", default="label")
    f.add_argument("--cluster-column", default="cluster")
    f.add_argument("--hard-negative-column", default="hard_negative")
    f.add_argument("--encoding", default="utf-8-sig")
    f.add_argument("--templated-as", choices=["exclude", "positive", "negative"], default="exclude")
    f.add_argument("--cumulative", action="store_true", help="fit on stream-scored data; needed for relational weights")
    f.add_argument("--window-count", type=int)
    f.add_argument("--window-hours", type=float)
    f.add_argument("--target-precision", type=float, default=0.90, help="precision `high` must reach (AC-3)")
    f.add_argument("--max-fpr", type=float, default=0.02, help="false-positive rate `high` must stay under (AC-2)")
    f.add_argument("--l2", type=float, default=0.01, help="pull toward uniform weights (global) or the global fit (bands)")
    f.add_argument("--iterations", type=int, default=400)
    f.add_argument("--min-weight", type=float, default=0.02, help="weights below this are set to 0")
    f.add_argument("--min-class", type=int, default=10, help="labeled documents needed in each class")
    f.add_argument("--min-band-class", type=int, default=10, help="per class, to fit a length band on its own")
    f.add_argument("--no-bands", action="store_true", help="use one global fit for every length band")
    f.add_argument("--folds", type=int, default=5, help="cross-validation folds (0 disables); groups stay together")
    f.add_argument("--seed", type=int, default=0)
    pack = sub.add_parser("pack", help="Pack tools").add_subparsers(dest="pack_command", required=True)
    pack.add_parser("validate", help="Check a pack file").add_argument("path")
    args = parser.parse_args(argv)

    if args.command == "pack":
        try:
            p = load_pack(args.path)
        except (PackError, OSError) as e:
            print(f"invalid: {e}", file=sys.stderr)
            return 1
        print(f"ok: {p.name} {p.version}")
        return 0
    if args.command == "analyze":
        from .analyze import run
        return run(args)
    if args.command == "fit":
        from .fit import run
        return run(args)
    if args.command == "evaluate":
        from .evaluate import run
        return run(args)
    print(f"crownscope {args.command}: not implemented yet", file=sys.stderr)
    return 2
