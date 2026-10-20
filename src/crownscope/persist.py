"""Save and reload a Cumulative Mode session (SRS FR-5.9, IR-5).

Format: JSON Lines (gzip if the path ends in .gz, detected by content on load). One header line, then one
line per stored document in insertion order, so eviction order survives a reload. No pickle: the file is
data only. It holds fingerprints and per-document records, never document text. Evidence that quotes the
text (phrases, fragments) is written only on request (NFR-7). The file is created readable by its owner only.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path

from .store import Entry

FORMAT = "crownscope-session"
VERSION = 5
READABLE_VERSIONS = (1, 2, 3, 4, 5)
TEXT_EVIDENCE = ("phrase", "fragment")  # evidence kinds that quote the document
# Options that define how a fingerprint is computed: stored signatures are only comparable if these match.
FINGERPRINT_OPTIONS = {
    "near_dup": ("shingle_size", "permutations"),
    "structure": ("opening_tokens", "closing_tokens", "paragraph_bucket"),
}
# Embeddings are comparable only for the same model and chunking. Unlike the fingerprints above they are
# optional per document, so this is compared only when the saved session actually holds embeddings.
EMBEDDING_OPTIONS = ("chunk_chars", "max_chunks")


class SessionFileError(ValueError):
    """The file is not a readable session, or it does not fit the pack it is loaded with."""


def fingerprint_spec(pack) -> dict:
    return {sig: ({k: pack.options[sig][k] for k in keys} if pack.options.get(sig) else None)
            for sig, keys in FINGERPRINT_OPTIONS.items()}


def novelty_spec(pack) -> dict | None:
    """What the stored sentence and fact hashes depend on: which sentences count, and the fact patterns."""
    nv = pack.options.get("novelty")
    if not nv or "novelty" not in pack.signals:
        return None
    rules = {r["name"]: r["regex"] for r in pack.lists.get("concrete", [])}
    return {"min_unit_chars": nv["min_unit_chars"], "facts": {n: rules.get(n) for n in nv["facts"]["rules"]}}


def semantic_novelty_spec(pack) -> dict | None:
    """What stored sentence vectors depend on: the model, and how many sentences are embedded."""
    sem = pack.options.get("novelty", {}).get("semantic")
    if not sem or not sem.get("enabled") or "novelty" not in pack.signals:
        return None
    return {"model": sem["model"].get("name"), "max_sentences": sem["max_sentences"]}


def embedding_spec(pack) -> dict | None:
    es = pack.options.get("embed_sim")
    if not es or "embed_sim" not in pack.signals:
        return None
    return {"model": es["model"].get("name"), **{k: es[k] for k in EMBEDDING_OPTIONS}}


def _entry_to_dict(e: Entry, include_evidence: bool) -> dict:
    return {
        "id": e.doc_id, "ts": e.ts,
        "minhash": base64.b64encode(e.minhash).decode("ascii") if e.minhash is not None else None,
        "structure": e.structure, "author": e.author, "source": e.source, "product": e.product,
        "cluster": e.cluster, "grade": e.grade, "score": e.score,
        "evidence": [list(ev) for ev in e.evidence if include_evidence or ev[1] not in TEXT_EVIDENCE],
        "chars": e.chars, "signals": [list(s) for s in e.signals] if e.signals is not None else None,
        "matches": e.matches, "repeats": e.repeats,
        "embedding": base64.b64encode(e.embedding).decode("ascii") if e.embedding is not None else None,
        "similar": e.similar, "similar_sum": e.similar_sum,
        "units": list(e.units), "facts": list(e.facts), "shared_mask": e.shared_mask, "unique_facts": e.unique_facts,
        "sentence_vectors": base64.b64encode(e.sentence_vectors).decode("ascii") if e.sentence_vectors else None,
        "sentence_dim": e.sentence_dim,
    }


def _entry_from_dict(d: dict) -> Entry:
    return Entry(
        doc_id=d["id"], ts=d["ts"],
        minhash=base64.b64decode(d["minhash"]) if d["minhash"] is not None else None,
        structure=d["structure"], author=d["author"], source=d["source"], product=d["product"],
        cluster=d["cluster"], grade=d["grade"], score=d["score"],
        evidence=tuple(tuple(ev) for ev in d["evidence"]),
        chars=d.get("chars"), signals=tuple(tuple(s) for s in d["signals"]) if d.get("signals") is not None else None,
        matches=d.get("matches", 0), repeats=d.get("repeats", 0),
        embedding=base64.b64decode(d["embedding"]) if d.get("embedding") is not None else None,
        similar=d.get("similar", 0), similar_sum=d.get("similar_sum", 0.0),
        units=tuple(d.get("units", ())), facts=tuple(d.get("facts", ())),
        shared_mask=d.get("shared_mask", 0), unique_facts=d.get("unique_facts", 0),
        sentence_vectors=base64.b64decode(d["sentence_vectors"]) if d.get("sentence_vectors") else None,
        sentence_dim=d.get("sentence_dim", 0),
    )


def save_session(session, path: str | Path, include_evidence: bool = False) -> None:
    """Write atomically: a crash mid-save leaves the previous file intact."""
    path = Path(path)
    store = session.store
    entries = store.entries()
    header = {
        "format": FORMAT, "version": VERSION,
        "pack": session.detector.pack.meta,
        "fingerprints": fingerprint_spec(session.detector.pack),
        "window": store.window, "hash_only": session.hash_only, "include_evidence": include_evidence,
        "counter": session._n, "seq": store.seq, "clusters": store.cluster_state(), "count": len(entries),
        # what the stored vectors came from, so a reload with another model is refused
        "embeddings": ({"spec": embedding_spec(session.detector.pack), "dim": store._embed_bytes // 4}
                       if store._embed_bytes else None),
        "novelty": novelty_spec(session.detector.pack) if any(e.units or e.facts for e in entries) else None,
        "novelty_semantic": ({"spec": semantic_novelty_spec(session.detector.pack), "dim": store._sentence_dim}
                             if store._sentence_dim else None),
    }
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as raw:
            out = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) if path.suffix == ".gz" else raw
            try:
                for obj in (header, *(_entry_to_dict(e, include_evidence) for e in entries)):
                    out.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            finally:
                if out is not raw:
                    out.close()
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_session(detector, path: str | Path, window: dict | None = None, hash_only: bool | None = None):
    """Rebuild a session saved with `save_session`, using `detector`'s pack.

    The pack must compute fingerprints the same way as when the file was saved (shingle size, number of
    permutations, structure rules); anything else about the pack, including thresholds, may have changed.
    `window` and `hash_only` override the saved values.
    """
    path = Path(path)
    try:
        with open(path, "rb") as probe:
            gz = probe.read(2) == b"\x1f\x8b"
        with (gzip.open(path, "rb") if gz else open(path, "rb")) as f:
            lines = [line for line in f.read().decode("utf-8").splitlines() if line]
        header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    except (OSError, IndexError, ValueError, EOFError) as e:  # gzip.BadGzipFile and JSON errors are ValueError/OSError
        raise SessionFileError(f"{path}: not a readable session file ({e})") from e
    if not isinstance(header, dict) or header.get("format") != FORMAT:
        raise SessionFileError(f"{path}: not a {FORMAT} file")
    required = ("version", "pack", "fingerprints", "window", "hash_only", "counter", "seq", "clusters", "count")
    if missing := [k for k in required if k not in header]:
        raise SessionFileError(f"{path}: header missing {', '.join(missing)}")
    if header.get("version") not in READABLE_VERSIONS:
        raise SessionFileError(f"{path}: unsupported version {header.get('version')!r}")
    if header["count"] != len(rows):
        raise SessionFileError(f"{path}: truncated, header says {header['count']} documents, found {len(rows)}")
    saved_emb = header.get("embeddings")
    if saved_emb and embedding_spec(detector.pack) != saved_emb["spec"]:
        raise SessionFileError(
            f"{path}: saved embeddings come from {saved_emb['spec']} but the pack uses {embedding_spec(detector.pack)}; "
            "vectors from different models cannot be compared")
    saved_sem = header.get("novelty_semantic")
    if saved_sem and semantic_novelty_spec(detector.pack) != saved_sem["spec"]:
        raise SessionFileError(
            f"{path}: saved sentence vectors come from {saved_sem['spec']} but the pack uses "
            f"{semantic_novelty_spec(detector.pack)}; vectors from different models cannot be compared")
    if header.get("novelty") and novelty_spec(detector.pack) != header["novelty"]:
        raise SessionFileError(
            f"{path}: saved sentence hashes were made with {header['novelty']} but the pack uses "
            f"{novelty_spec(detector.pack)}; stored documents cannot be compared with new ones")
    if header["fingerprints"] != fingerprint_spec(detector.pack):
        raise SessionFileError(
            f"{path}: saved fingerprints {header['fingerprints']} do not match the pack's "
            f"{fingerprint_spec(detector.pack)}; stored documents cannot be compared with new ones")

    session = detector.session(window=header["window"] if window is None else window,
                               hash_only=header["hash_only"] if hash_only is None else hash_only)
    try:
        session.store.restore([_entry_from_dict(r) for r in rows], header["clusters"], header["seq"])
    except (KeyError, TypeError, ValueError) as e:
        raise SessionFileError(f"{path}: inconsistent session data ({e})") from e
    if any(r.get("units") and "shared_mask" not in r for r in rows):
        session.store.rebuild_lexical_masks()  # saved before sentence masks existed: only a count was kept
    session._n = header["counter"]
    return session
