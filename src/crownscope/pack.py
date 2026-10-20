"""Pack loading and validation (SRS 4.7, 5.4). No thresholds or phrase lists live in code."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

KNOWN_SIGNALS = {
    "pattern", "style", "near_dup", "structure", "embed_sim",
    "ppl", "burst", "fast_detect", "binoculars", "novelty",
}
REQUIRED_KEYS = ("name", "version", "domain", "min_length", "signals", "weights", "thresholds")


class PackError(ValueError):
    pass


@dataclass
class Pack:
    name: str
    version: str
    domain: str
    min_length: int
    signals: list[str]
    weights: dict[str, dict[str, float]]  # length band -> signal -> weight
    thresholds: dict[str, float]
    options: dict = field(default_factory=dict)  # per-signal options, keyed by signal name
    cluster: dict = field(default_factory=dict)  # {min_span_hours}: floor for the arrival-rate span
    baseline: dict = field(default_factory=dict)  # population baseline, see baseline.py
    fit: dict = field(default_factory=dict)  # provenance written by `crownscope fit` (NFR-AUD-1)
    window: dict = field(default_factory=dict)
    lists: dict = field(default_factory=dict)

    @property
    def meta(self) -> dict:
        return {"name": self.name, "version": self.version}


def _validate_embed_sim(o: dict | None) -> None:
    if not o:
        raise PackError("embed_sim is enabled but options.embed_sim is missing")
    need = ("model", "min_chars", "chunk_chars", "max_chunks", "min_similarity", "lsh", "size", "density", "mix",
            "max_confidence", "reference_store_size", "evidence_cap")
    if missing := [k for k in need if k not in o]:
        raise PackError(f"options.embed_sim missing: {', '.join(missing)}")
    if not o["model"].get("name"):
        raise PackError("options.embed_sim.model.name is empty: name a sentence-embedding model (or a local directory)")
    if not 0 < o["min_similarity"] <= 1:
        raise PackError("options.embed_sim.min_similarity must be in (0, 1]")
    if o["chunk_chars"] < 1 or o["max_chunks"] < 1:
        raise PackError("options.embed_sim.chunk_chars and max_chunks must be at least 1")
    if not (o["lsh"].get("tables", 0) >= 1 and o["lsh"].get("bits", 0) >= 1):
        raise PackError("options.embed_sim.lsh needs tables >= 1 and bits >= 1")
    for key in ("size", "density"):
        if o[key]["lo"] == o[key]["hi"]:
            raise PackError(f"options.embed_sim.{key}: lo and hi must differ")
    if min(o["mix"].values()) < 0 or sum(o["mix"].values()) <= 0:
        raise PackError("options.embed_sim.mix needs non-negative values that are not all zero")


def _validate_ppl(o: dict | None) -> None:
    if not o:
        raise PackError("ppl is enabled but options.ppl is missing")
    need = ("model", "max_tokens", "max_chunks", "min_tokens", "perplexity", "full_confidence_tokens", "max_confidence")
    if missing := [k for k in need if k not in o]:
        raise PackError(f"options.ppl missing: {', '.join(missing)}")
    if not o["model"].get("name"):
        raise PackError("options.ppl.model.name is empty: name a causal language model (or a local directory)")
    if o["max_tokens"] < 8 or o["max_chunks"] < 1 or o["min_tokens"] < 2:
        raise PackError("options.ppl needs max_tokens >= 8, max_chunks >= 1 and min_tokens >= 2")
    ref = o["perplexity"]
    if not (ref.get("human", 0) > 0 and ref.get("machine", 0) > 0) or ref["human"] == ref["machine"]:
        raise PackError("options.ppl.perplexity needs positive, different `human` and `machine` values")


def _validate_burst(o: dict | None, ppl: dict | None) -> None:
    if not o:
        raise PackError("burst is enabled but options.burst is missing")
    need = ("max_tokens", "max_chunks", "min_sentences", "min_sentence_tokens", "spread",
            "full_confidence_sentences", "max_confidence")
    if missing := [k for k in need if k not in o]:
        raise PackError(f"options.burst missing: {', '.join(missing)}")
    if ppl is None:
        if not (o.get("model") or {}).get("name"):
            raise PackError("options.burst.model.name is empty: name a causal language model (or a local directory)")
    elif o.get("model") and o["model"].get("name") != ppl["model"].get("name"):
        raise PackError("options.burst.model and options.ppl.model name different models; a detector loads one "
                        "scoring model (leave burst.model out to use ppl's)")
    if o["max_tokens"] < 8 or o["max_chunks"] < 1:
        raise PackError("options.burst needs max_tokens >= 8 and max_chunks >= 1")
    if o["min_sentences"] < 2 or o["min_sentence_tokens"] < 2:
        raise PackError("options.burst needs min_sentences >= 2 and min_sentence_tokens >= 2")
    ref = o["spread"]
    if not (ref.get("human", 0) > 0 and ref.get("machine", 0) > 0) or ref["human"] == ref["machine"]:
        raise PackError("options.burst.spread needs positive, different `human` and `machine` values")


def _validate_novelty(o: dict | None, lists: dict) -> None:
    if not o:
        raise PackError("novelty is enabled but options.novelty is missing")
    need = ("min_unit_chars", "min_units", "min_chars", "full_confidence_chars", "reuse", "facts", "mix",
            "max_confidence", "reference_store_size", "evidence_cap")
    if missing := [k for k in need if k not in o]:
        raise PackError(f"options.novelty missing: {', '.join(missing)}")
    if o["min_unit_chars"] < 1 or o["min_units"] < 1:
        raise PackError("options.novelty.min_unit_chars and min_units must be at least 1")
    for key in ("reuse", "facts"):
        if o[key]["lo"] == o[key]["hi"]:
            raise PackError(f"options.novelty.{key}: lo and hi must differ")
    known = {r.get("name") for r in lists.get("concrete", [])}
    if unknown := [r for r in o["facts"]["rules"] if r not in known]:
        raise PackError(f"options.novelty.facts.rules names rules missing from lists.concrete: {', '.join(unknown)}")
    if set(o["mix"]) != {"reuse", "facts"} or min(o["mix"].values()) < 0 or sum(o["mix"].values()) <= 0:
        raise PackError("options.novelty.mix needs non-negative reuse and facts values that are not both zero")
    sem = o.get("semantic") or {}
    if sem.get("enabled"):
        need = ("model", "min_similarity", "max_sentences", "lsh")
        if missing := [k for k in need if k not in sem]:
            raise PackError(f"options.novelty.semantic missing: {', '.join(missing)}")
        if not sem["model"].get("name"):
            raise PackError("options.novelty.semantic.model.name is empty: name a sentence-embedding model "
                            "(or a local directory), or set semantic.enabled to false")
        if not 0 < sem["min_similarity"] <= 1:
            raise PackError("options.novelty.semantic.min_similarity must be in (0, 1]")
        if sem["max_sentences"] < 1:
            raise PackError("options.novelty.semantic.max_sentences must be at least 1")
        if not (sem["lsh"].get("tables", 0) >= 1 and sem["lsh"].get("bits", 0) >= 1):
            raise PackError("options.novelty.semantic.lsh needs tables >= 1 and bits >= 1")


def validate(data: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise PackError(f"pack missing keys: {', '.join(missing)}")
    unknown = set(data["signals"]) - KNOWN_SIGNALS
    if unknown:
        raise PackError(f"unknown signals: {', '.join(sorted(unknown))}")
    t = data["thresholds"]
    if not (0 <= t.get("medium", -1) <= t.get("high", -1) <= 1):
        raise PackError("thresholds must satisfy 0 <= medium <= high <= 1")
    for band, ws in data["weights"].items():
        absent = [s for s in data["signals"] if s not in ws]
        if absent:  # an enabled signal without a weight would be silently ignored
            raise PackError(f"weights[{band}] has no weight for enabled signal(s): {', '.join(absent)}")
    if "embed_sim" in data["signals"]:
        _validate_embed_sim(data.get("options", {}).get("embed_sim"))
    if "ppl" in data["signals"]:
        _validate_ppl(data.get("options", {}).get("ppl"))
    if "burst" in data["signals"]:
        opts = data.get("options", {})
        _validate_burst(opts.get("burst"), opts.get("ppl") if "ppl" in data["signals"] else None)
    if "novelty" in data["signals"]:
        _validate_novelty(data.get("options", {}).get("novelty"), data.get("lists", {}))
        opts = data.get("options", {})
        sem = (opts.get("novelty") or {}).get("semantic") or {}
        if sem.get("enabled") and "embed_sim" in data["signals"] and \
                sem["model"].get("name") != opts["embed_sim"]["model"].get("name"):
            raise PackError("options.novelty.semantic.model and options.embed_sim.model name different models; "
                            "a detector loads one embedding model")
    bad_opts = set(data.get("options", {})) - KNOWN_SIGNALS
    if bad_opts:
        raise PackError(f"options for unknown signals: {', '.join(sorted(bad_opts))}")
    for feature, cfg in data.get("options", {}).get("style", {}).get("features", {}).items():
        if cfg["weight"] < 0 or cfg["lo"] == cfg["hi"]:
            raise PackError(f"style feature {feature}: weight must be >= 0 and lo != hi")
    if data.get("cluster", {}).get("min_span_hours", 1) <= 0:
        raise PackError("cluster.min_span_hours must be > 0")
    bl = data.get("baseline")
    if bl:
        from .baseline import METRICS
        if set(bl["mad_floor"]) != set(METRICS) or any(v <= 0 for v in bl["mad_floor"].values()):
            raise PackError(f"baseline.mad_floor needs a positive value for each of {', '.join(METRICS)}")
        if bl["min_docs"] < 2 or bl["min_population"] < 3 or bl["z_threshold"] <= 0:
            raise PackError("baseline needs min_docs >= 2, min_population >= 3 and z_threshold > 0")
    opts = data.get("options", {})
    nd = opts.get("near_dup")
    if nd:
        if nd["permutations"] % nd["bands"]:
            raise PackError("near_dup.bands must divide near_dup.permutations")
        if not 0 < nd["min_similarity"] <= 1:
            raise PackError("near_dup.min_similarity must be in (0, 1]")
    for sig, key in (("near_dup", "matches"), ("structure", "repeats")):
        r = opts.get(sig, {}).get(key)
        if r and r["lo"] == r["hi"]:
            raise PackError(f"{sig}.{key}: lo and hi must differ")
    for rule in data.get("lists", {}).get("patterns", []):
        try:
            re.compile(rule["regex"])
        except (re.error, KeyError) as e:
            raise PackError(f"invalid pattern rule {rule!r}: {e}") from e
    for band, ws in data["weights"].items():
        bad = [s for s, w in ws.items() if s not in KNOWN_SIGNALS or w < 0]
        if bad:
            raise PackError(f"weights[{band}] invalid entries: {', '.join(bad)}")


def load_pack_data(ref: str | Path) -> dict:
    """The validated pack as plain data, for tools that rewrite a pack (e.g. `crownscope fit`)."""
    path = Path(ref)
    if path.suffix in (".yml", ".yaml") or path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = resources.files("crownscope.packs").joinpath(f"{ref}.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    validate(data)
    return data


def load_pack(ref: str | Path) -> Pack:
    """`ref` is a bundled pack name (blog, shopping) or a path to a YAML file."""
    data = load_pack_data(ref)
    return Pack(**{k: data[k] for k in Pack.__dataclass_fields__ if k in data})
