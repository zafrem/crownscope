"""Sentence embeddings for the `embed_sim` signal (SRS FR-3.5, NFR-3, NFR-6, NFR-9).

Nothing here loads a model unless a pack enables `embed_sim`. The model is an `Embedder`: anything with
`name`, `dim` and `embed(texts) -> vectors`. A pack names a provider and a model (`sentence-transformers`
with a model name or, for an on-premises install with no network, a local directory); code may also pass
an embedder to `Detector(pack, embedder=...)`.

Documents are embedded in chunks of `chunk_chars` (at most `max_chunks`), the unit vectors averaged and
renormalised, so long documents do not exceed a model's input limit. Vectors are stored as little-endian
float32.

Neighbour lookup avoids comparing a document with the whole window: random-hyperplane LSH. Each of `tables`
tables hashes a vector to `bits` sign bits (planes derived from a hash, so they are identical on every
run); documents sharing a bucket in any table are candidates and are verified by exact cosine similarity.
"""
from __future__ import annotations

import hashlib
import math
import operator
import struct
import sys
from array import array
from collections import defaultdict
from functools import lru_cache
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------- vectors

def pack_vector(vec) -> bytes:
    a = array("f", vec)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def unpack_vector(data: bytes) -> array:
    a = array("f")
    a.frombytes(data)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def quantize(vec) -> bytes:
    """A unit vector as signed bytes (one per component, scaled by 127): a quarter of the float32 size. Used for
    per-sentence vectors, where a document stores a dozen of them. Similarity from these is accurate to about 1%."""
    return struct.pack(f"{len(vec)}b", *[max(-127, min(127, round(x * 127))) for x in vec])


def unpack_q(data: bytes) -> array:
    a = array("b")
    a.frombytes(data)
    return a


def qcosine(a, b) -> float:
    return sum(map(operator.mul, a, b)) / (127 * 127)


def normalize(vec) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else list(vec)


def cosine(a, b) -> float:
    """Dot product; vectors are unit length, so this is the cosine similarity."""
    return sum(map(operator.mul, a, b))


def embed_document(embedder: Embedder, text: str, chunk_chars: int, max_chunks: int) -> list[float]:
    chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)][:max_chunks] or [text]
    vecs = [normalize(v) for v in embedder.embed(chunks)]
    if len(vecs) != len(chunks) or any(len(v) != embedder.dim for v in vecs):
        raise ValueError(f"embedder returned {len(vecs)} vectors of size "
                         f"{sorted({len(v) for v in vecs})} for {len(chunks)} chunks of dim {embedder.dim}")
    return normalize([sum(col) for col in zip(*vecs)])


# ---------------------------------------------------------------- providers

class SentenceTransformerEmbedder:
    """Wraps the `sentence-transformers` package (the `embed` extra). Not exercised by the test suite, which
    has no model available offline; the interface it implements is."""

    def __init__(self, name: str, local_only: bool = False, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # ImportError is reported by load_embedder
        kwargs = {"local_files_only": True} if local_only else {}
        self.model = SentenceTransformer(name, device=device, **kwargs)
        self.name = name
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()


def load_embedder(spec: dict | None) -> tuple[Embedder | None, str | None]:
    """(embedder, None) or (None, reason). A model that cannot be loaded is a missing signal, not a crash."""
    if not spec or not spec.get("name"):
        return None, "no embedding model configured (set options.embed_sim.model.name)"
    provider = spec.get("provider", "sentence-transformers")
    if provider != "sentence-transformers":
        return None, f"unknown embedding provider {provider!r}"
    try:
        return SentenceTransformerEmbedder(spec["name"], spec.get("local_only", False), spec.get("device")), None
    except ImportError as e:
        return None, f"sentence-transformers is not usable ({e}); install the `embed` extra"
    except Exception as e:  # missing model files, no network in local_only mode, out of memory ...
        return None, f"could not load embedding model {spec['name']!r}: {e}"


# ---------------------------------------------------------------- LSH

@lru_cache(maxsize=4)
def _planes(dim: int, count: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """`count` pseudo-random Gaussian directions in `dim` dimensions, derived from a hash: no RNG state."""
    planes = []
    for p in range(count):
        values: list[float] = []
        block = 0
        while len(values) < dim:
            raw = hashlib.blake2b(f"{seed}:{p}:{block}".encode(), digest_size=64).digest()
            u = struct.unpack(">8Q", raw)
            for i in range(0, 8, 2):  # Box-Muller: two uniforms -> one Gaussian
                u1, u2 = (u[i] + 1) / (2 ** 64 + 1), u[i + 1] / 2 ** 64
                values.append(math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2))
            block += 1
        planes.append(tuple(values[:dim]))
    return tuple(planes)


class EmbeddingIndex:
    def __init__(self, get_vector, tables: int = 12, bits: int = 8, seed: int = 0):
        self._get = get_vector  # doc_id -> packed vector (the store's entry is the single copy)
        self.tables, self.bits, self.seed = tables, bits, seed
        self._buckets: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._keys: dict[str, list[tuple[int, int]]] = {}

    def _signature(self, vec) -> list[tuple[int, int]]:
        planes = _planes(len(vec), self.tables * self.bits, self.seed)
        keys = []
        for t in range(self.tables):
            key = 0
            for b in range(self.bits):
                key = (key << 1) | (sum(map(operator.mul, vec, planes[t * self.bits + b])) > 0)
            keys.append((t, key))
        return keys

    def add(self, doc_id: str, packed: bytes) -> None:
        keys = self._signature(unpack_vector(packed))
        self._keys[doc_id] = keys
        for k in keys:
            self._buckets[k].add(doc_id)

    def remove(self, doc_id: str) -> None:
        for k in self._keys.pop(doc_id, ()):
            bucket = self._buckets[k]
            bucket.discard(doc_id)
            if not bucket:
                del self._buckets[k]

    def __len__(self) -> int:
        return len(self._keys)

    def query(self, vec, min_similarity: float, exclude: str | None = None) -> list[tuple[str, float]]:
        """(doc_id, cosine) at or above `min_similarity`, most similar first."""
        candidates: set[str] = set()
        for k in self._signature(vec):
            candidates |= self._buckets.get(k, set())
        found = []
        for doc_id in candidates - {exclude}:
            sim = cosine(vec, unpack_vector(self._get(doc_id)))
            if sim >= min_similarity:
                found.append((doc_id, sim))
        return sorted(found, key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------- sentence vectors (novelty)

try:  # optional: the `embed` extra brings numpy; everything below gives identical results without it
    import numpy as _np
except ImportError:  # pragma: no cover - depends on the environment
    _np = None

_Q_ONE = 127 * 127  # int8 dot product of two identical unit vectors, to scale a similarity back to ~1


@lru_cache(maxsize=4)
def _int_planes(dim: int, count: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """Integer hyperplanes (Gaussian directions scaled by 32 and rounded). Integers, so the sign of a dot with an
    int8 vector is exact: the same buckets with or without numpy, on any machine."""
    return tuple(tuple(round(32 * x) for x in plane) for plane in _planes(dim, count, seed))


@lru_cache(maxsize=4)
def _int_planes_np(dim: int, count: int, seed: int):
    return _np.array(_int_planes(dim, count, seed), dtype=_np.int32)


class SentenceIndex:
    """LSH over int8 sentence vectors: a document contributes up to a dozen, so it must stay cheap.

    Same scheme as `EmbeddingIndex` (sign bits of random hyperplanes, `tables` tables of `bits` bits, candidates
    verified exactly), but every operation is integer arithmetic. With numpy the signature and the verification
    are matrix products; without it they are the same sums in pure Python, just slower. A signature is computed
    once per vector and reused when the vector is then added.

    Size `bits` to the window: a random pair shares a table bucket with probability 2**-bits, so candidates per
    query are about tables * sentences / 2**bits.
    """

    def __init__(self, get_vector, tables: int = 24, bits: int = 12, seed: int = 0):
        self._get = get_vector  # key -> packed int8 vector (the store's entry is the single copy)
        self.tables, self.bits, self.seed = tables, bits, seed
        self._buckets: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._keys: dict[str, list[tuple[int, int]]] = {}
        self._cache: dict[bytes, list[tuple[int, int]]] = {}

    def _signature(self, q: bytes) -> list[tuple[int, int]]:
        if q in self._cache:
            return self._cache[q]
        dim, count = len(q), self.tables * self.bits
        if _np is not None:
            positive = (_int_planes_np(dim, count, self.seed) @ _np.frombuffer(q, dtype=_np.int8).astype(_np.int32)) > 0
            positive = positive.tolist()
        else:
            vec = unpack_q(q)
            positive = [sum(map(operator.mul, vec, plane)) > 0 for plane in _int_planes(dim, count, self.seed)]
        keys = []
        for t in range(self.tables):
            key = 0
            for b in range(self.bits):
                key = (key << 1) | positive[t * self.bits + b]
            keys.append((t, key))
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[q] = keys
        return keys

    def add(self, key: str, q: bytes) -> None:
        keys = self._signature(q)
        self._keys[key] = keys
        for k in keys:
            self._buckets[k].add(key)

    def remove(self, key: str) -> None:
        for k in self._keys.pop(key, ()):
            bucket = self._buckets[k]
            bucket.discard(key)
            if not bucket:
                del self._buckets[k]

    def __len__(self) -> int:
        return len(self._keys)

    def query(self, q: bytes, min_similarity: float) -> list[tuple[str, float]]:
        """(key, cosine) of stored vectors at or above `min_similarity`, most similar first."""
        candidates: set[str] = set()
        for k in self._signature(q):
            candidates |= self._buckets.get(k, set())
        if not candidates:
            return []
        keys = sorted(candidates)
        if _np is not None:
            matrix = _np.frombuffer(b"".join(self._get(k) for k in keys), dtype=_np.int8).reshape(len(keys), -1)
            dots = (matrix.astype(_np.int32) @ _np.frombuffer(q, dtype=_np.int8).astype(_np.int32)).tolist()
        else:
            vec = unpack_q(q)
            dots = [sum(map(operator.mul, vec, unpack_q(self._get(k)))) for k in keys]
        found = [(k, d / _Q_ONE) for k, d in zip(keys, dots) if d / _Q_ONE >= min_similarity]
        return sorted(found, key=lambda x: (-x[1], x[0]))
