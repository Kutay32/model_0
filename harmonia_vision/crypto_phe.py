"""
Homomorphic-style aggregation helpers for Harmonia Vision.

TenSEAL implements CKKS/BFV (not Paillier). For federated learning, this module
uses CKKS batched vectors to sum encrypted scaled weight chunks on the server.

When `tenseal` is unavailable or `use_phe=False`, training falls back to standard
FedAvg while still tracking communication sizes for benchmarking.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import tenseal as ts

    _HAS_TENSEAL = True
except ImportError:
    _HAS_TENSEAL = False
    ts = None  # type: ignore[assignment]


@dataclass
class SerializedEncryptedRound:
    """CKKS-encrypted weight payload (chunked) + shape metadata."""

    chunks: list[bytes]
    shapes: list[tuple[int, ...]]
    scale: float
    flat_len: int
    meta: dict[str, Any]


def _flatten_weights(weights: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    shapes = [w.shape for w in weights]
    flat = np.concatenate([w.reshape(-1).astype(np.float64) for w in weights])
    return flat, shapes


def _unflatten(flat: np.ndarray, shapes: list[tuple[int, ...]]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    off = 0
    for sh in shapes:
        n = int(np.prod(sh, dtype=np.int64))
        out.append(flat[off : off + n].reshape(sh))
        off += n
    return out


def make_ckks_context() -> Any:
    if not _HAS_TENSEAL:
        raise RuntimeError("tenseal is not installed")
    # Conservative CKKS setup; suitable for summing scaled weight vectors
    bits = [40, 21, 21, 21, 21, 40]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=bits)
    ctx.global_scale = 2**21
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def encrypt_weight_vector(weights: list[np.ndarray], scale: float, ctx: Any) -> SerializedEncryptedRound:
    flat, shapes = _flatten_weights(weights)
    scaled = (flat * scale).astype(np.float64)
    # CKKS: usable slots ~= poly_modulus_degree / 2 for this setup (8192 -> ~4096)
    chunk_size = min(len(scaled), 4096)
    if chunk_size <= 0:
        chunk_size = len(scaled)

    chunks: list[bytes] = []
    for start in range(0, len(scaled), chunk_size):
        vec = scaled[start : start + chunk_size]
        plain = ts.ckks_vector(ctx, vec.tolist())
        chunks.append(plain.serialize())

    return SerializedEncryptedRound(
        chunks=chunks,
        shapes=shapes,
        scale=float(scale),
        flat_len=int(len(scaled)),
        meta={"scheme": "ckks"},
    )


def add_encrypted_rounds(a: SerializedEncryptedRound, b: SerializedEncryptedRound, ctx: Any) -> SerializedEncryptedRound:
    if (
        a.shapes != b.shapes
        or a.scale != b.scale
        or a.flat_len != b.flat_len
        or len(a.chunks) != len(b.chunks)
    ):
        raise ValueError("Mismatched encrypted payloads")
    summed: list[bytes] = []
    for ca, cb in zip(a.chunks, b.chunks):
        va = ts.ckks_vector_from(ctx, ca)
        vb = ts.ckks_vector_from(ctx, cb)
        summed.append((va + vb).serialize())
    return SerializedEncryptedRound(
        chunks=summed, shapes=a.shapes, scale=a.scale, flat_len=a.flat_len, meta=a.meta
    )


def decrypt_weight_vector(blob: SerializedEncryptedRound, ctx: Any, total_examples: float) -> list[np.ndarray]:
    """Decrypt summed ciphertexts where each client encrypted `w_k * n_k` (FedAvg numerator)."""
    parts: list[float] = []
    for ch in blob.chunks:
        v = ts.ckks_vector_from(ctx, ch)
        parts.extend(v.decrypt())

    flat = np.asarray(parts, dtype=np.float64).ravel()
    flat = flat[: blob.flat_len]
    flat = flat / blob.scale / float(max(total_examples, 1e-8))
    return _unflatten(flat, blob.shapes)


def secure_aggregate_ckks(
    encrypted_payloads: list[bytes],
    secret_context_bytes: bytes,
    total_examples: float,
) -> tuple[list[np.ndarray], int]:
    """
    Sum encrypted client payloads and decrypt with the server-held secret context.
    Returns (averaged weight list, total bytes processed for comm accounting).
    """
    if not _HAS_TENSEAL:
        raise RuntimeError("tenseal not available")
    ctx = ts.context_from(secret_context_bytes)
    rounds = [pickle.loads(p) for p in encrypted_payloads]
    acc = rounds[0]
    for r in rounds[1:]:
        acc = add_encrypted_rounds(acc, r, ctx)
    avg = decrypt_weight_vector(acc, ctx, total_examples=total_examples)
    comm_bytes = sum(len(p) for p in encrypted_payloads)
    return avg, comm_bytes


def bytes_communicated_for_plain_ndarrays(weights: list[np.ndarray]) -> int:
    return int(sum(w.nbytes for w in weights))
