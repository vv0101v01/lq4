"""sparse_targets.py — store teacher targets as top-k, and measure what that costs.

Why this file exists
--------------------
A dense target for one 4 096-token sequence is `4096 × 42 207 × 2 B = 346 MB`.
Twenty thousand of them would be 6.9 TB.  Keeping only the k largest entries
per position costs `4096 × k × 6 B` (fp16 log-prob + int32 id) = 1.6 MB at k=64.

What "cost" means, exactly
--------------------------
The obvious metric ("KL between the dense and truncated target") is **infinite**:
the truncated target puts exactly zero on everything outside the top-k, so any
dense mass there diverges.  The two finite, exact quantities are:

    Z_k  = retained mass = sum of the top-k probabilities   (1 - Z_k = tail)
    TV(q, q'_k) = 1 - Z_k        where q'_k = q / Z_k on the top-k, 0 outside

TV = (1/2)·Σ|q − q'| = (1/2)·[(1/Z−1)·Z + (1−Z)] = 1 − Z.  `tv()` below checks
this identity numerically, so the reported number is verified, not asserted.

`rel_loss_floor(k)` = −log Z_k is the *extra* cross-entropy a student pays if it
matches q perfectly on the top-k while q's truth has mass in the tail: it is the
lower bound on the truncation's effect on the training objective.
"""

from __future__ import annotations

import numpy as np

DTYPE_BYTES = 6  # fp16 log-prob (2) + int32 token id (4)


def topk_encode(q: np.ndarray, k: int) -> dict:
    """q: (L, V) probabilities (rows sum to 1). Returns ids/lp/retained per row."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    k = max(1, min(int(k), q.shape[1]))
    idx = np.argpartition(-q, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(q, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1).astype(np.int32)
    vals = np.take_along_axis(vals, order, axis=1)
    Z = vals.sum(1)
    lp = np.log(np.maximum(vals, 1e-30)).astype(np.float16)
    return {"ids": idx, "logprob": lp, "retained": Z}


def topk_decode(enc: dict, V: int, floor_lp: float = -30.0) -> np.ndarray:
    """Rebuild a dense log-prob matrix: top-k as stored, the rest at `floor_lp`."""
    ids, lp = enc["ids"], enc["logprob"].astype(np.float64)
    L = ids.shape[0]
    out = np.full((L, V), floor_lp, dtype=np.float64)
    np.put_along_axis(out, ids.astype(np.int64), lp, axis=1)
    return out


def retained_curve(q: np.ndarray, ks=(8, 16, 32, 64, 128, 256, 512)) -> dict[int, float]:
    return {int(k): float(np.mean(topk_encode(q, k)["retained"])) for k in ks}


def tv(q: np.ndarray, k: int) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    enc = topk_encode(q, k)
    Z = enc["retained"]
    qp = np.zeros_like(q)
    rows = np.arange(q.shape[0])[:, None]
    qp[rows, enc["ids"].astype(np.int64)] = q[rows, enc["ids"].astype(np.int64)] / Z[:, None]
    return 0.5 * np.abs(q - qp).sum(1)


def rel_loss_floor(q: np.ndarray, k: int) -> float:
    Z = topk_encode(q, k)["retained"]
    return float(np.mean(-np.log(np.maximum(Z, 1e-30))))


def bytes_for(n_tokens: int, k: int, n_seq: int = 1) -> float:
    return n_tokens * k * DTYPE_BYTES * n_seq / 1e6  # MB


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)

    # 1. TV identity: 1 - Z, verified numerically (this is the whole point of the file)
    for alpha in (0.05, 0.3, 2.0):  # 0.05 = very peaked, 2.0 = flat
        q = rng.dirichlet(np.full(1000, alpha), size=64)
        q /= q.sum(1, keepdims=True)
        for k in (4, 16, 64):
            Z = topk_encode(q, k)["retained"]
            assert np.allclose(tv(q, k), 1 - Z, atol=1e-12), (alpha, k)
            # tv in [0,1] always
            assert (tv(q, k) >= -1e-12).all() and (tv(q, k) <= 1 + 1e-12).all()

    # 2. peaked distributions are cheap to store: k=64 should retain nearly everything
    # a realistic LM head at moderate temperature: top-1 ~40 %, top-64 > 99.9 %
    peaked = np.exp(rng.normal(0, 8, (64, 2000)))
    peaked /= peaked.sum(1, keepdims=True)
    Z64 = topk_encode(peaked, 64)["retained"].mean()
    assert Z64 > 0.999, Z64
    # a uniform-ish distribution cannot be compressed: Z_k ≈ k/V
    flat = np.full((8, 2000), 1 / 2000)
    Zf = topk_encode(flat, 64)["retained"].mean()
    assert abs(Zf - 64 / 2000) < 1e-9, Zf

    # 3. encode/decode round-trip preserves the top-k log-probs exactly
    q = np.exp(rng.normal(0, 2, (16, 500)))
    q /= q.sum(1, keepdims=True)
    enc = topk_encode(q, 32)
    dense = topk_decode(enc, 500)
    rows = np.arange(16)[:, None]
    got = dense[rows, enc["ids"].astype(np.int64)]
    want = np.log(np.maximum(q[rows, enc["ids"].astype(np.int64)], 1e-30))
    assert np.allclose(got, want, atol=2e-3), np.abs(got - want).max()  # fp16 storage
    # everything else sits at the floor
    mask = np.ones_like(dense, dtype=bool)
    mask[rows, enc["ids"].astype(np.int64)] = False
    assert np.allclose(dense[mask], -30.0)

    # 4. loss floor: peaked -> ~0, uniform -> log(V/k)
    assert rel_loss_floor(peaked, 64) < 1e-3
    assert abs(rel_loss_floor(flat, 64) - np.log(2000 / 64)) < 1e-9

    # 5. storage arithmetic used in the plan
    assert abs(bytes_for(4096, 64) - 1.573) < 0.01, bytes_for(4096, 64)
    assert abs(bytes_for(4096, 64, 20_000) / 1000 - 31.46) < 0.05, bytes_for(4096, 64, 20_000) / 1000

    print("sparse_targets selftest OK · TV == 1-Z verified · round-trip exact to fp16 · "
          f"k=64 → {bytes_for(4096, 64, 20_000)/1000:.1f} GB for 20k×4k")


if __name__ == "__main__":
    _selftest()
