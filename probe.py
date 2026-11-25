"""probe.py — measure the REAL fused-target concentration on the deployed models.

The k decision (top-64 vs top-256) must come from real logits, not a simulation.
This module runs one **teacher-forced** forward per model (no generation, ~1 s on
an A10G for a 200-token prompt) and reports, per position:

    Z_k = retained probability mass of the top-k fused entries
    H   = entropy of the fused target (nats)
    tail = 1 - Z_k  ( == total-variation distance to the truncated target )

Alignment note (important, and why the numbers are split)
--------------------------------------------------------
Two tokenizers that differ in *merge rules* produce different token boundaries on
the same text, so "position 7" means different text prefixes.  Teacher-forced
targets are therefore only exactly comparable between models whose tokenizations
agree.  Limite and R1 are both Qwen2-family and share 151 660 of 151 667 surface
tokens — essentially the same tokenizer — so their fused target is exact.  K2 and
Qwen3.5 are reported **individually** (their own Z_k), not fused, for that reason.

This is a measurement tool, not a training path.
"""

from __future__ import annotations

import numpy as np
import torch

from spine import N_PRAGMA, build_spine, fuse, step_logprobs


@torch.inference_mode()
def probe(engine, prompt: str, ks=(8, 16, 32, 64, 128, 256, 512), max_len: int = 384) -> dict:
    """Measure concentration of the REAL teacher laws on **one common text**.

    Two bugs made an earlier version of this report garbage; both are guarded now:

    1. each model's own chat template injects different special tokens (Limite's
       adds a 30-token system prompt, R1 adds 2), so "position t" was different
       text.  We therefore tokenize the prompt **as raw text** in every tokenizer.
    2. positions were compared by token index, which means different prefixes in
       different tokenizers.  We compare by **byte boundary**, and fuse only where
       all teachers agree.

    Do NOT expect the fused law to be sharper than every teacher: with opposed
    teachers a product of experts is *flatter* (two 0.9-confident models that
    disagree fuse to 0.5/0.5).  The fused law is always exactly the normalised
    geometric mean of the restricted teacher laws — that identity is what
    `tests/test_probe.py` pins down.
    """
    from bytes_vocab import vocab_bytes
    from teacher import byte_positions

    names = list(engine.names)
    dev = next(engine.model[names[0]].parameters()).device
    V = len(engine.spine)

    encs, lp_all, ends = {}, {}, {}
    for n in names:
        ids = engine.tok[n](prompt, return_tensors="pt", truncation=True, max_length=max_len,
                            add_special_tokens=False)["input_ids"][0]
        out = engine.model[n](input_ids=ids[None, :].to(dev), use_cache=False)
        lp_all[n] = torch.log_softmax(out.logits[0].float(), -1).cpu().numpy()  # (L_n, V_n)
        encs[n] = ids
        ends[n] = byte_positions(engine.tok[n], ids)

    stats = {"spine": V, "per_model": {}, "fused": None,
             "text_bytes": len(prompt.encode()), "tokens_per_model": {n: int(encs[n].numel()) for n in names}}

    # per-model: each on its own token boundaries (honest, not cross-comparable)
    for n in names:
        z = np.stack([step_logprobs(lp_all[n][t], engine.spine, n, allow_answer=False)[:V]
                      for t in range(lp_all[n].shape[0])])
        p = np.exp(z - np.logaddexp.reduce(z, axis=1, keepdims=True))
        stats["per_model"][n] = _curve(p, ks)

    # fused: only at byte boundaries ALL teachers share, using the cached forwards
    common = set(ends[names[0]])
    for n in names[1:]:
        common &= set(ends[n])
    common = sorted(common)
    stats["shared_positions"] = len(common)
    if common:
        idx_of = {n: {e: i for i, e in enumerate(ends[n])} for n in names}
        rows = []
        for e in common:
            per = [step_logprobs(lp_all[n][idx_of[n][e]], engine.spine, n, allow_answer=False)[:V] for n in names]
            z = fuse(per, [1.0] * len(per), "product")[:V]
            rows.append(z - np.logaddexp.reduce(z))
        stats["fused"] = _curve(np.exp(np.stack(rows)), ks)
        stats["fused"]["models"] = names
        stats["fused"]["positions"] = len(common)
        # NOTE: product fusion is NOT necessarily sharper — mutually opposed
        # teachers fuse to something flatter (see tests/test_probe.py).  Report the
        # relationship instead of asserting the wrong invariant.
        per_H = {n: stats["per_model"][n]["entropy"] for n in names}
        stats["checks"] = {
            "fused_entropy": stats["fused"]["entropy"],
            "teacher_entropies": per_H,
            "fused_flatter_than_every_teacher": bool(
                stats["fused"]["entropy"] >= max(per_H.values()) - 1e-6),
            "interpretation": (
                "fused flatter ⇒ teachers disagree (fusion averages their positions); "
                "fused sharper ⇒ teachers agree (fusion concentrates).  Both are correct."
            ),
        }
    return stats


def _curve(p: np.ndarray, ks) -> dict:
    p = np.asarray(p, dtype=np.float64)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
    out = {"positions": int(p.shape[0])}
    for k in ks:
        if k >= p.shape[1]:
            continue
        idx = np.argpartition(-p, k - 1, axis=1)[:, :k]
        Z = np.take_along_axis(p, idx, axis=1).sum(1)
        out[f"Z@{k}"] = float(Z.mean())
        out[f"tv@{k}"] = float((1 - Z).mean())
        out[f"floor@{k}"] = float(np.mean(-np.log(np.maximum(Z, 1e-30))))
        out[f"p10@{k}"] = float(np.percentile(Z, 10))   # worst decile matters for training
    H = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
    out["entropy"] = float(H.mean())
    out["top1"] = float(p.max(1).mean())
    # sanity: a uniform distribution over V has entropy ln V; flag if we ever
    # accidentally re-normalise to uniform (that is exactly the bug of 2026-11-25)
    V = p.shape[1]
    out["near_uniform"] = bool(abs(out["entropy"] - np.log(V)) < 1e-3)
    return out


def verdict(stats: dict, floor_tol: float = 0.05) -> str:
    """Pick k: smallest k whose truncation floor (-log Z_k) is under `floor_tol`.

    Why 0.05 nats: the plan's acceptance threshold for the distilled student is
    KL < 0.35 nats (§6).  The truncation floor is a *systematic* bias that adds to
    whatever the student cannot learn, so it should be a small fraction of that
    budget — 0.05 leaves ~85 % of the budget for actual modelling error.

    The storage cost is reported alongside, because k=256 costs 4x k=64 (126 GB vs
    31.5 GB) and is only worth it if the floor actually clears the bar.
    """
    # Fused alone is not enough: the target the student learns is the FUSED law, but
    # a k that truncates one teacher's mass badly is still a warning sign, and the
    # earlier version silently dropped per_model whenever `fused` was present
    # (found by the selftest: a 0.90 floor in a per_model row was never reported).
    rows = ([stats["fused"]] if stats.get("fused") else []) + list((stats.get("per_model") or {}).values())
    if not rows:
        return "no measurement"
    ks = sorted({int(k.split("@")[1]) for r in rows for k in r if k.startswith("floor@")})
    gb = lambda k: 20_000 * 4096 * k * 6 / 1e9  # noqa: E731
    for k in ks:
        if all(r.get(f"floor@{k}", 9e9) <= floor_tol for r in rows):
            return f"**k = {k}** (floor <= {floor_tol} nats, storage {gb(k):.0f} GB for 20k x 4k tokens)"
    # Nothing clears the bar.  Report the LARGEST k measured (the best available),
    # but name the residual bias from the WORST row — quoting the fused number alone
    # would understate how far off the worst teacher is.
    best = ks[-1]
    fl = [r.get(f"floor@{best}", float("nan")) for r in rows]
    return (f"k = {best} still has a floor of {max(fl):.2f} nats (> {floor_tol}) at {gb(best):.0f} GB — "
            f"either accept that bias and report it, or store the **dense** target for the tail positions only")


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    """`verdict()` is the function that turned a curve into the k=256 decision, so it
    gets its own tests — including the cases where it must refuse to answer."""
    # a curve that clears 0.05 only at k=256 (the measured real shape)
    stats = {
        "fused": {"floor@64": 0.064, "floor@256": 0.030},
        "per_model": {"a": {"floor@64": 0.1, "floor@256": 0.02}},
    }
    v = verdict(stats)
    assert "k = 256" in v, v
    assert "126" in v, v                       # cost must be reported, not hidden

    # a curve that already clears at 64 must not be upgraded to 256
    v64 = verdict({"fused": {"floor@64": 0.01, "floor@256": 0.001}, "per_model": {}})
    assert "k = 64" in v64, v64

    # nothing clears the bar -> say so and name the residual bias
    vbad = verdict({"fused": {"floor@64": 0.5, "floor@128": 0.4}, "per_model": {}})
    assert "still has a floor" in vbad and "0.40" in vbad, vbad   # best available k
    # ...and the WORST row is the one quoted, not the first
    vworst = verdict({"fused": {"floor@128": 0.4}, "per_model": {"a": {"floor@128": 0.9}}})
    assert "0.90" in vworst, vworst

    # empty input must not raise
    assert verdict({}) == "no measurement"
    assert verdict({"fused": {}, "per_model": {}}) == "no measurement"

    # the tolerance is honoured in both directions
    tight = {"fused": {"floor@64": 0.03, "floor@256": 0.001}, "per_model": {}}
    assert "k = 64" in verdict(tight, floor_tol=0.05)
    assert "k = 256" in verdict(tight, floor_tol=0.01)

    print("probe selftest OK · verdict picks the smallest k · reports cost · refuses on empty")


if __name__ == "__main__":
    _selftest()
