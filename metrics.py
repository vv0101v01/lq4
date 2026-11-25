"""metrics.py — the §6 gate arithmetic, separated from the Gradio plumbing.

`evaluate_student` used to do five jobs in one 128-line function: pick the teacher
subset, build the shapes, run the per-row loop, format the Markdown table and write
the JSON report.  Only the middle two are interesting, and they were untestable
because reaching them required a GPU and four Hugging Face downloads.

Everything here is pure: tensors in, numbers out.  `tests/test_metrics.py` pins the
invariants that matter — a perfect student scores KL 0 and top-1 100 %, and the gate
booleans flip exactly at the documented thresholds.
"""

from __future__ import annotations

import torch


def aligned_pairs(student_offsets: list[int], teacher_offsets: list[int]) -> list[tuple[int, int]]:
    """Pairs `(student_pos - 1, teacher_pos)` whose character boundaries coincide.

    Next-token prediction: the teacher's target at position j is the token *after*
    teacher offset j, so it is compared against the student's prediction at
    `m[o] - 1` — the position that predicts the token starting at char `o`.
    """
    m = {o: i for i, o in enumerate(student_offsets)}
    return [(m[o] - 1, j) for j, o in enumerate(teacher_offsets) if o in m and m[o] - 1 >= 0]


def score_positions(student_logits: torch.Tensor, *, ki: torch.Tensor, kv: torch.Tensor,
                    spine_ids: torch.Tensor, T: float = 1.0) -> tuple[torch.Tensor, int]:
    """(`per-position KL`, `number of top-1 agreements`) for one row.

    `student_logits` is ALREADY restricted to the aligned positions and has shape
    (n, V_student).  Indexing it a second time by position is the out-of-range gather
    that tripped a CUDA device-side assert.
    """
    from targets import distill_kl

    kl = distill_kl(student_logits, ki, kv, spine_ids, T=T)
    fused_top = ki.gather(-1, kv.argmax(-1, keepdim=True)).squeeze(-1)  # spine index
    stu_top = student_logits.index_select(-1, spine_ids).argmax(-1)     # spine index
    return kl, int((fused_top == stu_top).sum())


# The §6 gates are constants here, not scattered through the UI string.
KL_GATE = 0.35      # nats
TOP1_GATE = 70.0    # percent
FLOOR_GATE = 0.01   # nats (reference only — truncation bias, not a defect)


def verdict(kl_mean: float, top1_pct: float, floor: float) -> dict:
    """`{gate: (value, passed)}` — the single place the thresholds are applied."""
    return {
        "kl": (kl_mean, kl_mean < KL_GATE),
        "top1": (top1_pct, top1_pct >= TOP1_GATE),
        "floor": (floor, floor < FLOOR_GATE),
    }


def summary(per_row: list[dict]) -> dict:
    """Aggregate per-row results the same way `evaluate_student` reports them.

    KL is weighted by positions (it is a sum over positions); top-1 is a plain
    position average.  Mixing the two — averaging per-row means — would let a row
    with 2 positions carry as much weight as one with 200.
    """
    total = sum(r["positions"] for r in per_row)
    if total == 0:
        return {"positions": 0, "kl": None, "top1": None, "floor": None}
    kl = sum((r["kl"] or 0.0) * r["positions"] for r in per_row) / total
    t1 = 100.0 * sum((r["top1"] or 0.0) * r["positions"] for r in per_row) / total
    floors = [r["floor"] for r in per_row if r.get("floor") is not None]
    return {"positions": total, "kl": kl, "top1": t1,
            "floor": (sum(floors) / len(floors)) if floors else 0.0}


def table(kl_mean: float, top1_pct: float, floor: float) -> str:
    """The Markdown gate table.  Kept here so tests can assert on the exact words."""
    def row(name, value, ok):
        return f"| {name} | {value} | {'✅ PASS' if ok else '❌ FAIL'} |"

    v = verdict(kl_mean, top1_pct, floor)
    return "\n".join([
        "| gate | value | |",
        "|---|---|---|",
        row("KL(student ‖ fused)", f"{v['kl'][0]:.4f} nats", v["kl"][1]),
        row("top-1 agreement", f"{v['top1'][0]:.1f} %", v["top1"][1]),
        row("truncation floor (reference)", f"{v['floor'][0]:.4f} nats", v["floor"][1]),
    ])


def _selftest() -> None:
    torch.manual_seed(0)
    V, VS, n = 512, 2048, 40
    spine_ids = torch.arange(V)
    # a student that reproduces the fused law exactly must score 0 and 100 %
    tgt = torch.log_softmax(torch.randn(n, V) * 2, -1)
    from targets import build_T_target

    ki, kv, _ = build_T_target(tgt, k=64, T=1.0)
    perfect = torch.full((n, VS), -1e4)
    perfect[:, :V] = tgt
    kl, agree = score_positions(perfect, ki=ki, kv=kv, spine_ids=spine_ids)
    assert kl.max() < 1e-4, kl.max()
    assert agree == n, agree

    # offsets: the student predicts at m[o]-1, and position 0 is never a target
    assert aligned_pairs([1, 5, 9], [5, 9]) == [(0, 0), (1, 1)]
    assert aligned_pairs([1, 5], [0, 5]) == [(0, 1)]

    # gates flip exactly at the thresholds
    assert verdict(0.34, 70.1, 0.0)["kl"][1] and verdict(0.34, 70.1, 0.0)["top1"][1]
    assert not verdict(0.35, 70.0, 0.0)["kl"][1]      # strict <
    assert verdict(0.35, 70.0, 0.0)["top1"][1]        # inclusive >=

    # positional weighting: a 2-position row must not outweigh a 200-position row
    s = summary([{"positions": 200, "kl": 1.0, "top1": 0.5, "floor": 0.0},
                 {"positions": 2, "kl": 100.0, "top1": 0.0, "floor": 0.0}])
    assert abs(s["kl"] - (200 * 1.0 + 2 * 100.0) / 202) < 1e-6, s
    assert "❌ FAIL" in table(2.0, 40.0, 0.0) and "✅ PASS" in table(2.0, 40.0, 0.0)


if __name__ == "__main__":
    _selftest()
    print("metrics: all invariants hold")
