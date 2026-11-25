"""distill.py — L3: compress the 6-in-1 ensemble into ONE student model.

The Space runs L2 (exact logit fusion) live.  L3 asks: can one 0.8B checkpoint
reproduce what the four teachers decide together?  Everything needed already
exists in this package:

* `spine.py` maps each teacher's 64k/151k/248k vocabulary onto the 42 207-token
  shared alphabet — so "the fused law" is a well-defined distribution there;
* `fuse.py` computes it at decode time; here we compute it **teacher-forced**
  (one forward per sequence, no autoregressive sampling), which is what makes
  distillation affordable: ~100x more supervised positions per GPU-second.

Two objects:

    fused_target_logprobs(logits, spine, names, weights, rule, T)
        -> (B, L, |V|) log-probs, the teacher signal

    distill_loss(student_logits, target_logp, spine_ids, T, weight=None)
        -> T^2 * KL(q || p) summed over the spine support

The student is Qwen3.5-0.8B: its 248 077-token vocabulary is a strict superset of
the spine, so no teacher signal is lost in projection (unlike the reverse).

Self-test runs offline (no network, no checkpoints): see `__main__`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from spine import N_PRAGMA, fuse, step_logprobs


def fused_target_logprobs(logits: list[torch.Tensor], spine, names: list[str], weights: list[float],
                          rule: str = "product (log-linear)") -> torch.Tensor:
    """Teacher signal: the L2 fused law at every position, on the spine.

    logits: one (B, L, V_m) tensor per teacher, ALREADY indexed to that teacher's
    vocabulary (caller slices with the model's own vocab size).
    returns (B, L, |V|) log-probs, pragma channels dropped and renormalised.
    """
    B, L = logits[0].shape[:2]
    V = len(spine)
    out = torch.empty((B, L, V), dtype=torch.float32, device=logits[0].device)
    z = torch.empty((B, L, V + N_PRAGMA), dtype=torch.float32, device=logits[0].device)
    for b in range(B):
        for t in range(L):
            per = []
            for name, lg in zip(names, logits):
                lp = torch.log_softmax(lg[b, t].float(), -1).cpu().numpy()
                per.append(step_logprobs(lp, spine, name, allow_answer=False))
            z[b, t] = torch.from_numpy(fuse(per, weights, rule)).to(out.device)
    out[:] = torch.log_softmax(z[..., :V], dim=-1)  # drop pragma, renormalise
    return out


def distill_loss(student_logits: torch.Tensor, target_logp: torch.Tensor, spine_ids: torch.Tensor,
                 T: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
    """T^2 * KL(q_teacher || p_student) over the shared alphabet only.

    student_logits: (B, L, V_student) raw logits of the student
    target_logp:    (B, L, |V|) teacher log-probs (from fused_target_logprobs)
    spine_ids:      (|V|,) the student's token id for each spine entry
    weight:         optional (B, L) per-position weights (e.g. 1 + lambda*JS)
    """
    V = spine_ids.numel()
    student_lp = torch.log_softmax(student_logits.float() / T, dim=-1)
    gathered = student_lp.index_select(-1, spine_ids.to(student_lp.device))  # (B, L, |V|)
    # a student that puts exactly zero mass on a spine token would otherwise give
    # log(0) inside the support; floor like the target does.
    gathered = torch.log_softmax(gathered.clamp_min(-60.0), dim=-1)
    q = target_logp.to(gathered.device).exp().clamp_min(1e-12)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    kl = (q * (q.clamp_min(1e-12).log() - gathered)).sum(-1)  # (B, L)
    if weight is not None:
        kl = kl * weight
    return (T * T) * (kl.mean() if reduction == "mean" else kl.sum())


def js_disagreement(logits: list[torch.Tensor], spine, names: list[str]) -> torch.Tensor:
    """Per-position teacher disagreement (nats) — use as a loss weight: positions
    where the teachers disagree are exactly where fusion adds information."""
    B, L = logits[0].shape[:2]
    out = torch.zeros((B, L), dtype=torch.float32)
    for b in range(B):
        for t in range(L):
            per = []
            for name, lg in zip(names, logits):
                lp = torch.log_softmax(lg[b, t].float(), -1).cpu().numpy()
                per.append(step_logprobs(lp, spine, name, allow_answer=False))
            P = np.stack([np.exp(p) for p in per])
            M = P.mean(0)
            out[b, t] = float(np.mean([0.5 * np.sum(Pi * (np.log(np.maximum(Pi, 1e-300)) - np.log(np.maximum(M, 1e-300)))) for Pi in P]))
    return out


# --------------------------------------------------------------------------------------
# self-test: two teachers, one student that IS the fusion of them => loss 0
# --------------------------------------------------------------------------------------
def _selftest() -> None:
    from spine import SharedSpine

    strings = [b"a", b"b", b"c"]
    ids = {"T1": np.array([0, 1, 2]), "T2": np.array([0, 2, 1]), "S": np.array([5, 6, 7])}
    # feeds must index this *teacher*'s fake vocab (sizes V1=4, V2=3)
    feeds = {"T1": {"think": [3], "end": [], "answer": [3]},
             "T2": {"think": [1], "end": [], "answer": [1]}}
    spine = SharedSpine(["T1", "T2"], strings, {k: ids[k] for k in ("T1", "T2")}, feeds)

    torch.manual_seed(0)
    V1, V2, VS = 4, 3, 10
    lg1 = torch.randn(1, 5, V1)
    lg2 = torch.randn(1, 5, V2)

    target = fused_target_logprobs([lg1, lg2], spine, ["T1", "T2"], [0.5, 0.5])
    assert target.shape == (1, 5, 3), target.shape
    assert torch.allclose(target.exp().sum(-1), torch.ones(1, 5), atol=1e-5)
    assert (target <= 0).all()

    # a student whose spine logits equal T * target has log_softmax == target ⇒ KL 0,
    # for every temperature (T^2 scaling must not move the optimum)
    s_ids = torch.tensor(ids["S"])
    for T in (0.5, 1.0, 2.0):
        student = torch.full((1, 5, VS), -20.0)
        student[..., s_ids] = target * T
        loss0 = distill_loss(student, target, s_ids, T=T)
        assert float(loss0) < 1e-5, (T, float(loss0))

    # perturbing the student raises the loss and produces gradients
    T = 2.0
    student2 = (torch.full((1, 5, VS), -20.0)).requires_grad_(True)
    with torch.no_grad():
        student2[..., s_ids] = target * T
        student2[..., int(s_ids[0])] += 1.5
    loss1 = distill_loss(student2, target, s_ids, T=T)
    assert float(loss1.detach()) > 1e-4
    loss1.backward()
    assert student2.grad is not None and float(student2.grad.abs().sum()) > 0

    # a *wrong* spine mapping costs more than the right one
    student3 = torch.full((1, 5, VS), -20.0)
    student3[..., s_ids] = target.flip(-1) * T
    right = float(distill_loss(student.clone(), target, s_ids, T=T))
    wrong = float(distill_loss(student3, target, s_ids, T=T))
    assert wrong > right + 1e-3, (right, wrong)

    # per-position weights are applied
    w = torch.tensor([[1.0, 2.0, 1.0, 1.0, 1.0]])
    lw = distill_loss(student3, target, s_ids, weight=w)
    assert float(lw) > 0

    # disagreement: identical teachers ⇒ JS 0
    js = js_disagreement([lg1, lg1], spine, ["T1", "T1"])
    assert float(js.abs().max()) < 1e-6

    print("distill selftest OK · target on spine sums to 1 · KL(teacher||teacher)=0 · grads flow")


if __name__ == "__main__":
    _selftest()
