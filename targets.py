"""targets.py — 蒸餾目標：一個分佈 = 一組 (idx, val)，其餘一律 -inf。

四種目標都喺呢個模組，全部係純 torch / numpy，**只需要 CPU**，
因為蒸餾最容易錯嘅地方係「目標係邊個位置的邊個分佈」，而唔係模型本身：

  TargetPair      (idx (n,k) int64, val (n,k) log-prob，已 renormalise)  ← 目前 Space 用嘅
  hard_target     一個 token（= CE / SFT），用嚟做 CE 錨定，防止學生漂走
  build_kl_mask   KL 只喺「學生同老師都講得出」的 token 上計，避免向 -inf 借懲罰

背景（實測）：
  * `fuse.py` 的 spine 係 4 個 tokenizer 的 byte 交集，|V| ≈ 42 184，
    所以 teacher 的 `z` 向量尾部有大量 -inf（spine 外 token）。
  * `sparse_kl()` 現行寫法只 gather top-k index，唔會踩到 -inf
    → 呢個係正確嘅（唔會 nan），但會**完全忽略** spine 外質量。
  * 若改成 dense KL，就必須用 mask；下面 `dense_kl_masked()` 係參考實作 + 測試對象。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NEG = -1e30           # 代替 -inf：令 softmax / logsumexp 唔會產生 nan（同 train_distill 的 -1e9 一致，取更負值）


# ---------------------------------------------------------------------------------------
# 1. top-k 稀疏目標（現行，k=256 由 §8.1 實測定）
# ---------------------------------------------------------------------------------------
def topk_target(logp: torch.Tensor, k: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., V) log-prob → (idx (..., k) int64, val (..., k) log-prob, renormalised)。

    尾部（spine 外 + 長尾）被丟棄；renormalise 之後 val 自己就是一個合法分佈，
    所以 `sparse_kl` 比較的是「截斷後的老師」而唔係「完整老師」——
    代價由 §8.1 量到：k=256 → 0.030 nats floor。
    """
    v, idx = torch.topk(logp, min(k, logp.shape[-1]), dim=-1)
    return idx, torch.log_softmax(v, dim=-1)


def spine_gather(student_logits: torch.Tensor, spine_ids: torch.Tensor) -> torch.Tensor:
    """Project student logits (..., V_student) onto the spine alphabet (..., |V|).

    **Why this must exist.** `topk_target` returns indices into the *spine* (~42k
    entries) because that is the only alphabet all four teachers share.  The
    student has its own 248k vocabulary.  Gathering the student's log-probs with
    spine indices silently reads the *wrong* 42k rows of the student vocabulary —
    that is the 49-nat floor the first real Space run showed.  Project first, then
    gather with spine indices; never mix the two index spaces.
    """
    sid = spine_ids.to(student_logits.device).long()
    lp = torch.log_softmax(student_logits.float(), dim=-1)
    sel = lp.index_select(-1, sid)           # (..., |V|)
    # The teacher's law is only defined ON the spine, so the student must be
    # renormalised there too.  Skip this and any mass the student puts outside
    # the spine is silently charged as KL — measured 1.69 nats on a toy case where
    # the student equals the teacher exactly on the spine.
    return torch.log_softmax(sel, dim=-1)


def sparse_kl(student_logits: torch.Tensor, idx: torch.Tensor, val: torch.Tensor,
              T: float = 2.0, spine_ids: torch.Tensor | None = None) -> torch.Tensor:
    """T^2 * KL(q_T || p_T) only on the stored top-k tokens (memory-safe).

    `spine_ids` (|V|,) maps each spine entry to the student's token id and is
    **required**: without it the gather reads the wrong rows (measured: 49-nat
    floor).  Both sides are softened to the same temperature (standard KD).
    """
    if spine_ids is None:
        raise ValueError("sparse_kl needs spine_ids: teacher indices live in the spine, "
                         "student logits in its own vocabulary — see spine_gather()")
    lp = torch.log_softmax(spine_gather(student_logits, spine_ids) / T, dim=-1)
    q = soften(val, T)                     # teacher softened too (standard KD)
    s = lp.gather(-1, idx.long())
    return (T * T) * (q.exp() * (q - s)).sum(-1)


# ---------------------------------------------------------------------------------------
# 2. dense masked KL（參考實作，用嚟驗證 sparse 版冇偷工減料）
# ---------------------------------------------------------------------------------------
def soften(logp: torch.Tensor, T: float) -> torch.Tensor:
    """把 log-分佈降溫 **兩邊都要做**：q_T = softmax(log q / T)。

    為何重要：Space 現行 `train_distill.sparse_kl` 只把**學生**除以 T，老師 q 唔降溫。
    咁樣「學生 = 老師」時 KL 唔會係 0（T=2 時實測 0.197 nats @ 隨機 logits），
    而 T² 前因子亦唔再係「還原梯度尺度」嘅那個 T²。
    標準 KD 係 `T²·KL(q_T ‖ p_T)`，兩邊同溫——下面的 `dense_kl_masked` 用呢個定義。
    """
    return torch.log_softmax(logp / T, dim=-1)


def dense_kl_masked(student_logits: torch.Tensor, teacher_logp: torch.Tensor,
                    T: float = 2.0) -> torch.Tensor:
    """KL(q ‖ p_student) 只喺 q 的 support 上計。

    `teacher_logp` 尾部係 NEG（≈ -e30）。直接 exp() 會係 0，0·(0 - s) = 0，
    所以 **NaN 風險來自 s 的 -inf**，唔係來自 q：
      - 若學生在某些格給 -inf（例如 masked head），-inf 的 p 會被 0·inf 拖出 nan；
      - 正常 softmax 唔會出 -inf，但 float16 的 log_softmax 可以（-1e4 → -inf）。
    因此 clamp 學生的 log-prob 下限，並用 `where` 遮罩，而唔係靠 IEEE 規則。
    """
    lq = soften(teacher_logp.clamp_min(NEG), T)
    q = lq.exp()
    ls = torch.log_softmax(student_logits.float() / T, dim=-1).clamp_min(-1e30)
    terms = torch.where(q > 0, q * (lq.clamp_min(-1e30) - ls), torch.zeros_like(q))
    kl = terms.sum(-1)
    return (T * T) * kl


def support_kl_masked(student_logits: torch.Tensor, teacher_logp: torch.Tensor,
                      T: float = 2.0) -> torch.Tensor:
    """同上，但寫成兩項相減的形式，方便逐項斷言（測試用）。"""
    lq = soften(teacher_logp.clamp_min(NEG), T)
    logz = torch.logsumexp(lq, dim=-1, keepdim=True)
    lqn = lq - logz                                                  # 歸一化後的 log q
    q = lqn.exp()
    ls = torch.log_softmax(student_logits.float() / T, dim=-1).clamp_min(-1e30)
    ent = torch.where(q > 0, -q * lqn, torch.zeros_like(q)).sum(-1)   # H(q)
    cross = torch.where(q > 0, -q * ls, torch.zeros_like(q)).sum(-1)  # −E_q[log p]
    return (T * T) * (cross - ent)


# ---------------------------------------------------------------------------------------
# 3. hard target（CE 錨定）
# ---------------------------------------------------------------------------------------
def hard_target(token_ids: torch.Tensor, vocab: int) -> torch.Tensor:
    """(n,) int64 → (n, vocab) log-prob，one-hot（-inf 其他地方）。

    用途：λ·CE(學生, 真答案) 同 KL 並用，避免學生只學會「平嘅融合分佈」
    而失去生成能力（計劃 §7「混入學生原有能力」）。
    """
    out = torch.full((token_ids.numel(), vocab), NEG)
    out[torch.arange(token_ids.numel()), token_ids.clamp(0, vocab - 1)] = 0.0
    return torch.log_softmax(out, dim=-1)


# ---------------------------------------------------------------------------------------
# 5. 真正的「同溫」目標：先建構 q_T 再截斷（課本 KD 的稀疏版）
# ---------------------------------------------------------------------------------------
def build_T_target(fused_logp: torch.Tensor, k: int = 256, T: float = 2.0
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(..., |V|) fused log-prob → (idx, val_T, tail_mass_T)。

    為何要呢個：`train_distill.sparse_kl` 係「截斷後才降溫」，即
      q_k = renormalise(π_k)        （只在 top-k 上）
      q_T = soften(softmax(q_k), T) （再降溫）
    咁樣做，**spine 外質量（10–18 %）會被靜靜哋重新分配**，T 越低影響越大。
    正確次序係先降溫再截斷：
      q_T = softmax(π / T)，val_T = q_T 在 top-k 上的 log，tail_mass_T = 1 − Σ val_T.exp()
    學生只需要在 top-k 上擬合 val_T，剩下 tail_mass_T 留給「其餘 token」——
    呢個 tail_mass_T 就係 `distill.py` 的 §3.3 方案 A/B 分野所在的質量。
    """
    pi = fused_logp.float().exp()                    # 已是合法分佈
    qT = torch.softmax(fused_logp.float() / float(T), dim=-1)
    v, idx = torch.topk(qT, min(k, qT.shape[-1]), dim=-1)
    tail = (1.0 - v.sum(-1)).clamp_min(0.0)
    val = v.clamp_min(1e-12).log()
    del pi
    return idx, val, tail


# ---------------------------------------------------------------------------------------
# 4. 自我測試
# ---------------------------------------------------------------------------------------
def distill_kl(student_logits: torch.Tensor, idxT: torch.Tensor, valT: torch.Tensor,
               spine_ids: torch.Tensor, T: float = 2.0, tail_mass: torch.Tensor | None = None,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """T^2 * KL(q || p) on the k stored tokens — the loss the trainer uses.

    Three separate bugs were found here by testing "does a perfect student score 0";
    all three produced a floor that could never train away.  Each line below is
    load-bearing:

    1. **Index spaces.** `idxT` indexes the *spine* (~42k entries, the only alphabet
       all four teachers share); the student indexes its own vocabulary (248k).
       `spine_gather` projects the student onto the spine first.  Gathering student
       log-probs with spine indices gave the 49-nat floor seen on the live Space.
    2. **One temperature.** `valT` comes from `build_T_target`, which already
       applied softmax(pi/T) before truncating.  Softening it again applies T twice
       and a perfect student scores 3.14 nats at T=2.
    3. **Same denominator.** The student must be normalised over the full spine and
       then *sliced* to the k tokens, exactly like `valT` is a slice of a
       full-support law.  Renormalising the student over the k tokens instead makes
       the two laws disagree (0.132 nats at k=4/8) even when they are identical.

    `tail_mass` is the teacher's mass outside the k tokens, added explicitly instead
    of being redistributed over k: at k=256 §8.1 measured it at 2.98 % = 0.030 nats.
    """
    ki = idxT.long()
    # student / T over the full spine, then slice — one temperature, full denominator
    p = torch.log_softmax(spine_gather(student_logits, spine_ids) / T, dim=-1)
    ps = p.gather(-1, ki)
    # teacher: valT is already softmax(pi/T) sliced to k.  No second softening.
    q = valT.float()
    kl = (q.exp() * (q - ps)).sum(-1)
    if pos_weight is not None:
        kl = kl * pos_weight
    out = (T * T) * kl
    return out if tail_mass is None else out + (T * T) * tail_mass.to(out.device)


def _selftest() -> None:
    torch.manual_seed(0)
    V = 64
    logits_t = torch.randn(3, 17, V) * 3
    t_logp = torch.log_softmax(logits_t, -1)
    # 尾部（例如 spine 外）標成 NEG
    t_logp[..., 40:] = NEG
    s_logits = torch.randn(3, 17, V)

    # (a) sparse == dense，當 k = V 而且 q 冇 NEG 格
    full = torch.log_softmax(logits_t, -1)
    idx, val = topk_target(full, V)
    a = sparse_kl(s_logits, idx, val, T=1.0, spine_ids=torch.arange(V))
    b = dense_kl_masked(s_logits, full, T=1.0)
    assert torch.allclose(a, b, atol=1e-4), (a[:2], b[:2])

    # (a2) THE regression that caught three separate bugs in distill_kl:
    #      a student that reproduces the teacher exactly must score 0 at every k, T.
    for _k in (4, 16, 64):
        for _T in (0.5, 1.0, 2.0):
            _ids = torch.randperm(V)      # a PERMUTATION: spine ids are sorted, but the
                                          # selftest must also hold for any bijection
            _logpi = torch.log_softmax(torch.randn(2, V) * 2, -1)
            _stu = torch.full((2, V), float("-inf"))
            _stu[:, _ids] = _logpi
            _idx, _val, _tail = build_T_target(_logpi, k=_k, T=_T)
            _kl = distill_kl(_stu, _idx, _val, _ids, T=_T)
            assert _kl.max() < 1e-5, (_k, _T, _kl.max())

    # (b) 學生 = 老師 ⇒ KL = 0（任何 T）
    for T in (0.5, 1.0, 2.0):
        z = dense_kl_masked(logits_t, full, T=T)
        assert z.abs().max() < 1e-4, (T, z.abs().max())

    # (c) NEG 尾部：dense masked KL 有限、唔會 nan，而且等於「先截斷再計」
    c = dense_kl_masked(s_logits, t_logp, T=1.0)
    assert torch.isfinite(c).all(), c
    trunc = full.clone()
    trunc[..., 40:] = NEG
    d = dense_kl_masked(s_logits, trunc, T=1.0)
    assert torch.allclose(c, d, atol=1e-5)

    # (d) support_kl_masked 同 dense_kl_masked 一致
    e = support_kl_masked(s_logits, t_logp, T=1.0)
    assert torch.allclose(c, e, atol=1e-4), (c[:2], e[:2])

    # (e) 學生在 support 上完全錯 ⇒ KL 爆炸（確認梯度有信號）
    bad = s_logits.clone()
    bad[..., :40] = -s_logits[..., :40] - 50.0
    assert dense_kl_masked(bad, t_logp, T=1.0).mean() > c.mean() + 1.0

    # (f) hard target：CE 對正確 token = 0，錯誤 token > 0
    ids = torch.tensor([3, 40, 5])
    ht = hard_target(ids, V)
    assert torch.isfinite(ht).all() and (ht.exp().sum(-1) - 1).abs().max() < 1e-5
    ce_ok = -(ht.exp() * torch.log_softmax(ht, -1)).sum(-1)          # 自己 vs 自己 ⇒ 0
    assert ce_ok.abs().max() < 1e-4, ce_ok
    wrong = torch.log_softmax(torch.randn(3, V) * 5, -1)
    assert (-(ht.exp() * wrong).sum(-1) > 1.0).all()

    # (g) topk_target 的 val 係合法分佈
    _, v2 = topk_target(t_logp, 16)
    assert (v2.exp().sum(-1) - 1).abs().max() < 1e-5

    # (h) build_T_target：T=1 且 k=|V| ⇒ tail = 0，val = log q
    fp = torch.log_softmax(torch.randn(2, 5, 40) * 2, -1)
    i1, v1, t1 = build_T_target(fp, k=40, T=1.0)
    assert t1.abs().max() < 1e-5, t1
    # idx 係排序後的，所以要 scatter 返去原本的 token 位置才比得
    back = torch.full_like(fp.exp(), -1.0).scatter(-1, i1, v1.exp())
    assert torch.allclose(back, fp.exp(), atol=1e-5)
    # T>1 令分佈變平：top-1 機率下降，tail 隨 k 增加
    i2, v2, t2 = build_T_target(fp, k=8, T=2.0)
    assert (v2.exp()[..., 0].mean() < fp.exp().max(-1).values.mean()), "T=2 應該更平"
    assert (t2 > 0).all()
    i3, v3, t3 = build_T_target(fp, k=38, T=2.0)
    assert (t3.mean() < t2.mean()), "k 越大 tail 越小"
    # 次序正確：先降溫再截斷 ⇒ val 必須等於 softmax(fp/T) 的 top-k（唔係 renormalise 之後再降溫）
    qT = torch.softmax(fp / 2.0, -1)
    vv, _ = torch.topk(qT, 8, dim=-1)
    assert torch.allclose(v2.exp(), vv, atol=1e-6), "val 必須係 softmax(fp/T) 的 top-k"

    print("targets selftest OK · sparse==dense · KL(student=teacher)=0 · NEG-safe · "
          "support KL · hard target · top-k renormalised")


if __name__ == "__main__":
    _selftest()


# ---------------------------------------------------------------------------------------
# 5. THE canonical loss used by the trainer
# ---------------------------------------------------------------------------------------


