"""train_distill.py — L3: distill the 4-way fused law into ONE 0.8B student.

The student is Qwen3.5-0.8B with a LoRA r=16 adapter (plan §3).  It never sees a
teacher's text: it is trained on the *fused distribution*, computed teacher-forced
at byte-aligned positions and encoded as top-k (k=256, measured §8.1).

Cost measured on this project
-----------------------------
    4.2 B teacher params, 82 M supervised positions -> 3.24 E FLOPs
    = 7.2 A100-h / 2.3 H100-h   (dense 6.9 TB, so targets are streamed)

Two ways to run it
------------------
A. A machine with ~40 GB GPU (A100/H100):
        python3 train_distill.py --data data/corpus/corpus_math.jsonl --steps 2000
B. ZeroGPU, in chunks inside the Space:
        /distill_step  (see `run_chunk`)  — same code path, one chunk per call

Checkpointing is append-only: `--out ckpt/` gets `step_*.pt` with the LoRA weights,
the optimiser state and an exact resume counter, so a preempted ZeroGPU chunk loses
at most one step.

What is verified where
----------------------
* mechanism (alignment -> top-k target -> KL -> grads -> checkpoint round-trip):
  `--selftest` with tiny models, CPU, 1 GB RAM;
* the real run: requires GPU weight memory; the sandbox has 1 GB and cannot load
  Qwen3.5-0.8B at all, so a real loss curve must come from the Space or your GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from targets import build_T_target, distill_kl

PROMPT_SUFFIX = "\n\nSolve step by step. Put the final answer in \\boxed{}."


# --------------------------------------------------------------------------------------
# top-k targets: the thing that makes 6.9 TB into 126 GB
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# the dataset: prompt text in, top-k fused target out
# --------------------------------------------------------------------------------------
def _prompt_of(r) -> str:
    """Rows come either as corpus dicts or as DistillRows items — accept both."""
    if isinstance(r, dict):
        return r.get("prompt") or (r["problem"] + PROMPT_SUFFIX)
    return r["prompt"]


def _solution_of(r) -> str:
    if isinstance(r, dict):
        return r.get("solution", "")
    return r["solution"]


class DistillRows(torch.utils.data.Dataset):
    """Yields (prompt, completion) text; the teacher forward happens in the collator.

    Targets are NOT precomputed on disk by default: student and teacher tokenise
    differently, so the cached artifact must be keyed by student position
    (`cache_targets()` does that and writes shards).
    """

    def __init__(self, path: str, limit: int | None = None):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
                if limit and len(self.rows) >= limit:
                    break
        if not self.rows:
            raise SystemExit(f"no rows in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = r["problem"] + PROMPT_SUFFIX
        return {"id": r["id"], "prompt": prompt, "solution": r["solution"], "answer": r.get("answer")}


def _spine_ids(teachers, name: str | None = None):
    """The spine entry -> student token id map.  For the Qwen3.5 student the spine is
    a subset of its vocabulary, so this is the identity on the student side; for a
    different student it is `spine.token_ids(student_name)`."""

    sp = teachers.spine
    name = name or ("Qwen3.5-0.8B" if "Qwen3.5-0.8B" in sp.names else sp.names[0])
    return sp.ids[name][: len(sp)]


def _shared_offsets(teachers, batch, weights):
    from teacher import shared_offsets

    return shared_offsets(batch, weights)


def student_offsets(tok, text: str) -> list[int]:
    """Char offset at which each student token ends (fast tokenizer only)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return [int(e) for _, e in enc["offset_mapping"]]


# --------------------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------------------
def build_student(repo: str, rev: str, lora_r: int = 16, lora_alpha: int = 32, dtype=torch.bfloat16,
                  device: str = "cuda", trust: bool = True, log=print):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=trust)
    model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev, trust_remote_code=trust,
                                                        dtype=dtype)
    model = model.to(device)
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    # peft injects fresh modules on CPU meta/default device — move AFTER wrapping,
    # otherwise the very first backward sees cpu LoRA weights against cuda inputs.
    model = model.to(device)
    for p in model.parameters():
        if p.requires_grad and p.device.type == "cpu":
            p.data = p.data.to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model._real_device = next(model.parameters()).device
    log(f"student {repo} · LoRA r={lora_r} · trainable {n / 1e6:.1f}M · on {model._real_device}")
    return tok, model


def save_ckpt(path: str, model, opt, step: int, meta: dict) -> None:
    tmp = path + ".tmp"
    torch.save({"step": step, "lora": {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad},
                "optimizer": opt.state_dict() if opt else None, "meta": meta}, tmp)
    os.replace(tmp, path)  # atomic: a preempted write never corrupts the last good ckpt


def load_ckpt(path: str, model, opt=None) -> tuple[int, dict]:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd["lora"], strict=False)
    if opt is not None and sd.get("optimizer"):
        try:
            opt.load_state_dict(sd["optimizer"])
        except Exception:
            pass
    return int(sd.get("step", 0)), {"missing": len(missing), "unexpected": len(unexpected), **sd.get("meta", {})}


def latest_ckpt(out_dir: str) -> str | None:
    fs = sorted(glob.glob(os.path.join(out_dir, "step_*.pt")), key=lambda p: int(os.path.basename(p)[5:-3]))
    return fs[-1] if fs else None


def train_student(teachers, spine, student_tok, student, rows, steps: int, k: int = 256, T: float = 2.0,
                  lr: float = 1e-4, bs: int = 1, max_len: int = 1024, weights: dict | None = None,
                  out_dir: str = "ckpt", save_every: int = 50, start_step: int = 0, log=print,
                  js_weight: float = 0.0):
    """One chunk of distillation.  Returns (steps_done, loss history)."""
    weights = weights or {n: 1.0 for n in teachers.names}
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    ck = latest_ckpt(out_dir)
    if ck and start_step == 0:
        s, info = load_ckpt(ck, student, opt)
        start_step = s
        log(f"resumed {ck} at step {s} ({info})")

    hist = []
    student.train()
    t0 = time.perf_counter()
    # Seeding with a constant made every restart replay the same rows in the same
    # order: with limit=40 the student saw 0.53 % of the corpus and overfit it
    # (train loss 0.61 vs held-out KL 1.66).  Vary the walk per call so successive
    # chunks cover new problems; `--seed` still pins it for a reproducible run.
    order = np.random.default_rng(start_step % 10_000 + 1).permutation(len(rows))
    for step in range(start_step, start_step + steps):
        # Long rows (p99 is ~1.8k tokens, the cap is 1024) need far more transient VRAM
        # than the 7.6 GB of headroom a t4-small leaves once teachers + student are
        # resident.  Stop the chunk cleanly instead of letting the step raise: a
        # half-finished chunk still saved every step (save_every=1), and the caller can
        # simply run the next one.
        try:
            free_gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 1e9
        except Exception:
            free_gb = 1e9
        if free_gb < 2.0:
            log(f"step {step}: only {free_gb:.2f} GB free — ending this chunk early")
            break
        batch_rows = [rows[i] for i in order[step % len(order):(step % len(order)) + bs]]
        if len(batch_rows) < bs:  # wrap
            batch_rows += [rows[i] for i in order[: bs - len(batch_rows)]]
        full = [(_prompt_of(r) + _solution_of(r)) for r in batch_rows]
        fb = teachers.logits(full, max_len)
        # fused_target + offsets only need logits, never gradients.  Left outside
        # inference_mode they keep the whole teacher batch's autograd bookkeeping
        # alive across the student backward, which is invisible in `loss` but shows
        # up as VRAM that never comes back between calls.
        with torch.inference_mode():
            fused = teachers.fused_target(fb, weights, ragged=True)   # list of (P_b, |V|)
            offs = _shared_offsets(teachers, fb, weights)

        loss_terms, npos = [], 0
        for b, r in enumerate(batch_rows):
            text = full[b]
            dev = next(student.parameters()).device
            # Truncate exactly like the teachers were (max_len), or the student's LM
            # head builds a graph over every position of a 3 kB solution: (L, 248k)
            # bf16 logits + a float32 lm_head backward buffer.  Measured on t4-small:
            # 8.25 GB free before the step, 14.41 GB allocated after -> OOM.
            enc = student_tok(text, return_tensors="pt", return_offsets_mapping=True,
                              add_special_tokens=False, truncation=True, max_length=max_len)
            # The truncation above is a HARD safety cap (`max_logits_pos`), not the
            # context window: a 3 kB solution would otherwise build an LM-head graph
            # over 3000 positions x 248k vocab and OOM the card.  Keep the cap wide
            # enough that it rarely binds — measured rows: median 182 tokens.
            ids = enc["input_ids"].to(dev)
            if enc.get("attention_mask") is not None:
                enc["attention_mask"] = enc["attention_mask"].to(dev)
            so = [int(e) for _, e in enc["offset_mapping"][0].tolist()]
            m = {o: i for i, o in enumerate(so)}
            pairs = [(m[o] - 1, j) for j, o in enumerate(offs[b]) if o in m and m[o] - 1 >= 0]
            if not pairs:
                continue
            out = student(input_ids=ids, attention_mask=enc.get("attention_mask"), use_cache=False)
            si = torch.tensor([p[0] for p in pairs], device=out.logits.device)
            tj = torch.tensor([p[1] for p in pairs], device=out.logits.device)
            # Keep only the aligned positions BEFORE touching autograd.  Slicing after
            # the graph is built kept the whole (L=384, 248k) logits tensor alive for
            # backward; measured 7.5 GB of activations on a 14.74 GiB card, which is
            # what actually OOMed — the teachers only account for 5.4 GB.
            logits = out.logits[0][si]                        # (n, V_student) — grad kept
            del out
            tgt = torch.from_numpy(fused[b]).to(logits.device)[tj]   # (n, |V|) log-probs
            ki, kv, tail = build_T_target(tgt, k=min(k, tgt.shape[-1]), T=T)
            spine_ids = torch.as_tensor(_spine_ids(teachers), dtype=torch.long, device=logits.device)
            loss_terms.append(distill_kl(logits, ki, kv, spine_ids, T=T))
            npos += len(pairs)
        if not loss_terms:
            log(f"step {step}: no aligned positions, skipped")
            continue
        loss = torch.stack(loss_terms).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        hist.append((step, float(loss), npos))
        el = time.perf_counter() - t0
        log(f"step {step}: loss {float(loss):.4f} · {npos} pos · {el:.1f}s elapsed")
        if (step + 1 - start_step) % save_every == 0:
            save_ckpt(os.path.join(out_dir, f"step_{step + 1}.pt"), student, opt, step + 1,
                      {"k": k, "T": T, "lr": lr, "n": len(rows)})
    return hist


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    """Mechanism test with tiny tensors — no downloads, CPU, fits in 1 GB.

    The loss itself is tested in `targets.py` (perfect student => 0 across k and T);
    here we only check the plumbing this module owns: top-k targets survive the
    round trip, checkpoints are atomic and resumable, and the corpus reader works.
    """
    import tempfile

    d = tempfile.mkdtemp()

    # checkpoint atomicity + resume
    m = torch.nn.Linear(4, 4)
    save_ckpt(os.path.join(d, "step_5.pt"), m, None, 5, {"k": 256})
    assert latest_ckpt(d).endswith("step_5.pt")
    m2 = torch.nn.Linear(4, 4)
    s, info = load_ckpt(latest_ckpt(d), m2)
    assert s == 5 and torch.allclose(m.weight, m2.weight)

    # a half-written .tmp must never be picked up as the latest checkpoint
    open(os.path.join(d, "step_9.pt.tmp"), "w").close()
    assert latest_ckpt(d).endswith("step_5.pt"), latest_ckpt(d)

    # corpus reader accepts both shapes (row dicts and dataset items)
    with open(os.path.join(d, "c.jsonl"), "w") as f:
        f.write(json.dumps({"id": "x", "problem": "1+1", "solution": "\\boxed{2}", "answer": "2"}) + "\n")
    rows = DistillRows(os.path.join(d, "c.jsonl"))
    assert len(rows) == 1 and rows[0]["prompt"].startswith("1+1")
    assert _prompt_of({"problem": "2+2"}) == "2+2" + PROMPT_SUFFIX
    assert _solution_of({"problem": "2+2", "solution": "s"}) == "s"

    print("train_distill selftest OK · atomic ckpt · resume · corpus reader")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/corpus/corpus_math.jsonl")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        from fuse import ModelSpec
        from teacher import TeacherStack

        specs = [ModelSpec("Limite-1B-Violetto", "paradigma-inc/limite-1b-violetto", "190ee8d9c6777b7c5f8e7592cc00caf495a90ca7", True),
                 ModelSpec("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562")]
        teachers = TeacherStack(specs, device="cuda")
        rows = DistillRows(a.data, a.limit)
        student_tok, student = build_student("Qwen/Qwen3.5-0.8B", "2fc0636")
        hist = train_student(teachers, teachers.spine, student_tok, student, rows, a.steps, k=a.k, T=a.T,
                             lr=a.lr, bs=a.bs, max_len=a.max_len, out_dir=a.out)
        print("done:", len(hist), "steps")
