"""teacher.py — forward-only teacher stack for L3 distillation.

`FusionEngine` (fuse.py) is built for *interactive* decoding: it holds four
cached models and samples step by step.  Distillation needs the opposite:

  * one teacher-forced forward per (model, sequence) → (B, L, V_m)
  * **no** autoregressive loop, no sampling, no KV-cache juggling
  * batched, `inference_mode`, and cheap enough that 82 M positions is 7 A100-h

`TeacherStack` is that, sharing the model loading and the compatibility patches
with the live app so there is exactly **one** description of how these four
checkpoints load (the Space's app.py imports the same class).

Typical use
-----------
    stack = TeacherStack(SPECS, dtype=torch.bfloat16, device="cuda")
    out = stack.logits(prompts)          # list of dicts: ids, logits, lens
    q = stack.fused_target(out, weights=[1,1,1,1])
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from bytes_vocab import token_to_bytes

from spine import N_PRAGMA, build_spine, fuse, step_logprobs


@dataclass
class TeacherBatch:
    """Raw teacher output for one batch of prompts."""

    names: list[str]
    ids: dict[str, torch.Tensor]  # name -> (B, L_n) model token ids
    logits: dict[str, torch.Tensor]  # name -> (B, L_n, V_n) fp32
    lens: dict[str, list[int]]


class TeacherStack:
    """Loads N causal LMs and produces teacher-forced logits (never samples)."""

    def __init__(self, specs, dtype=torch.bfloat16, device: str = "cuda", log=print, attn: str | None = None):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.specs = {s.name: s for s in specs}
        self.order = [s.name for s in specs]
        self.dtype = dtype
        self.device = device
        self.tok, self.model, self.stops = {}, {}, {}

        for name in self.order:
            s = self.specs[name]
            t0 = time.perf_counter()
            try:
                kw = dict(revision=s.rev, trust_remote_code=s.trust)
                cls = AutoModelForImageTextToText if s.kind == "vlm" else AutoModelForCausalLM
                tok = AutoTokenizer.from_pretrained(s.repo, **kw)
                extra = {"attn_implementation": attn} if attn else {}
                model = cls.from_pretrained(s.repo, dtype=dtype, **kw, **extra)
                model = model.to(device).eval()
                for p in model.parameters():
                    p.requires_grad_(False)  # teachers are frozen by construction
                self.tok[name] = tok
                self.model[name] = model
                self.stops[name] = _stop_ids(tok)
                log(f"teacher {name}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B · {time.perf_counter() - t0:.0f}s")
            except Exception as e:  # a missing teacher must not kill the run
                log(f"teacher {name} FAILED: {type(e).__name__}: {e}")

        if not self.model:
            raise RuntimeError("no teacher loaded")
        self.names = [n for n in self.order if n in self.model]
        self.spine = build_spine({n: (self.tok[n], self.stops[n]) for n in self.names})
        _TOK_OF.update(self.tok)

    # ------------------------------------------------------------------ encoding
    def encode(self, prompts: list[str], max_len: int = 2048) -> dict[str, torch.Tensor]:
        """Each teacher tokenizes the same surface text with its own tokenizer."""
        out = {}
        for n in self.names:
            tok = self.tok[n]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False)
            # left-pad so the last position is always the final prompt token
            out[n] = enc
        return out

    # ------------------------------------------------------------------- forward
    @torch.inference_mode()
    def logits(self, prompts: list[str], max_len: int = 2048) -> TeacherBatch:
        # the device is whatever the weights actually live on: reusing an engine's
        # stack via __new__ can leave self.device as a string that disagrees with reality
        dev = next(iter(self.model.values())).device if torch.cuda.is_available() else torch.device("cpu")
        encs = self.encode(prompts, max_len)
        ids, logits, lens = {}, {}, {}
        for n in self.names:
            enc = encs[n]
            inp = enc["input_ids"].to(dev)
            attn = enc.get("attention_mask")
            attn = attn.to(dev) if attn is not None else torch.ones_like(inp)
            out = self.model[n](input_ids=inp, attention_mask=attn, use_cache=False)
            ids[n] = inp.cpu()
            logits[n] = out.logits.float().cpu()
            lens[n] = attn.sum(-1).tolist()
        return TeacherBatch(self.names, ids, logits, lens)

    # ------------------------------------------------------- the distillation target
    def fused_target(self, batch: TeacherBatch, weights: dict[str, float] | None = None,
                     rule: str = "product (log-linear)", allow_answer: bool = False, log=print,
                     ragged: bool = False):
        """(B, L, |V|) log-probs of the fused law, for positions whose tokenizations agree.

        Positions are **byte-aligned**: only token boundaries that every teacher
        shares are used, so "next token" means the same unit of text for all of
        them.  Measured yield on real text: 88-100 % of the shortest model's
        tokens (95.4 % pooled) — see `probe.py` / the Space's Diagnostics tab.

        Falls back to nothing: if no shared boundary exists it raises, because a
        mis-aligned target is worse than no target.
        """
        return fused_target(batch, self.spine, weights or {n: 1.0 for n in batch.names}, rule,
                            tok_of=self.tok, log=log, ragged=ragged)

    def per_model_target(self, batch: TeacherBatch, name: str) -> np.ndarray:
        V = len(self.spine)
        B, L = batch.logits[name].shape[:2]
        out = np.empty((B, L, V), dtype=np.float32)
        for b in range(B):
            for t in range(L):
                lp = torch.log_softmax(batch.logits[name][b, t], -1).cpu().numpy()
                z = step_logprobs(lp, self.spine, name, allow_answer=False)[:V]
                out[b, t] = z - np.logaddexp.reduce(z)
        return out


def _special_ids(tok) -> set[int]:
    """Ids that are formatting, not surface text.

    `token_to_bytes("<eos>")` returns b"<eos>" — every character of a special token
    is a printable ASCII byte, so the byte table cannot tell `<eos>` from the
    literal text "<eos>".  Measured consequence before this fix: a sequence
    ["ab", "<eos>", "c"] reported offsets [2, 7, 8] instead of [2, 2, 3], i.e. every
    later position was shifted by the tokenizer's own bookkeeping.  The tokenizer
    knows which ids are special, so ask it.
    """
    ids: set[int] = set()
    for name in ("all_special_ids", "additional_special_tokens_ids"):
        v = getattr(tok, name, None)
        if v:
            ids.update(int(i) for i in v)
    return ids


def byte_positions(tok, ids: torch.Tensor) -> list[int]:
    """For one sequence: byte offset (in the *surface text*) at which each token ends.

    Different tokenizers split the same text at different places, so a token index
    is not comparable across models — a **byte offset** is.  Token k of model A
    covers bytes [ends[k-1], ends[k]); we only compare positions where a boundary
    in A coincides with a boundary in B.
    """
    specials = _special_ids(tok)
    out, pos = [], 0
    for i in ids.tolist():
        i = int(i)
        if i in specials:
            out.append(pos)      # formatting token: contributes no surface bytes
            continue
        s = tok.convert_ids_to_tokens(i)
        b = token_to_bytes(s) if isinstance(s, str) else None
        if b is None:            # non-byte key (some added tokens): same treatment
            out.append(pos)
            continue
        pos += len(b)
        out.append(pos)
    return out


def shared_boundaries(ends_a: list[int], ends_b: list[int]) -> list[tuple[int, int]]:
    """Pairs (i, j) such that token i of A and token j of B end at the same byte."""
    ib = {e: i for i, e in enumerate(ends_a)}
    return [(ib[e], j) for j, e in enumerate(ends_b) if e in ib]


def fused_target(batch: TeacherBatch, spine, weights: dict[str, float], rule: str = "product (log-linear)",
                 tok_of: dict | None = None, log=print, ragged: bool = False):
    """Fuse teachers at every position where **all** their token boundaries coincide.

    batch.ids are each teacher's own ids; `tok_of` supplies the tokenizer needed to
    compute surface byte offsets (defaults to spine.vocabs if present).

    Returns (B, P, |V|) **only when every row has the same number of aligned
    positions**.  Rows differ in length, so batching distinct texts makes that
    rarely true — pass `ragged=True` for the general case, which returns a list of
    (P_b, |V|) arrays.  (Measured: stacking without this raised
    "all input arrays must have the same shape" on the very first evaluation run.)
    """
    from bytes_vocab import token_to_bytes

    names = [n for n in batch.names if n in weights]
    if len(names) < 2:
        raise ValueError("need at least two teachers for a fused target")
    if tok_of is None:
        raise ValueError("fused_target needs tok_of={name: tokenizer} to align on surface bytes")

    V = len(spine)
    B = batch.ids[names[0]].shape[0]
    per_batch = []
    for b in range(B):
        ends = {n: byte_positions(tok_of[n], batch.ids[n][b]) for n in names}
        # intersect byte boundaries across all teachers
        common = set(ends[names[0]])
        for n in names[1:]:
            common &= set(ends[n])
        common = sorted(common)
        if not common:
            raise ValueError("no shared byte boundaries between teachers on this batch")
        pos = []
        for e in common:
            idx = {n: ends[n].index(e) for n in names}   # token index ending at this byte
            row = []
            for n in names:
                lp = torch.log_softmax(batch.logits[n][b, idx[n]], -1).cpu().numpy()
                row.append(step_logprobs(lp, spine, n, allow_answer=False)[:V])
            z = fuse(row, [weights[n] for n in names], rule)[:V]
            pos.append(z - np.logaddexp.reduce(z))
        per_batch.append(np.stack(pos))
    lens = {p.shape[0] for p in per_batch}
    if len(lens) == 1:
        return np.stack(per_batch)              # (B, P, V) — the aligned case
    if ragged:
        return per_batch                        # list of (P_b, V)
    raise ValueError(
        f"rows have different numbers of aligned positions {sorted(lens)}; "
        f"pass ragged=True for a list of (P_b, V) arrays")


def _char_offsets(tok, ids: torch.Tensor) -> list[int]:
    """Char offset (in the surface text) at which each token ends.

    Used for the *student* grid: the student's fast tokenizer gives offsets
    directly, the teachers' do not, so both sides are expressed as character
    offsets into the same string and intersected.
    """
    specials = _special_ids(tok)
    out, pos = [], 0
    for i in ids.tolist():
        i = int(i)
        if i in specials:
            out.append(pos)
            continue
        s = tok.convert_ids_to_tokens(i)
        b = token_to_bytes(s) if isinstance(s, str) else None
        if b is None:
            out.append(pos)
            continue
        pos += len(b.decode("utf-8", errors="replace"))
        out.append(pos)
    return out


def shared_offsets(batch: "TeacherBatch", weights: dict[str, float]) -> list[list[int]]:
    """For each row: the surface offsets where *every* teacher has a token boundary.

    `fused_target` returns positions in this same order, so `fused[b][j]` is the
    fused law just before offset `shared_offsets(...)[b][j]`.
    """
    names = [n for n in batch.names if n in weights]
    if not names:
        raise ValueError("no teacher in weights")
    out = []
    for b in range(batch.ids[names[0]].shape[0]):
        common = set(_char_offsets(_TOK_OF[names[0]], batch.ids[names[0]][b]))
        for n in names[1:]:
            common &= set(_char_offsets(_TOK_OF[n], batch.ids[n][b]))
        out.append(sorted(common))
    return out


_TOK_OF: dict = {}


def _stop_ids(tok) -> list[int]:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    vocab = tok.get_vocab()
    for t in ("<|im_end|>", "<|ifm|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"):
        if t in vocab:
            ids.add(int(vocab[t]))
    return sorted(ids)


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    """The alignment primitives, with fake tokenizers (no weights, no downloads).

    Real-tokenizer alignment is covered by tests/test_teacher.py; here we pin the
    arithmetic that everything else depends on.
    """
    import torch

    class FakeTok:
        """Each id is one surface piece; optionally multi-byte UTF-8."""
        def __init__(self, pieces):
            self.pieces = pieces
        def convert_ids_to_tokens(self, i):
            return self.pieces[int(i)]

    # byte_positions: cumulative byte length, monotone, ends where the text ends
    tok = FakeTok(["ab", "cd", "e"])
    out = byte_positions(tok, torch.tensor([0, 1, 2]))
    assert out == [2, 4, 5], out
    assert all(b > a for a, b in zip(out, out[1:])), "must be strictly increasing"

    # a special token has no surface bytes -> its end offset equals the previous one
    class FakeTokSpec(FakeTok):
        all_special_ids = [1]
    tok2 = FakeTokSpec(["ab", "<eos>", "c"])
    out2 = byte_positions(tok2, torch.tensor([0, 1, 2]))
    assert out2 == [2, 2, 3], out2
    # ...and WITHOUT the tokenizer telling us, "<eos>" is a perfectly valid 5-byte
    # string: this is why _special_ids must consult the tokenizer, not guess.
    assert byte_positions(FakeTok(["ab", "<eos>", "c"]), torch.tensor([0, 1, 2])) == [2, 7, 8]

    # byte-table semantics, NOT UTF-8: the table maps one character to one BYTE
    # (GPT-2's bytes_to_unicode is a 256-entry permutation), so "é" is 1 byte here
    # while the *decoded* UTF-8 form is 2.  Multi-byte text arrives as several
    # tokens, which is exactly what makes the offset map meaningful.
    tok3 = FakeTok(["é", "x"])
    assert byte_positions(tok3, torch.tensor([0, 1])) == [1, 2]
    tok3b = FakeTok(["Ã", "©", "x"])          # UTF-8 bytes of é, as two tokens
    assert byte_positions(tok3b, torch.tensor([0, 1, 2])) == [1, 2, 3]

    # shared_boundaries is an intersection, not a zip
    assert shared_boundaries([2, 4, 6], [4, 6]) == [(1, 0), (2, 1)]
    assert shared_boundaries([2, 4], [3, 5]) == []
    # and it is symmetric in the pairing direction
    a = shared_boundaries([1, 2, 3], [1, 3])
    b = shared_boundaries([1, 3], [1, 2, 3])
    assert [(x, y) for x, y in a] == [(0, 0), (2, 1)]
    assert [(y, x) for x, y in b] == [(0, 0), (2, 1)]

    # _char_offsets uses CHARACTER length (not bytes) for the student grid
    tok4 = FakeTok(["é", "x"])          # 2 bytes but 1 character
    assert _char_offsets(tok4, torch.tensor([0, 1])) == [1, 2]

    # _stop_ids collects eos plus anything that looks like a stop token
    class T:
        eos_token_id = 5
        def get_vocab(self):
            return {"<eos>": 5, "<|im_end|>": 6, "plain": 7}
    sids = _stop_ids(T())
    assert 5 in sids and 6 in sids and 7 not in sids, sids

    print("teacher selftest OK · byte offsets monotone · boundary intersection · char vs byte · stop ids")


if __name__ == "__main__":
    _selftest()
