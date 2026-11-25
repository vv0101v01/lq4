"""sixin1.fuse — n-way exact logit fusion over a shared alphabet.

`FusionEngine` holds 1-4 causal models (any subset of K2 / Limite / R1 /
Qwen3.5) plus the shared spine built from their tokenizers.  `run()` decodes a
single sequence cooperatively: at every step each model scores the *same*
position and the fused law picks the next symbol.

Load-time compatibility patches required by transformers 5.15 are documented
where they are applied; every one is a no-op on older/newer versions that do
not need it.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from spine import ANSWER, N_PRAGMA, SharedSpine, THINK, build_spine, fuse

END = 0  # pseudo-token index inside spine space (= len(spine))


@dataclass
class ModelSpec:
    name: str
    repo: str
    rev: str
    trust: bool = False
    kind: str = "causal"  # "causal" | "vlm"
    kwargs: dict = field(default_factory=dict)  # apply_chat_template kwargs
    prompt_suffix: str = ""


def stop_ids_for(tok) -> list[int]:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    vocab = tok.get_vocab()
    for t in ("<|im_end|>", "<|ifm|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<|im_end|>\n"):
        if t in vocab:
            ids.add(int(vocab[t]))
    return sorted(ids)


def _balanced_after_boxed(text: str) -> bool:
    """True if some \\boxed{...} in `text` has balanced braces."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return False
    depth, j = 0, i + len("\\boxed{")
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return True
            depth -= 1
        j += 1
    return False


class FusionEngine:
    def __init__(self, specs: list[ModelSpec], dtype=torch.bfloat16, device: str | None = None, log=print):
        # Never hard-code cuda: a ZeroGPU restart can land on a CPU-only worker and
        # then every load fails with "Found no NVIDIA driver", which surfaces only as
        # "models failed to load".  Falling back keeps the Diagnostics tab truthful.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            log(f"⚠️ {device} requested but no GPU is visible — using cpu")
            device = "cpu"
        self.specs = {s.name: s for s in specs}
        self.order = [s.name for s in specs]
        if device == "cpu" and dtype in (torch.bfloat16, torch.float16):
            log("⚠️ CPU cannot run bf16/fp16 kernels reliably — using float32")
            dtype = torch.float32
        self.tok = {}
        self.model = {}
        self.stops = {}
        self.spine: SharedSpine | None = None
        self.dtype = dtype
        self.device = device
        self.log = log

        from transformers import AutoModelForCausalLM, AutoTokenizer

        for name in self.order:
            s = self.specs[name]
            try:
                kw = dict(revision=s.rev, trust_remote_code=s.trust)
                if s.kind == "vlm":
                    from transformers import AutoModelForImageTextToText as _cls
                else:
                    _cls = AutoModelForCausalLM
                tok = AutoTokenizer.from_pretrained(s.repo, **kw)
                model = _cls.from_pretrained(s.repo, dtype=dtype, **kw)
                model = model.to(device).eval()
                self.tok[name] = tok
                self.model[name] = model
                self.stops[name] = stop_ids_for(tok)
                log(f"loaded {name}: {type(model).__name__}")
            except Exception as e:  # keep the Space alive; consensus loses one voice
                log(f"FAILED {name}: {type(e).__name__}: {e}")

        if len(self.tok) < 2:
            raise RuntimeError("need at least two models loaded to fuse")

        self.spine = build_spine({n: (self.tok[n], self.stops[n]) for n in self.tok})
        self.names = [n for n in self.order if n in self.tok]

    # ------------------------------------------------------------------ prompt
    def _prompt_ids(self, name: str, prompt: str, system: str | None) -> torch.Tensor:
        s = self.specs[name]
        tok = self.tok[name]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **s.kwargs)
        text += s.prompt_suffix
        return tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=3072)["input_ids"]

    # -------------------------------------------------------------------- step
    @torch.inference_mode()
    def run(self, prompt: str, system: str | None = None, active: dict[str, float] | None = None,
            rule: str = "product (log-linear)", max_new_tokens: int = 512, temperature: float = 0.6,
            top_p: float = 0.95, seed: int = 0, force_answer: bool = True, think_share: float = 0.75,
            stop_after_boxed: bool = True, boxed_grace: int = 8):
        if not prompt or not prompt.strip():
            raise ValueError("empty prompt")
        names = [n for n in self.names if (active is None or active.get(n, 0) > 0)]
        if len(names) < 2:
            raise ValueError("need at least two active models")
        w = np.array([max(active[n], 1e-6) for n in names], dtype=np.float64) if active else np.ones(len(names))
        w = w / w.sum()

        ids = {n: self._prompt_ids(n, prompt, system) for n in names}
        dev = next(self.model[names[0]].parameters()).device
        caches, logit = {}, {}
        for n in names:
            out = self.model[n](input_ids=ids[n].to(dev), use_cache=True)
            caches[n] = out.past_key_values
            logit[n] = out.logits[0, -1].float()

        if not think_share or think_share <= 0:
            # start directly in the answer: feed each model its answer pragma
            for n in names:
                feed = self.spine.feeds[n]["answer"]
                out = self.model[n](input_ids=torch.tensor([feed], device=dev), past_key_values=caches[n], use_cache=True)
                caches[n], logit[n] = out.past_key_values, out.logits[0, -1].float()

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        # ---- stop policy ------------------------------------------------------
        # Measured failure (2026-11-25): after deriving a correct answer the four
        # models do not agree on END, so fusion keeps going, "double-checks", and
        # restarts the problem — burying the right answer.  With a balanced
        # \boxed{...} in hand we therefore stop `boxed_grace` tokens later.
        # This is a decoding policy, not a change to the fused law.
        probe_buf: list[int] = []
        boxed_seen_at = None
        stopped_by_policy = False
        V = len(self.spine)
        pieces: list[int] = []
        steps = agree = lead = 0
        closed = forced = False
        js_acc = 0.0
        t0 = time.perf_counter()

        for step in range(int(max_new_tokens)):
            z = [self._step_logprobs(n, logit[n]) for n in names]
            fused = fuse(z, w.tolist(), rule)
            if force_answer and think_share > 0 and not closed and step >= int(think_share * int(max_new_tokens)):
                choice, forced = V + THINK, True
            else:
                choice = self._sample(fused, temperature, top_p, gen)

            # diagnostics
            steps += 1
            tops = [int(np.argmax(zi)) for zi in z]
            agree += int(len(set(tops)) == 1)
            lead += int(z[0][choice] >= max(zi[choice] for zi in z[1:]) - 1e-12) if len(z) > 1 else 1
            # Jensen-Shannon over the fused support, ignoring -inf entries
            m = np.mean([np.exp(zi) for zi in z], axis=0)
            lpm = np.log(np.maximum(m, 1e-300))
            js = []
            for zi in z:
                ok = np.isfinite(zi)
                js.append(0.5 * float(np.sum(np.exp(zi[ok]) * (zi[ok] - lpm[ok]))))
            js_acc += float(np.mean(js))

            if choice == V + THINK:
                closed = True
            if choice == V + END:
                break
            pieces.append(choice)
            if stop_after_boxed:
                probe_buf.append(choice)
                if len(probe_buf) > 400:
                    del probe_buf[: len(probe_buf) - 400]
                if boxed_seen_at is None and len(probe_buf) % 4 == 0:
                    tail = self.decode(probe_buf[-120:])
                    if "\\boxed{" in tail and _balanced_after_boxed(tail):
                        boxed_seen_at = len(pieces)
                if boxed_seen_at is not None and len(pieces) - boxed_seen_at >= boxed_grace:
                    stopped_by_policy = True
                    break

            # feed the chosen symbol back to every model in its own spelling
            for n in names:
                if choice < V:
                    feed = [int(self.spine.ids[n][choice])]
                elif choice == V + THINK:
                    feed = self.spine.feeds[n]["think"]
                else:  # pragma: answer
                    feed = self.spine.feeds[n]["answer"]
                out = self.model[n](input_ids=torch.tensor([feed], device=dev), past_key_values=caches[n], use_cache=True)
                caches[n], logit[n] = out.past_key_values, out.logits[0, -1].float()

        dt = time.perf_counter() - t0
        text = self.decode(pieces)
        n = max(steps, 1)
        diag = dict(tokens=steps, seconds=dt, tok_per_s=steps / max(dt, 1e-6), agreement=agree / n,
                    lead={names[0]: lead / n}, js_per_step=js_acc / n, forced=forced,
                    stopped_by_boxed=stopped_by_policy,
                    models=", ".join(names), rule=rule, weights=", ".join(f"{n_}={wi:.2f}" for n_, wi in zip(names, w)))
        return dict(text=text, reasoning=text, diag=diag, steps=pieces, names=names)

    def _step_logprobs(self, name: str, logits: torch.Tensor) -> np.ndarray:
        from spine import step_logprobs

        lp = torch.log_softmax(logits, -1).cpu().numpy()
        return step_logprobs(lp, self.spine, name, allow_answer=False)

    def _sample(self, fused: np.ndarray, temperature: float, top_p: float, gen) -> int:
        """sample on CPU: |V|+3 ~ 42k entries per step, and a CPU generator is
        valid whatever device the models sit on (CUDA generators are not)."""
        if temperature <= 0:
            return int(np.argmax(fused))
        probs = torch.softmax(torch.from_numpy(fused.astype(np.float64) / temperature), -1)
        srt, idx = torch.sort(probs, descending=True)
        srt = srt * ((torch.cumsum(srt, 0) - srt) < top_p)
        srt = srt / srt.sum()
        j = int(torch.multinomial(srt, 1, generator=gen))
        return int(idx[j])

    def decode(self, pieces: list[int]) -> str:
        """decode the fused walk with the *load-order primary* tokenizer."""
        from spine import decode as _decode

        primary = self.names[0]
        return _decode(self.spine, pieces, primary, self.tok[primary])


@dataclass
class BridgeStats:
    rows: list[dict]

    def table(self):
        import pandas as pd

        return pd.DataFrame(self.rows)


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    """CPU-only checks of the pieces that do not need model weights."""
    # the boxed-stop policy
    cases = [
        (r"answer is \boxed{5}", True),
        (r"\boxed{5", False),
        (r"\boxed{\frac{1}{2}}", True),
        (r"no box at all", False),
        (r"\boxed{a}\boxed{", False),      # last box unbalanced -> False
        (r"\boxed{a} then \boxed{b}", True),
    ]
    for text, want in cases:
        got = _balanced_after_boxed(text)
        assert got == want, (text, got, want)

    # ModelSpec defaults: trust must be opt-in, kind must default to causal
    s = ModelSpec("n", "repo", "rev")
    assert s.trust is False and s.kind == "causal" and s.prompt_suffix == ""

    # stop_ids_for must never raise on a well-formed tokenizer-like object
    class FakeTok:
        eos_token_id = 7
        def get_vocab(self):
            return {"<eos>": 7, "<|end|>": 9, "plain": 11}

    ids = stop_ids_for(FakeTok())
    assert 7 in ids, ids

    print("fuse selftest OK · boxed-stop policy · ModelSpec defaults · stop ids")


if __name__ == "__main__":
    _selftest()
