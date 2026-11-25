"""shared alphabet across several tokenizers, for exact logit fusion.

Four different BPE vocabularies (K2 64 256, Limite 151 680, Qwen3.5 248 320,
R1 151 936) share **42 184 byte-identical tokens** — enough to cover 90-100 % of
real Chinese/English/math text token-for-token.  On that intersection the four
models predict literally the same next symbol, so a fused step is exact:

    p_fused(v) ∝ prod_m p_m(v) ** w_m      (product / log-linear)
             or  sum_m w_m p_m(v)          (mixture)

Each model's distribution is first restricted to the shared alphabet V and
re-normalised.  That multiplies every model by a *constant per step*, so for the
product rule it cancels exactly (a convex combination of log-probs, gamma=1),
and for the mixture rule it is the only possible reading — neither introduces
an approximation.  Three pseudo-tokens extend the alphabet:

    END            every model's stop set, pooled
    pragma:think   "</think>" as the tokenizer spells it in that model's stream
                   (a *special* token for R1, plain text for Limite)
    pragma:answer  the model asks to leave the thinking block (budget forcing)

Weights are normalised over the *active* models only, so a failed load reduces
the consensus instead of breaking it (frame1: the skeleton keeps moving when a
limb breaks).
"""

from __future__ import annotations

import numpy as np

END, THINK, ANSWER = 0, 1, 2
N_PRAGMA = 3

_THINK_TEXT = "</think>"
_ANSWER_PREFIX = "\n\n"


class SharedSpine:
    """The intersection alphabet + per-model atomic id maps and pragma feeds."""

    def __init__(self, names: list[str], strings: list[bytes], ids: dict[str, np.ndarray], feeds: dict[str, dict[str, list[int]]]):
        self.names = names
        self.strings = strings
        self.ids = ids  # name -> int32[|V|] model token id
        self.feeds = feeds  # name -> {"think": [ids], "answer": [ids]}
        self.index = {s: i for i, s in enumerate(strings)}

    def __len__(self) -> int:
        return len(self.strings)

    def token_ids(self, name: str) -> np.ndarray:
        return self.ids[name]


def _token_id(tok, piece: str) -> list[int]:
    ids = tok(piece, add_special_tokens=False)["input_ids"]
    return [int(i) for i in ids]


def build_spine(tok_of: dict) -> SharedSpine:
    """tok_of: name -> (tokenizer, stop_ids).  Intersection over surface bytes."""
    from bytes_vocab import vocab_bytes

    names = list(tok_of)
    surfaces = {n: vocab_bytes(tok_of[n][0]) for n in names}
    inter = set(surfaces[names[0]])
    for n in names[1:]:
        inter &= set(surfaces[n])
    strings = sorted(inter)
    ids = {n: np.array([surfaces[n][s] for s in strings], dtype=np.int64) for n in names}
    feeds = {}
    for n in names:
        tok, stops = tok_of[n]
        feeds[n] = {
            "think": _token_id(tok, _THINK_TEXT),
            "end": list(stops),
            "answer": _token_id(tok, _ANSWER_PREFIX),
        }
    return SharedSpine(names, strings, ids, feeds)


# --------------------------------------------------------------------------------------
# pure-numpy core (unit-tested); the Space calls the torch twin below
# --------------------------------------------------------------------------------------
def step_logprobs(lp_model: np.ndarray, spine: SharedSpine, name: str, allow_answer: bool = True) -> np.ndarray:
    """map a model's raw log-softmax (length = model vocab) to the shared alphabet.

    Returns log-probs over |V| + 3 pseudo-tokens.
    """
    out = np.empty(len(spine) + N_PRAGMA, dtype=np.float64)
    out[: len(spine)] = lp_model[spine.ids[name]]
    out[len(spine)] = _logsumexp(lp_model[spine.feeds[name]["end"]])
    out[len(spine) + 1] = np.sum(lp_model[spine.feeds[name]["think"]])
    out[len(spine) + 2] = np.sum(lp_model[spine.feeds[name]["answer"]]) if allow_answer else -np.inf
    # renormalise over (shared alphabet + pragma).  For the product rule this is a
    # per-step constant that cancels in the fused softmax; the mixture needs it.
    return out - _logsumexp(out)


def _logsumexp(x: np.ndarray) -> float:
    if np.size(x) == 0:
        return -np.inf  # a model with no stop token can never emit END
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def _safe_exp(z):
    with np.errstate(over="ignore"):
        return np.exp(np.minimum(z, 0.0))


def fuse(logps: list[np.ndarray], weights: list[float], rule: str) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    z = np.stack(logps)
    if rule.startswith("product"):
        out = (w[:, None] * z).sum(0)
    else:
        out = np.log(np.maximum((w[:, None] * _safe_exp(z)).sum(0), 1e-300))
    return out - _logsumexp(out)


def decode(spine: SharedSpine, ids: list[int], primary: str, tokenizer) -> str:
    """decode a walk over the shared alphabet with one model's tokenizer."""
    buf, text = [], []
    for i in ids:
        if i < len(spine):
            buf.append(int(spine.ids[primary][i]))
        else:
            text.append(tokenizer.decode(buf, skip_special_tokens=False))
            buf = []
            text.append({len(spine): "</think>", len(spine) + 1: "", len(spine) + 2: "\n\n"}[i])
    text.append(tokenizer.decode(buf, skip_special_tokens=False))
    return "".join(text)


def selftest() -> None:
    """3 fake models with nested alphabets, hand-checkable posteriors."""

    class _Backend:  # mimics tokenizers.Tokenizer.get_vocab()
        def __init__(self, surf):
            self.vocab = {s.decode("latin-1"): i for s, i in surf.items()}

        def get_vocab(self):
            return dict(self.vocab)

    class FakeTok:
        """unknown pieces map to an out-of-range sentinel (prob ~ 0)"""

        def __init__(self, surf):
            self.surf = surf
            self.sentinel = max(surf.values()) + 1
            self.backend_tokenizer = _Backend(surf)

        def __call__(self, piece, add_special_tokens=False):
            return {"input_ids": [self.surf.get(piece.encode(), self.sentinel)]}

        def decode(self, ids, skip_special_tokens=False):
            inv = {v: k for k, v in self.surf.items()}
            return b"".join(inv.get(int(i), b"?") for i in ids).decode("utf-8", "replace")

    A = FakeTok({b"a": 0, b"b": 1, b"ab": 2, b"c": 3})  # 4 tokens + sentinel 4
    B = FakeTok({b"a": 0, b"ab": 1, b"b": 2})  # 3 tokens + sentinel 3
    C = FakeTok({b"a": 0, b"b": 1, b"c": 2})  # 3 tokens + sentinel 3
    spine = build_spine({"A": (A, []), "B": (B, []), "C": (C, [])})
    assert spine.strings == [b"a", b"b"], spine.strings  # 3-way intersection
    assert list(spine.ids["A"]) == [0, 1] and list(spine.ids["B"]) == [0, 2]

    # A: (a,b,ab,c | sentinel) ; B: (a,ab,b | sentinel) — both restrict to (1/3, 2/3)
    lpA = np.log([0.1, 0.2, 0.5, 0.2, 1e-30])
    lpB = np.log([0.1, 0.7, 0.2, 1e-30])
    zA, zB = step_logprobs(lpA, spine, "A"), step_logprobs(lpB, spine, "B")
    assert np.allclose(np.exp(zA[:2]), [1 / 3, 2 / 3]), np.exp(zA[:2])
    assert np.allclose(np.exp(zB[:2]), [1 / 3, 2 / 3]), np.exp(zB[:2])
    # identical models ⇒ fusion returns that distribution, both rules
    for rule in ("product", "mixture"):
        f = fuse([zA, zB], [0.5, 0.5], rule)
        assert np.allclose(np.exp(f[:2]), [1 / 3, 2 / 3]), (rule, np.exp(f[:2]))
    # no stop token ⇒ END is impossible
    assert np.isneginf(zA[2]) and np.isneginf(zB[2])

    # with a stop id (3 = "c", outside the spine): END takes c's mass, renormalised
    spine_s = build_spine({"A": (A, [3]), "B": (B, [])})  # |V| = 3: a, ab, b
    zAs = step_logprobs(lpA, spine_s, "A")
    assert np.allclose(np.exp(zAs[:4]), [0.1, 0.5, 0.2, 0.2]), np.exp(zAs[:4])
    assert np.isneginf(step_logprobs(lpB, spine_s, "B")[3])

    # disagreement: the fused law is the (weighted) geometric mean of the two
    # restricted+renormalised inputs — property test, not hand arithmetic.
    #   A vocab = (a, b, ab, c, _)        B vocab = (a, ab, b, _)
    pa = np.array([0.5, 0.5, 1e-30, 1e-30, 1e-30])
    pb = np.array([0.3, 1e-30, 0.7, 1e-30])
    zA2, zB2 = step_logprobs(np.log(pa), spine, "A"), step_logprobs(np.log(pb), spine, "B")
    ra, rb = pa[[0, 1]], pb[[0, 2]]
    ra, rb = ra / ra.sum(), rb / rb.sum()
    fp = fuse([zA2, zB2], [0.5, 0.5], "product")
    zs = np.sqrt(ra * rb)
    assert np.allclose(np.exp(fp[:2]), zs / zs.sum()), (np.exp(fp[:2]), zs / zs.sum())
    fm = fuse([zA2, zB2], [0.5, 0.5], "mixture")
    assert np.allclose(np.exp(fm[:2]), 0.5 * ra + 0.5 * rb), (np.exp(fm[:2]), 0.5 * ra + 0.5 * rb)
    # weights are renormalised: [2,1] == [2/3,1/3]
    assert np.allclose(fuse([zA2, zB2], [2, 1], "product"), fuse([zA2, zB2], [2 / 3, 1 / 3], "product"))
    # decode round-trip on the primary tokenizer
    assert decode(spine, [0, 1], "A", A) == "ab"
    print("shared spine selftest OK · |V| =", len(spine), "· strings:", spine.strings)


if __name__ == "__main__":
    selftest()
