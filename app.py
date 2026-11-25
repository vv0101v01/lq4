"""6in1 — frame1 as the skeleton of five small models.

    spine   frame1 (MoonBit): state space, coupling graph, consensus, certificates
    limbs   K2-Horizon-0.9B · Limite-1B-Violetto · Qwen3.5-0.8B · DeepSeek-R1-Distill-Qwen-1.5B
    sense   LFM2.5-Embedding-350M (1024-d, routes / judges / wires the graph)

Two synthesis levels are live in this Space:

L1  system-level: one entry point, six organs (tabs Router · Fusion · Graph · Diagnostics)
L2  exact logit fusion: the four generators share 42 207 byte-identical tokens; on that
    alphabet every step is decided by the *fused* law of all active models
    (product / mixture), so the four decode one single sequence together.
    (Every number in this file is recomputed live — see `spine_table()` / `AUDIT_*`
    and `audit_vocab.py`; nothing here is asserted from memory.)

Load-time compatibility patches for transformers 5.15 are applied at import
(see `_patch_lfm` / Limite config fix); they are no-ops elsewhere.
"""

from __future__ import annotations

import json
import math
import glob
import os
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gradio as gr  # noqa: E402
import spaces  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

import graph  # noqa: E402
from fuse import FusionEngine, ModelSpec, stop_ids_for  # noqa: E402
import probe  # noqa: E402
from spine import build_spine  # noqa: E402

DTYPE = torch.bfloat16

# --------------------------------------------------------------------------------------
# model zoo (revisions verified 2026-11-25)
# --------------------------------------------------------------------------------------
# Loading all four teachers fills a t4-small (14.6 / 14.74 GiB) and leaves no room for
# the student, the LFM embedder, or any activation.  Only the first two are loaded at
# import; `_ensure_teachers(names)` loads more on demand (and frees what it drops).
WARM_NAMES = {"K2-Horizon-0.9B", "Limite-1B-Violetto"}
SPECS = [
    ModelSpec(
        name="K2-Horizon-0.9B",
        repo="IFM/K2-Horizon-0.9B",
        rev="6b4f28e81b8a77c8b44465a93a452eea198d4ceb",
        trust=True,
        kwargs={"reasoning_effort": "low"},
    ),
    ModelSpec(
        name="Limite-1B-Violetto",
        repo="paradigma-inc/limite-1b-violetto",
        rev="190ee8d9c6777b7c5f8e7592cc00caf495a90ca7",
        trust=True,
    ),
    ModelSpec(
        name="Qwen3.5-0.8B",
        repo="Qwen/Qwen3.5-0.8B",
        rev="2fc06364715b967f1860aea9cf38778875588b17",
        kind="vlm",
        kwargs={"enable_thinking": False},
        prompt_suffix="<think>\n\n</think>\n\n",  # leave the empty think block the template opened
    ),
    ModelSpec(
        name="DeepSeek-R1-Distill-Qwen-1.5B",
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        rev="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    ),
]
SHORT = {
    "K2-Horizon-0.9B": "K2",
    "Limite-1B-Violetto": "Limite",
    "Qwen3.5-0.8B": "Qwen3.5",
    "DeepSeek-R1-Distill-Qwen-1.5B": "R1",
}
LFM_REPO, LFM_REV = "LiquidAI/LFM2.5-Embedding-350M", "f35ae2c91d687658dbf1f2b449382f0b019b9808"

LOAD_LOG: list[str] = []


# --------------------------------------------------------------------------------------
# compatibility patches (transformers 5.15)
# --------------------------------------------------------------------------------------
def _patch_limite():
    """Limite's modeling code reads `config.use_cache`, which transformers 5.15
    no longer defines on PretrainedConfig.  Re-attach it as a plain attribute."""
    from transformers import PretrainedConfig

    if not hasattr(PretrainedConfig, "use_cache"):
        PretrainedConfig.use_cache = True


def _patch_lfm():
    """LFM2.5-Embedding ships remote code written for transformers 4.56.  On 5.15:

    * `Lfm2ShortConv.forward` passes `seq_idx=` which the remote non-causal conv
      does not accept, and `slow_forward` no longer exists on the class;
    * the remote module *re-installs its own patch* when first imported, which
      overwrites anything installed earlier.

    Hence this is idempotent and must be called *after* the encoder is built
    (constructing the model re-runs the remote `_install_patches()`), and once
    at import for safety.
    """
    import sys

    import transformers.models.lfm2.modeling_lfm2 as lfm2

    def wrap(fn):
        if getattr(fn, "_sixin1_tolerant", False):
            return fn

        def tolerant(self, hidden_states, past_key_values=None, cache_position=None, attention_mask=None, **kw):
            return fn(self, hidden_states, past_key_values=past_key_values, cache_position=cache_position, attention_mask=attention_mask)

        tolerant._sixin1_tolerant = True
        return tolerant

    cls = lfm2.Lfm2ShortConv
    # `_shortconv_forward` (installed by the remote module) calls
    # `self.slow_forward(...)`, so wrapping slow_forward is enough — and it must
    # be re-wrapped after any later _install_patches() call.
    real = getattr(cls, "slow_forward", None) or cls.forward
    cls.slow_forward = wrap(real)
    if getattr(cls, "forward", None) is not None and not getattr(cls.forward, "_sixin1_tolerant", False):
        cls.forward = lambda self, *a, **k: self.slow_forward(*a, **k)
    # vars(): never trigger transformers' lazy module __getattr__ (no torchvision here)
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_lfm2_bidirectional") and mod is not None:
            inner = vars(mod).get("_noncausal_shortconv_forward")
            if inner is not None:
                setattr(mod, "_noncausal_shortconv_forward", wrap(inner))


_patch_limite()
_patch_lfm()


# --------------------------------------------------------------------------------------
# load the six organs (cuda at import: ZeroGPU emulates CUDA outside @spaces.GPU)
# --------------------------------------------------------------------------------------
def _load_engine():
    t0 = time.perf_counter()
    try:
        warm = [sp for sp in SPECS if sp.name in WARM_NAMES]
        eng = FusionEngine(warm, dtype=DTYPE, log=LOAD_LOG.append)   # device inferred
        if len(warm) < len(SPECS):
            LOAD_LOG.append(f"ℹ️ loaded {len(warm)}/{len(SPECS)} generators to leave VRAM for "
                            f"the student; the fusion tab loads the rest on demand")
        LOAD_LOG.append(f"✅ spine: |V| = {len(eng.spine):,} shared tokens over {len(eng.names)} generators ({time.perf_counter() - t0:.0f}s)")
        return eng
    except Exception as e:
        LOAD_LOG.append(f"❌ fusion engine: {type(e).__name__}: {e}")
        if "NVIDIA driver" in str(e) or not torch.cuda.is_available():
            LOAD_LOG.append("⚠️ this worker has **no GPU**: ZeroGPU gives a T4 only while the "
                            "Space holds quota. Reload the page / retry later; the fusion "
                            "tabs need it, the Diagnostics numbers do not.")
        traceback.print_exc()
        return None


def _teacher_stack(keep: list[str]):
    """A TeacherStack over the teachers that survived `_select_teachers`.

    `TeacherStack.model` is a plain dict copy, so evicting from `ENGINE` does not
    shrink an already-built stack — the stack has to be built *after* eviction.
    """
    from teacher import TeacherStack
    import teacher as _t

    ts = TeacherStack.__new__(TeacherStack)
    ts.specs, ts.order = ENGINE.specs, list(keep)
    ts.tok = {n: ENGINE.tok[n] for n in keep}
    ts.model = {n: ENGINE.model[n] for n in keep}
    ts.stops = {n: ENGINE.stops[n] for n in keep}
    ts.names, ts.spine = list(keep), ENGINE.spine
    ts.dtype, ts.device = ENGINE.dtype, ENGINE.device
    _t._TOK_OF.update(ENGINE.tok)
    return ts


def _ensure_teachers(want: list[str] | None = None, log=print) -> list[str]:
    """Make sure the requested teachers are resident, evicting any that are not.

    t4-small has 14.74 GiB.  Four teachers are 14.6 GiB and the student + activations
    need ~3 GB more, so the resident set has to be chosen per request instead of
    loading everything at import and hoping.
    """
    import gc

    global ENGINE
    if ENGINE is None:
        return []
    want = list(want or ENGINE.order)
    have = [n for n in ENGINE.order if n in ENGINE.model]
    # free what we do not need
    for n in [h for h in have if h not in want]:
        m = ENGINE.model.pop(n, None)
        ENGINE.tok.pop(n, None); ENGINE.stops.pop(n, None)
        if m is not None:
            try:
                m.to("meta")
            except Exception:
                pass
            del m
    gc.collect(); torch.cuda.empty_cache()
    # load what is missing
    tok_of = {n: (ENGINE.tok[n], ENGINE.stops[n]) for n in ENGINE.model}
    for sp in SPECS:
        if sp.name in want and sp.name not in ENGINE.model:
            try:
                if ENGINE.model and _vram_free_gb() < 2.2:
                    log(f"⏸️ not loading {sp.name}: only {_vram_free_gb():.1f} GB free")
                    continue
                from transformers import AutoModelForCausalLM, AutoTokenizer

                kw = dict(revision=sp.rev, trust_remote_code=sp.trust)
                if sp.kind == "vlm":
                    from transformers import AutoModelForImageTextToText as _cls
                else:
                    _cls = AutoModelForCausalLM
                tok = AutoTokenizer.from_pretrained(sp.repo, **kw)
                model = _cls.from_pretrained(sp.repo, dtype=ENGINE.dtype, **kw).to(ENGINE.device).eval()
                ENGINE.tok[sp.name] = tok
                ENGINE.model[sp.name] = model
                ENGINE.stops[sp.name] = stop_ids_for(tok)
                tok_of[sp.name] = (tok, ENGINE.stops[sp.name])
                log(f"⬆️ loaded {sp.name} on demand ({_vram_free_gb():.1f} GB free)")
            except Exception as e:
                log(f"❌ {sp.name}: {type(e).__name__}: {e}")
    ENGINE.names = [n for n in ENGINE.order if n in ENGINE.model]
    if len(ENGINE.names) >= 2:
        ENGINE.spine = build_spine(tok_of)
    return ENGINE.names


def _load_lfm():
    t0 = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(LFM_REPO, revision=LFM_REV)
        model = AutoModel.from_pretrained(LFM_REPO, revision=LFM_REV, dtype=DTYPE, trust_remote_code=True, attn_implementation="sdpa")
        _patch_lfm()  # construction re-runs the remote _install_patches(); wrap after it
        model = model.to("cuda").eval()
        LOAD_LOG.append(f"✅ LFM2.5-Embedding-350M: {type(model).__name__} (1024-d, bidirectional, non-causal conv) {time.perf_counter() - t0:.0f}s")
        return tok, model
    except Exception as e:
        LOAD_LOG.append(f"❌ LFM2.5-Embedding-350M: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None


SKIP_LOAD = os.environ.get("SIXIN1_SKIP_LOAD") == "1"  # lets CI build the UI without a GPU
def _should_load() -> bool:
    """Loading 4 teachers needs ~10 GB of weights — fine on a ZeroGPU T4 (16 GB VRAM),
    fatal on `cpu-basic` (16 GB *RAM total*, where the runtime alone uses several GB).
    A ZeroGPU restart that lands on cpu-basic must therefore skip loading and let the
    UI explain, instead of dying with 'Memory limit exceeded (16.0G)'.
    """
    if SKIP_LOAD:
        return False
    if not torch.cuda.is_available():
        LOAD_LOG.append("⏸️ no GPU on this worker — models not loaded. "
                        "Fusion/distillation need the ZeroGPU runtime (T4).")
        return False
    free = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0.0
    if free < 9.0:
        LOAD_LOG.append(f"⏸️ only {free:.1f} GB of VRAM free — need ~10 GB for the teachers; not loading.")
        return False
    return True


ENGINE: FusionEngine | None = _load_engine() if _should_load() else None
for _l in LOAD_LOG:  # mirror to stdout: the HF log view is the only forensic trail
    print("[load]", _l, flush=True)
LFM_TOK, LFM = (None, None) if SKIP_LOAD else _load_lfm()


@torch.inference_mode()
def lfm_embed(texts: list[str]) -> np.ndarray:
    """CLS-pooled, L2-normalised LFM vectors; one text at a time (no padding)."""
    out = []
    for t in texts:
        ids = LFM_TOK(t, truncation=True, max_length=512, return_tensors="pt")["input_ids"].to("cuda")
        h = LFM(input_ids=ids, use_cache=False).last_hidden_state[0, 0].float()
        out.append(F.normalize(h, dim=0).cpu().numpy())
    return np.stack(out).astype(np.float64)


# ---- frame1 agent population (128 role cards + real LFM vectors, int8) ----------------
AGENTS = json.load(open(os.path.join(HERE, "data", "agents.json"), encoding="utf-8"))
FAMILY_NAMES = AGENTS["family_names"]
AGENT_FAM = np.array([a["family"] for a in AGENTS["agents"]])
AGENT_TEXT = [a["text"] for a in AGENTS["agents"]]
AGENT_VEC = np.load(os.path.join(HERE, "data", "agent_vecs_int8.npy")).astype(np.float64)
AGENT_UNIT = AGENT_VEC / np.linalg.norm(AGENT_VEC, axis=1, keepdims=True)
TRACE = pd.read_csv(os.path.join(HERE, "data", "hexad_trace.csv"))
FAMILY_TO_GEN = {
    0: "K2-Horizon-0.9B",
    1: "Limite-1B-Violetto",
    2: "DeepSeek-R1-Distill-Qwen-1.5B",
    3: "Qwen3.5-0.8B",
}
GEN_NAMES = [s.name for s in SPECS]


# --------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------
CLOSE_RE = re.compile(r"</(?:ifm\|)?think(?:_fast|_faster)?>")
OPEN_RE = re.compile(r"<(?:ifm\|)?think(?:_fast|_faster)?>")


def split_reasoning(raw: str) -> tuple[str, str]:
    m = None
    for m in CLOSE_RE.finditer(raw or ""):
        pass
    if m is None:
        return "", (raw or "").strip()
    return OPEN_RE.sub("", raw[: m.start()]).strip(), raw[m.end():].strip()


def last_boxed(text: str):
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text or "")]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i: j - 1].strip() if depth == 0 else None


# --------------------------------------------------------------------------------------
# L1 · router (LFM query vector → 128-agent population → per-model weights)
# --------------------------------------------------------------------------------------
def route(qvec: np.ndarray, top_k: int = 16, temp: float = 0.1):
    s = AGENT_UNIT @ qvec
    top = np.argsort(-s)[:top_k]
    w = np.exp((s[top] - s[top].max()) / temp)
    w /= w.sum()
    fam = np.zeros(5)
    for i, wi in zip(top, w):
        fam[AGENT_FAM[i]] += wi
    gen = np.array([fam[f] for f in range(4)])
    gen = gen / gen.sum() if gen.sum() > 0 else np.full(4, 0.25)
    agents = pd.DataFrame(
        {
            "agent": [int(i) for i in top[:8]],
            "family": [FAMILY_NAMES[AGENT_FAM[i]] for i in top[:8]],
            "cosine": [round(float(s[i]), 4) for i in top[:8]],
            "role card": [AGENT_TEXT[i].removeprefix("document: ")[:110] for i in top[:8]],
        }
    )
    return {FAMILY_TO_GEN[f]: float(gen[f]) for f in range(4)}, agents


def _router_duration(prompt, mode, *a, **k):
    n = 4 if mode == "All four" else 2
    return int(min(180, 15 + n * 40))


@spaces.GPU(duration=_router_duration)
def hexad_router(prompt: str, mode: str = "Auto (LFM router)", max_new_tokens: int = 512, temperature: float = 0.6, seed: int = 0):
    """L1: route with LFM → active models answer as one fused stream."""
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    if ENGINE is None or LFM is None:
        raise gr.Error("Models failed to load — see Diagnostics.")
    t0 = time.perf_counter()
    weights, agents = route(lfm_embed(["query: " + prompt])[0])
    if mode == "All four":
        active = {n: 1.0 for n in GEN_NAMES}
        chosen = [n for n in GEN_NAMES if n in ENGINE.names]
    else:
        ranked = sorted(weights, key=weights.get, reverse=True)
        chosen = [n for n in ranked if n in ENGINE.names][:2]
        active = {n: max(weights[n], 1e-3) for n in chosen}
    if len(chosen) < 2:
        raise gr.Error("Fewer than two generators loaded.")
    status = f"Router → **{', '.join(SHORT[n] for n in chosen)}** (LFM top-16 agent vote) · fusing…"
    wdf = pd.DataFrame({"model": [SHORT[n] for n in GEN_NAMES], "weight": [round(weights[n], 4) for n in GEN_NAMES]})
    yield status, wdf, agents, "", pd.DataFrame()

    r = ENGINE.run(prompt, active=active, rule="product (log-linear)", max_new_tokens=int(max_new_tokens),
                   temperature=float(temperature), seed=int(seed), think_share=0.75)
    reasoning, answer = split_reasoning(r["text"])
    box = last_boxed(answer or reasoning)
    d = r["diag"]
    header = (
        f"### 🧬 6in1 — fused answer over {len(r['names'])} generators\n"
        f"_{d['models']} · {d['rule']} · {d['tokens']} tok · {d['seconds']:.1f}s · "
        f"top-1 agreement {100 * d['agreement']:.0f}% · mean JS {d['js_per_step']:.3f} nats"
        f"{' · budget-forced answer' if d['forced'] else ''} · total {time.perf_counter() - t0:.1f}s_"
    )
    if box:
        header += f"\n\n**boxed:** `{box}`"
    out = header + "\n\n" + (answer or "_(still thinking when the budget ran out)_")
    dist = pd.DataFrame({"model": [SHORT[n] for n in GEN_NAMES], "router weight": [round(weights[n], 4) for n in GEN_NAMES]})
    yield out, wdf, agents, reasoning if len(reasoning) < 20000 else reasoning[:10000] + "\n…\n" + reasoning[-10000:], dist


# --------------------------------------------------------------------------------------
# L2 · n-way exact fusion (explicit active set + weights + rule)
# --------------------------------------------------------------------------------------
def _fusion_duration(active, *a, max_new_tokens=384, **k):
    n = max(1, len(active or []))
    return int(min(180, 15 + n * 45))


@spaces.GPU(duration=_fusion_duration)
def fuse_models(prompt, active, k2_w, limite_w, qwen_w, r1_w, rule, max_new_tokens, temperature, top_p, seed, think_share):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    if ENGINE is None:
        raise gr.Error("Models failed to load — see Diagnostics.")
    if not active or len(active) < 2:
        raise gr.Error("Select at least two generators.")
    wmap = {"K2-Horizon-0.9B": k2_w, "Limite-1B-Violetto": limite_w, "Qwen3.5-0.8B": qwen_w, "DeepSeek-R1-Distill-Qwen-1.5B": r1_w}
    # only two teachers are resident at import (see WARM_NAMES); bring in whatever the
    # user actually selected, and drop the rest so the VRAM budget still fits
    resident = _ensure_teachers(list(active), log=LOAD_LOG.append)
    if len(resident) < 2:
        raise gr.Error("Could not load the selected generators — see Diagnostics.")
    active = [n for n in active if n in ENGINE.names]
    if len(active) < 2:
        raise gr.Error("Selected generators did not load — see Diagnostics.")
    r = ENGINE.run(prompt, active={n: float(wmap[n]) for n in active}, rule=rule,
                   max_new_tokens=int(max_new_tokens), temperature=float(temperature), top_p=float(top_p),
                   seed=int(seed), think_share=float(think_share))
    reasoning, answer = split_reasoning(r["text"])
    box = last_boxed(answer or reasoning)
    d = r["diag"]
    wparts = [wp.split("=")[1] for wp in d["weights"].split(", ")]
    mix = " + ".join(f"{SHORT[n]}×{v}" for n, v in zip(r["names"], wparts))
    head = (f"**{d['rule']}** · {mix}\n\n"
            f"_{d['tokens']} tok · {d['seconds']:.1f}s · {d['tok_per_s']:.1f} tok/s · top-1 agreement {100 * d['agreement']:.0f}%"
            f" · mean JS {d['js_per_step']:.3f} nats{' · budget-forced' if d['forced'] else ''}_")
    md = head + ("\n\n**boxed:** `" + box + "`" if box else "") + "\n\n" + (answer or "_(still thinking when the budget ran out)_")
    sdf = pd.DataFrame({"metric": list(d), "value": [str(v) for v in d.values()]})
    return md, reasoning, sdf


def pairwise_md() -> str:
    """Live k-way intersection sizes — computed from the loaded tokenizers, never hard-coded."""
    if ENGINE is None:
        return "_models not loaded yet — see Diagnostics above_"
    import itertools

    from bytes_vocab import vocab_bytes

    surf = {n: set(vocab_bytes(ENGINE.tok[n])) for n in ENGINE.names}
    pairs = []
    for a, b in itertools.combinations(ENGINE.names, 2):
        pairs.append(f"{SHORT[a]}∩{SHORT[b]} **{len(surf[a] & surf[b]):,}**")
    four = len(set.intersection(*surf.values())) if len(surf) == 4 else None
    txt = ("逐步融合只在此交集 + 3 個 pseudo-token（END / `</think>` / answer）上進行。\n\n"
           + "每對模型之間嘅交集：" + " · ".join(pairs))
    if four is not None:
        txt += f"\n\n四者交集（spine）**{four:,}** —— 與 Diagnostics 上方 `build_spine()` 嘅 |V| 一致。"
    return txt


def spine_table() -> pd.DataFrame:
    if ENGINE is None:
        return pd.DataFrame()
    names = ENGINE.names
    rows = []
    for n in names:
        mv = 0
        try:
            mv = max(int(i) for i in ENGINE.tok[n].get_vocab().values()) + 1
        except Exception:
            pass
        rows.append(
            {
                "model": SHORT[n],
                "vocab": mv,
                "in shared spine": int(len(ENGINE.spine)),
                "surface tokens": len(ENGINE.tok[n].get_vocab()),
                "stop ids": ", ".join(str(i) for i in ENGINE.stops[n]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# frame1 graph tab (CPU) + custom population (GPU for LFM)
# --------------------------------------------------------------------------------------
FAM_COLORS = ["#e4572e", "#7b2cbf", "#2a9d8f", "#f4a261", "#3a86ff"]


def _graph_figure(emb, fam, w, names, title):
    xy = graph.pca2(emb)
    fig, ax = _plt().subplots(figsize=(7, 5.6), dpi=110)
    for i, j in np.argwhere(np.triu(w, 1) > 0):
        cross = fam[i] != fam[j]
        ax.plot(*zip(xy[i], xy[j]), color="#999" if not cross else "#111", lw=0.35 if not cross else 0.4, alpha=0.35 if not cross else 0.18, zorder=1)
    for f in sorted(set(fam.tolist())):
        m = fam == f
        ax.scatter(xy[m, 0], xy[m, 1], s=26, color=FAM_COLORS[f % len(FAM_COLORS)], label=names[f], zorder=2, edgecolor="white", lw=0.4)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best", frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _dynamics(w_ring, w_knn, emb, steps=200):
    x0 = graph.pca2(emb)
    rows = []
    for name, w in (("ring", w_ring), ("LFM k-NN", w_knn)):
        d = graph.diffuse(w, x0, steps)
        r = graph.kuramoto(w, steps)
        rows += [dict(step=t, value=float(d[t] / d[0]), series=f"{name} · concept dispersion") for t in range(steps + 1)]
        rows += [dict(step=t, value=float(r[t]), series=f"{name} · Kuramoto R") for t in range(steps + 1)]
    return pd.DataFrame(rows)


def _stats_df(w_ring, w_knn, fam):
    a, b = graph.graph_stats(w_ring, fam), graph.graph_stats(w_knn, fam)
    keys = ["agents", "edges", "cross_family_edges", "components", "lambda2", "lambda_max", "heat_CFL_dt"]
    fmt = lambda v: f"{v:.5g}" if isinstance(v, float) else str(v)  # noqa: E731
    return pd.DataFrame({"metric": keys, "ring": [fmt(a[k]) for k in keys], "LFM k-NN": [fmt(b[k]) for k in keys]}), b["lambda2"] / max(a["lambda2"], 1e-12)


def frame1_graph(k=8, tau=0.25):
    """CPU-only; streams one intermediate frame so a ZeroGPU Space never times out
    inside `demo.load` (which runs outside any GPU lease)."""
    yield gr.update(), gr.update(), gr.update()
    w_knn = graph.knn_graph(AGENT_VEC, AGENT_FAM, int(k), float(tau))
    w_ring = graph.ring_graph(len(AGENT_FAM))
    df, ratio = _stats_df(w_ring, w_knn, AGENT_FAM)
    fig = _graph_figure(AGENT_VEC, AGENT_FAM, w_knn, FAMILY_NAMES, f"128 agents · LFM k-NN (k={int(k)}, τ={tau}) · λ₂ ×{ratio:.0f} vs ring")
    yield df, fig, _dynamics(w_ring, w_knn, AGENT_VEC)


@spaces.GPU(duration=40)
def custom_graph(lines, k=3, tau=0.25):
    if LFM is None:
        raise gr.Error("LFM2.5-Embedding failed to load — see Diagnostics.")
    rows = [l.strip() for l in (lines or "").splitlines() if l.strip()][:96]
    if len(rows) < 4:
        raise gr.Error("Give at least 4 lines.")
    fams, texts = [], []
    for r in rows:
        f, _, t = r.partition("|") if "|" in r else ("agents", "", r)
        fams.append(f.strip())
        texts.append(t.strip())
    names = list(dict.fromkeys(fams))
    fam = np.array([names.index(f) for f in fams])
    emb = lfm_embed(["document: " + t for t in texts])
    w_knn = graph.knn_graph(emb, fam, int(k), float(tau))
    w_ring = graph.ring_graph(len(rows))
    df, ratio = _stats_df(w_ring, w_knn, fam)
    fig = _graph_figure(emb, fam, w_knn, names, f"{len(rows)} agents · LFM k-NN (k={int(k)}, τ={tau}) · λ₂ ×{ratio:.1f} vs ring")
    return df, fig, _dynamics(w_ring, w_knn, emb, steps=120)


@spaces.GPU(duration=30)
def lfm_selfcheck(n=16):
    if LFM is None:
        raise gr.Error("LFM failed to load.")
    idx = np.linspace(0, len(AGENT_TEXT) - 1, int(n)).astype(int)
    v = lfm_embed([AGENT_TEXT[i] for i in idx])
    cos = np.sum(v * AGENT_UNIT[idx], axis=1)
    s_hf, s_cpp = v @ v.T, AGENT_UNIT[idx] @ AGENT_UNIT[idx].T
    iu = np.triu_indices(len(idx), 1)
    corr = float(np.corrcoef(s_hf[iu], s_cpp[iu])[0, 1])
    return (
        f"**LFM self-check** on {len(idx)} frame1 role cards: cos(HF, llama.cpp Q8_0) mean **{cos.mean():.4f}**, min {cos.min():.4f}; "
        f"similarity-matrix Pearson r = **{corr:.4f}**  (≥0.98 ⇒ the encoder patches reproduce the GGUF embedder)"
    )


@spaces.GPU(duration=60)
def measure_sparsity(prompt: str = "Find the remainder when 7^100 is divided by 13.", max_len: int = 256):
    """L3 planning: measure how concentrated the REAL fused target is.

    Decides whether top-64 targets cost 31.5 GB or top-256 costs 126 GB.
    One teacher-forced forward per model — no generation.
    """
    if ENGINE is None:
        raise gr.Error("Models failed to load — see Diagnostics.")
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    st = probe.probe(ENGINE, prompt, max_len=int(max_len))
    rows = []
    for n, r in st["per_model"].items():
        rows.append({"target": SHORT[n], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
    if st["fused"]:
        rows.append({"target": "FUSED (Limite+R1, exact)", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st["fused"].items()}})
    df = pd.DataFrame(rows)
    fused = st.get("fused") or (list(st["per_model"].values())[0])
    lines = [
        f"### measured on {st['spine']:,}-token spine · {fused['positions']} positions",
        "",
        f"- mean entropy **{fused['entropy']:.3f} nats** · mean top-1 prob **{fused['top1']:.3f}**",
    ]
    Z64 = fused.get("Z@64")
    Z256 = fused.get("Z@256")
    if Z64:
        lines.append(f"- **top-64 retains {100 * Z64:.2f}%** of the mass (tail {100 * (1 - Z64):.2f}%) → 31.5 GB for 20k×4k sequences")
    if Z256:
        lines.append(f"- top-256 retains {100 * Z256:.3f}% (tail {100 * (1 - Z256):.3f}%) → 126 GB")
    if any(r.get("near_uniform") for r in st["per_model"].values()):
        lines.append("- ⚠️ a per-model row is uniform (bug guard tripped) — treat that row as invalid")
    p10 = fused.get("p10@64")
    if p10 is not None:
        lines.append(f"- worst decile at k=64: **Z = {100 * p10:.1f}%** (a mean can hide bad positions)")
    chk = st.get("checks") or {}
    if chk:
        lines.append(f"- fused vs teachers: **{'flatter (teachers disagree)' if chk.get('fused_flatter_than_every_teacher') else 'sharper (teachers agree)'}**")
    lines += ["", f"**{probe.verdict(st)}**"]
    md = "\n".join(lines)
    return md, df


# --------------------------------------------------------------------------------------
# L3 baseline: run the fused system on licence-checked MATH problems, score exact-match
# --------------------------------------------------------------------------------------
CORPUS_PATH = os.path.join(HERE, "data", "corpus", "corpus_math.jsonl")
CORPUS_GZ = CORPUS_PATH + ".gz"
TEST_PATH = os.path.join(HERE, "data", "corpus", "corpus_test.jsonl")


_CORPUS_CACHE: dict[str, list] = {}


def corpus_rows(path=None):
    """Read the corpus, skipping any unparsable line instead of failing the request."""
    path = path or CORPUS_PATH
    if path in _CORPUS_CACHE:
        return _CORPUS_CACHE[path]
    gz = path + ".gz"
    if os.path.exists(gz):
        try:
            import gzip

            with gzip.open(gz, "rt", encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
            _CORPUS_CACHE[path] = rows
            LOAD_LOG.append(f"📚 corpus: {len(rows):,} records ({os.path.basename(gz)})")
            return rows
        except Exception as e:  # corrupt gzip (truncated LFS fetch) -> fall through
            LOAD_LOG.append(f"⚠️ gzip corpus unreadable ({type(e).__name__}) — falling back to jsonl")
    if not os.path.exists(path):
        return []
    rows, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    if bad:
        LOAD_LOG.append(f"⚠️ corpus: skipped {bad} unparsable lines")
    LOAD_LOG.append(f"📚 corpus: {len(rows):,} records loaded from {os.path.basename(path)}")
    if not rows:
        LOAD_LOG.append("❌ corpus file present but yielded 0 records — truncated upload?")
    _CORPUS_CACHE[path] = rows
    return rows


def _norm_ans(a):
    if a is None:
        return None
    import unicodedata

    a = unicodedata.normalize("NFKC", a)
    a = a.replace("\\!", "").replace("\\,", "").replace(" ", "").replace("$", "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return a.rstrip(".").strip() or None


def _extract_answer(text: str):
    """Prefer the last \\boxed{...}; fall back to the last number on the last line."""
    box = last_boxed(text or "")
    if box:
        return _norm_ans(box), "boxed"
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        return None, "empty"
    tail = lines[-1]
    nums = re.findall(r"-?\d[\d,\.]*", tail)
    if nums:
        return _norm_ans(nums[-1]), "last_number"
    return _norm_ans(tail[:80]), "last_line"


@spaces.GPU(duration=lambda n, *a: int(min(180, 8 + int(n) * 22)))
def run_baseline(n: int = 6, offset: int = 0, max_new_tokens: int = 320, temperature: float = 0.3,
                 budget: float = 0.85, seed: int = 0, split: str = "train"):
    """Score the live 4-way fusion on MATH problems (exact-match)."""
    rows = corpus_rows(TEST_PATH) if str(split).startswith("test") else corpus_rows()
    if not rows:
        raise gr.Error("Corpus not found — upload data/corpus/corpus_train.jsonl.")
    if ENGINE is None:
        raise gr.Error("Models failed to load — see Diagnostics.")
    lo = max(0, int(offset))
    batch = rows[lo : lo + int(n)]
    if not batch:
        raise gr.Error("offset past the end of the corpus.")

    out_rows = []
    ok = 0
    t0 = time.perf_counter()
    for i, r in enumerate(batch):
        prompt = r["problem"] + "\n\nSolve step by step and put the final answer in \\boxed{}."
        try:
            res = ENGINE.run(prompt, active={n_: 1.0 for n_ in ENGINE.names},
                             rule="product (log-linear)", max_new_tokens=int(max_new_tokens),
                             temperature=float(temperature), top_p=0.95, seed=int(seed) + i,
                             think_share=float(budget))
            got, how = _extract_answer(res["text"])
            pred_reason, _ = split_reasoning(res["text"])
            if got is None:
                got, how = _extract_answer(pred_reason)
        except Exception as e:
            got, how = None, f"error:{type(e).__name__}"
        want = _norm_ans(r.get("answer"))
        hit = bool(got is not None and want is not None and got == want)
        ok += hit
        out_rows.append({
            "i": lo + i, "subject": r["subject"], "want": want, "got": got,
            "match": "✅" if hit else "❌", "how": how,
        })
    dt = time.perf_counter() - t0
    md = (f"### L3 baseline · {len(batch)} problems ({batch[0]['subject']} …)\n\n"
          f"- **exact-match {ok}/{len(batch)} = {100 * ok / len(batch):.0f}%**\n"
          f"- {dt:.0f}s total · {dt / len(batch):.1f}s/problem · max_new={int(max_new_tokens)}\n"
          "- scoring: last `boxed{}` (fallback: last number on last line), answers normalised\n"
          f"- these rows are the denominator for the distilled student's §6 acceptance test")
    return md, pd.DataFrame(out_rows)


# --------------------------------------------------------------------------------------
# L3 distillation: a chunked trainer that survives ZeroGPU preemption
# --------------------------------------------------------------------------------------
# Checkpoints must live on the persistent volume.  `/app` is rebuilt from the repo on
# every restart, so the earlier `HERE/ckpt` silently lost all progress (measured: the
# step counter fell from 185 back to 6 after a restart).  `/data` is the Space's
# bucket volume; fall back to a local dir only when it is not mounted.
_PERSIST = "/data"
CKPT_DIR = os.path.join(_PERSIST if os.path.isdir(_PERSIST) else HERE, "ckpt")

DISTILL_STATE = {"step": 0, "history": [], "positions": 0, "started": None, "last_err": None}


def _distill_rows(limit=None):
    import train_distill

    path = os.path.join(HERE, "data", "corpus", "corpus_math.jsonl")
    return train_distill.DistillRows(path, limit)


@spaces.GPU(duration=280)
def distill_step(steps: int = 4, k: int = 256, lr: float = 1e-4, max_len: int = 768, limit: int = 1200,
                 fresh: bool = False, teachers_n: int = 2, seed: int = 0):
    """One chunk of LoRA distillation.  Safe to call repeatedly: resumes from ckpt."""
    import gc

    import train_distill as TD
    from teacher import TeacherStack

    t0 = time.perf_counter()
    if ENGINE is None:
        raise gr.Error("models failed to load — see Diagnostics")
    rows = _distill_rows(limit if limit and limit > 0 else None)
    if fresh:
        for f in glob.glob(os.path.join(CKPT_DIR, "step_*.pt")):
            os.remove(f)
        DISTILL_STATE.update(step=0, history=[], positions=0, started=time.time())
    if DISTILL_STATE["started"] is None:
        DISTILL_STATE["started"] = time.time()

    # The Space runs on t4-small (14.74 GiB) and the four teachers already occupy
    # 14.60 GiB of it.  A student cannot be created without freeing memory first, so
    # the teachers *not* selected are evicted from the engine (they are re-loadable
    # from the Hub).  Measured: creating the student with all four resident dies with
    # `CUDA out of memory ... 14.74 GiB ... 2.12 MiB free`.
    keep, evicted = _select_teachers(ENGINE, teachers_n)
    if evicted:
        LOAD_LOG.append(f"♻️ evicted {evicted} → {_vram_free_gb():.1f} GB free")
    teachers = _teacher_stack(keep)   # rebuild after eviction (models dict is a copy)
    s_tok = ENGINE.tok[keep[0]]
    student_tok, student = TD.build_student("Qwen/Qwen3.5-0.8B", "2fc0636", dtype=torch.bfloat16,
                                            device=str(ENGINE.device), log=lambda m: LOAD_LOG.append(m))
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)
    ck = TD.latest_ckpt(CKPT_DIR)
    resumed_from = None
    if ck and not fresh:
        st, _ = TD.load_ckpt(ck, student, opt)
        DISTILL_STATE["step"] = max(DISTILL_STATE["step"], st)
        resumed_from = f"{os.path.basename(ck)} @ step {st}"
        LOAD_LOG.append(f"↩️ resumed {resumed_from}")

    LOAD_LOG.append(f"📊 budget before train: {_vram_report()}")
    try:
        hist = TD.train_student(
            teachers, ENGINE.spine, student_tok, student, rows, steps,
            k=k, T=2.0, lr=lr, bs=1, max_len=max_len, out_dir=CKPT_DIR, save_every=1,
            log=lambda m: LOAD_LOG.append(m),
        )
    except BaseException:
        # An exception raised inside the training loop keeps every frame of that loop
        # alive through the traceback, so `del student` there releases nothing and the
        # device stays at 14.73 GB (measured: it did).  Leaving the except block with
        # `raise` clears the frames — the free below then actually returns memory.
        raise
    finally:
        # Unconditional: the next chunk must start from the same VRAM state whether
        # this one succeeded or not (the failure used to cascade 322 -> 12 MiB).
        for _n in ("student", "opt"):
            if _n in locals():
                locals()[_n] = None
        student = opt = None
        gc.collect()
        torch.cuda.empty_cache()
        LOAD_LOG.append(f"🧹 after train: {_vram_report()}")
    DISTILL_STATE["history"] += [(s, float(l), n) for s, l, n in hist]
    DISTILL_STATE["step"] += len(hist)
    DISTILL_STATE["positions"] += sum(n for _, _, n in hist)

    # `history` accumulates for the life of the worker, so reporting its head as
    # "first" printed the same 26.8133 for every chunk ever after — a display bug that
    # looked exactly like "the checkpoint is not loading".  Report THIS chunk only.
    h = DISTILL_STATE["history"][-len(hist):] if (hist := locals().get("hist")) else []
    dt = time.perf_counter() - t0
    md = [f"### L3 distillation · chunk of {steps} steps in {dt:.0f}s",
          f"- loss code: `targets.distill_kl` (support chosen at T=1, T^2 scaling)",
          f"- checkpoint dir: `{CKPT_DIR}`{' (persistent)' if CKPT_DIR.startswith('/data') else ' ⚠️ NOT persistent — will be lost on restart'}",
          f"- resumed: {resumed_from or 'no (fresh start)'}",
          f"- **step {DISTILL_STATE['step']}** · {DISTILL_STATE['positions']:,} supervised positions total",
          f"- k={k} · T=2.0 · LoRA r=16 on Qwen3.5-0.8B · lr={lr:g}",
          f"- teachers: {', '.join(keep)} (t4-small: 14.74 GiB, all four do not fit alongside the student)"]
    if h:
        losses = [l for _, l, _ in h]
        md.append(f"- loss: first {losses[0]:.4f} → last {losses[-1]:.4f} · min {min(losses):.4f} · "
                  f"Δ {losses[0] - losses[-1]:+.4f}")
    if ck:
        md.append(f"- checkpoint: `{os.path.basename(ck)}` (resumes automatically)")
    return "\n".join(md)


def _ckpt_facts() -> dict:
    """What the persistent volume actually holds — the only durable source of truth."""
    import torch as _t

    files = sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")),
                   key=lambda p: int(os.path.basename(p)[5:-3]))
    if not files:
        return {"steps": 0, "files": 0, "dir": CKPT_DIR}
    last = files[-1]
    try:
        sd = _t.load(last, map_location="cpu", weights_only=False)
        meta = sd.get("meta", {})
        return {"steps": int(sd.get("step", 0)), "files": len(files), "dir": CKPT_DIR,
                "k": meta.get("k"), "T": meta.get("T"), "lr": meta.get("lr"), "rows": meta.get("n")}
    except Exception as e:
        return {"steps": int(os.path.basename(last)[5:-3]), "files": len(files), "dir": CKPT_DIR,
                "error": f"{type(e).__name__}: {e}"}


def _select_teachers(engine, want: int):
    """Keep `want` teachers resident and actually free the rest.

    `dict.pop` alone is not enough: the module stays alive through autograd hooks
    and the accelerator's device map, so VRAM is not returned and the next
    `from_pretrained` dies with "14.74 GiB in use ... 2.12 MiB free" (measured
    twice on t4-small).  Moving to the meta device releases the real storage.
    """
    import gc

    names = [n for n in engine.order if n in engine.model]
    want = max(2, min(int(want), len(names)))
    keep = names[:want]
    evicted = [n for n in names if n not in keep]
    for n in evicted:
        m = engine.model.pop(n, None)
        engine.tok.pop(n, None)
        engine.stops.pop(n, None)
        if m is not None:
            try:
                m.to("meta")
            except Exception:
                pass
            del m
    if evicted:
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    engine.names = keep
    return keep, evicted


def _vram_free_gb() -> float:
    return torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0.0


def _vram_report() -> str:
    """`free` alone cannot tell "teachers too big" from "previous call leaked"."""
    if not torch.cuda.is_available():
        return "no gpu"
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    resident = ",".join(ENGINE.names) if ENGINE is not None else "-"
    return (f"free {free / 1e9:.2f}/{total / 1e9:.2f} GB · pytorch {alloc / 1e9:.2f} GB "
            f"· resident [{resident}]")


REPORT_DIR = os.path.join(_PERSIST if os.path.isdir(_PERSIST) else HERE, "reports")


@spaces.GPU(duration=60)
def vram_diag():
    """What is actually on the device right now?  Measured, not assumed.

    Called before every training chunk while the memory budget is being fixed; it
    answers the only question that matters — did the teachers really load, and how
    much is still free for the student.
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    lines = [f"**device** `{torch.cuda.get_device_name(0)}`",
             f"**free** {_vram_free_gb():.2f} GB",
             f"**pytorch allocated** {torch.cuda.memory_allocated() / 1e9:.2f} GB",
             f"**pytorch reserved** {torch.cuda.memory_reserved() / 1e9:.2f} GB"]
    if ENGINE is None:
        lines.append("**ENGINE** None")
    else:
        lines.append(f"**ENGINE resident** {ENGINE.names} ({len(ENGINE.names)}/{len(SPECS)} specs)")
        lines.append(f"**spine** |V| = {len(ENGINE.spine):,}" if ENGINE.spine is not None else "**spine** None")
    for l in LOAD_LOG[-12:]:
        lines.append(f"- {l}")
    return "\n".join(lines)


@spaces.GPU(duration=240)
def evaluate_student(n: int = 8, offset: int = 0, k: int = 256, max_len: int = 512,
                     teachers_n: int = 2, offset_test: bool = True):
    """Measure the distilled student against the fused teacher — the §6 gates.

    Computes, on held-out MATH **test** rows (never seen in training):
      * KL(q_fused || p_student) on the stored support   (gate: < 0.35 nats)
      * top-1 agreement with the fused argmax            (gate: >= 70 %)
      * truncation floor for reference                   (k=256: 0.030 nats)
    Nothing here looks at L2 task accuracy; that needs generation and is separate.
    """
    import gc

    import train_distill as TD
    from targets import build_T_target, distill_kl

    t0 = time.perf_counter()
    if ENGINE is None:
        raise gr.Error("models failed to load — see Diagnostics")
    keep, evicted = _select_teachers(ENGINE, teachers_n)
    if evicted:
        LOAD_LOG.append(f"♻️ evicted {evicted} → {_vram_free_gb():.1f} GB free")

    from teacher import shared_offsets

    teachers = _teacher_stack(keep)

    rows = corpus_rows(TEST_PATH) if offset_test else corpus_rows()
    lo = max(0, int(offset))
    batch = rows[lo:lo + int(n)]
    if not batch:
        raise gr.Error("offset past the end of the test split")

    student_tok, student = TD.build_student("Qwen/Qwen3.5-0.8B", "2fc0636", dtype=torch.bfloat16,
                                            device=str(next(iter(teachers.model.values())).device),
                                            log=lambda m: LOAD_LOG.append(m))
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
    ck = TD.latest_ckpt(CKPT_DIR)
    ck_step = 0
    if ck:
        ck_step, _ = TD.load_ckpt(ck, student, opt)
    student.eval()

    spine_ids = torch.as_tensor(TD._spine_ids(teachers), dtype=torch.long).cuda()
    texts = [r["problem"] + TD.PROMPT_SUFFIX + r["solution"] for r in batch]
    fb = teachers.logits(texts, max_len=max_len)
    fused = teachers.fused_target(fb, {n_: 1.0 for n_ in keep}, ragged=True)
    offs = shared_offsets(fb, {n_: 1.0 for n_ in keep})

    dev = next(student.parameters()).device
    total_pos = 0
    kl_sum = 0.0
    agree = 0
    floors = []
    per_row = []
    for b, r in enumerate(batch):
        enc = student_tok(texts[b], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(dev)
        so = _t._char_offsets(student_tok, ids[0].cpu())
        m = {o: i for i, o in enumerate(so)}
        pairs = [(m[o] - 1, j) for j, o in enumerate(offs[b]) if o in m and m[o] - 1 >= 0]
        if not pairs:
            per_row.append({"subject": r["subject"], "positions": 0, "kl": None, "top1": None})
            continue
        with torch.no_grad():
            logits = student(input_ids=ids,
                             attention_mask=enc.get("attention_mask", torch.ones_like(ids)).to(dev),
                             use_cache=False).logits[0]
        si = torch.as_tensor([p[0] for p in pairs], device=dev)
        tj = torch.as_tensor([p[1] for p in pairs], device=dev)
        tgt = torch.from_numpy(fused[b]).to(dev)[tj]                    # (n, |V|) log-probs
        ki, kv, tail = build_T_target(tgt, k=min(k, tgt.shape[-1]), T=1.0)
        with torch.no_grad():
            kl = distill_kl(logits[si], ki, kv, spine_ids, T=1.0)
            # top-1 agreement on the student's *own* vocabulary vs the fused argmax
            fused_top = ki.gather(-1, kv.argmax(-1, keepdim=True)).squeeze(-1)   # spine idx
            stu_top_spine = logits[si].index_select(-1, spine_ids).argmax(-1)
            a = int((fused_top == stu_top_spine).sum())
        total_pos += len(pairs)
        kl_sum += float(kl.sum())
        agree += a
        floors.append(float(tail.mean()))
        per_row.append({"subject": r["subject"], "positions": len(pairs),
                        "kl": round(float(kl.mean()), 4), "top1": round(a / len(pairs), 3)})

    del student, opt
    gc.collect(); torch.cuda.empty_cache()

    if total_pos == 0:
        raise gr.Error("no aligned positions in this batch")
    kl_mean = kl_sum / total_pos
    agree_pct = 100.0 * agree / total_pos
    floor = sum(floors) / len(floors)
    dt = time.perf_counter() - t0

    def gate(name, value, ok):
        return f"| {name} | {value} | {'✅ PASS' if ok else '❌ FAIL'} |"

    md = [f"### L3 evaluation · {len(batch)} held-out MATH problems ({dt:.0f}s)",
          f"checkpoint **step {ck_step}** (`{os.path.basename(ck) if ck else 'none'}`) · "
          f"{total_pos:,} aligned positions · k={k}",
          "",
          "| gate | value | |",
          "|---|---|---|",
          gate("KL(student ‖ fused)", f"{kl_mean:.4f} nats", kl_mean < 0.35),
          gate("top-1 agreement", f"{agree_pct:.1f} %", agree_pct >= 70.0),
          gate("truncation floor (reference)", f"{floor:.4f} nats", floor < 0.01),
          "",
          f"> §6 gates: KL < 0.35 nats, top-1 ≥ 70 %. The student is compared to the "
          f"**fused** law (the thing it is trained on), not to any single teacher.",
          ""]
    out = {f"eval_step{ck_step}": {"kl": kl_mean, "top1": agree_pct, "positions": total_pos,
                                   "floor": floor, "rows": len(batch)}}
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(os.path.join(REPORT_DIR, f"eval_step{ck_step}.json"), "w") as f:
            json.dump(out, f, indent=1)
        md.append(f"saved to `{REPORT_DIR}/eval_step{ck_step}.json`")
    except Exception as e:
        md.append(f"(could not save report: {type(e).__name__})")
    return "\n".join(md), pd.DataFrame(per_row)


def distill_md():
    f = _ckpt_facts()
    h = DISTILL_STATE["history"]
    if not h:
        return ("### L3 distillation\n\nNot started. The plan needs ~7.2 A100-h at k=256 for the full "
                "20 k-sequence run; each click of *Run chunk* trains a few steps and checkpoints.")
    losses = [l for _, l, _ in h]
    el = time.time() - (DISTILL_STATE["started"] or time.time())
    return (f"### L3 distillation\n\n"
            f"- **{f['steps']} steps on disk** (`{f['dir']}`, {f['files']} checkpoints)\n"
            f"- this process: {len(h)} steps in {el / 60:.1f} min · "
            f"loss {losses[0]:.3f} → {losses[-1]:.3f} (min {min(losses):.3f})\n"
            f"- targets: k={f.get('k') or 256} · T={f.get('T')}\n\n"
            f"> counts on disk survive restarts; the in-process numbers reset with the worker")


def diagnostics_md():
    lines = [f"- {l}" for l in LOAD_LOG]
    lines.append(f"- torch {torch.__version__} · transformers {transformers.__version__} · gradio {gr.__version__}")
    if ENGINE is not None:
        lines.append(f"- active generators: {', '.join(ENGINE.names)} · shared spine **{len(ENGINE.spine):,}** tokens")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
CSS = """
#col-container { max-width: 1180px; margin: 0 auto; }
"""

def _spine_size_text() -> str:
    return f"{len(ENGINE.spine):,}" if ENGINE is not None else "—"


INTRO = f"""# 🧬 6in1 — frame1 的骨骼，五個模型的器官
**K2-Horizon-0.9B · Limite-1B-Violetto · Qwen3.5-0.8B · DeepSeek-R1-Distill-Qwen-1.5B** 共用一個 **{_spine_size_text()}** 個 token 的表面字母表，
以 **exact logit fusion** 逐 token 合成一條序列；**LFM2.5-Embedding-350M** 做路由、判斷與 [frame1](https://github.com/vv0101v01/frame1) 128-agent 耦合圖的度量器官；
**frame1**（MoonBit）係骨骼：狀態空間、耦合圖、共識、可判定憑證。 四模型詞表兩兩不同，直接合併權重不可行 → L1 系統合成 · L2 輸出融合。
"""

with gr.Blocks(title="6in1") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(INTRO)
        with gr.Tab("🧭 Router + fused answer"):
            with gr.Row():
                prompt = gr.Textbox(show_label=False, placeholder="Ask anything — maths, code, planning, 中文…", container=False, scale=5, lines=2)
                run = gr.Button("Run 6in1", variant="primary", scale=1)
            with gr.Row():
                mode = gr.Radio(["Auto (LFM router)", "All four"], value="Auto (LFM router)", label="Who answers")
                max_new = gr.Slider(64, 1024, value=384, step=32, label="max new tokens")
                temperature = gr.Slider(0, 1.2, value=0.6, step=0.05, label="temperature (0 = greedy)")
                seed = gr.Number(value=0, precision=0, label="seed")
            summary = gr.Markdown()
            with gr.Row():
                wplot = gr.BarPlot(x="model", y="weight", title="LFM router weights (top-16 agent vote)", height=220, scale=1)
                agents_df = gr.Dataframe(label="Nearest frame1 agents to the query", wrap=True, scale=2)
            with gr.Accordion("Fused reasoning", open=False):
                reasoning = gr.Textbox(lines=14, show_label=False)
            dist_df = gr.Dataframe(label="Router distribution over the four generators")
            rout_outs = [summary, wplot, agents_df, reasoning, dist_df]
            run.click(hexad_router, [prompt, mode, max_new, temperature, seed], rout_outs)
            prompt.submit(hexad_router, [prompt, mode, max_new, temperature, seed], rout_outs)
            gr.Examples(
                examples=[
                    ["What is the sum of all positive integers n such that n^2 + 12n - 2007 is a perfect square? Put the final answer in \\boxed{}."],
                    ["Write a Python function that returns the longest palindromic substring of a string, and explain its complexity."],
                    ["用廣東話解釋乜嘢係代數連通度 (algebraic connectivity)。"],
                ],
                inputs=[prompt],
                outputs=rout_outs,
                fn=hexad_router,
                cache_examples=False,
            )

        with gr.Tab("🔀 Exact 4-way fusion"):
            gr.Markdown(
                f"四個生成器的詞表兩兩不同（64 256 / 151 680 / 248 320 / 151 936），但**交集有 {_spine_size_text()} 個 byte-identical token**"
                "（由 tokenizer 即時計算，見 Diagnostics），覆蓋真實中英數文字約 82–100%（視語言；英文代碼最低）。"
                "每一步先在交集上把每個模型的分布**限制並重新歸一化**，再合成 `p ∝ ∏ p_m^{w_m}`（product）或 `∑ w_m p_m`（mixture）——"
                "乘上嘅係每步常數，product 規則下完全抵銷，所以呢個係**精確**融合，唔係近似。"
            )
            with gr.Row():
                fprompt = gr.Textbox(show_label=False, placeholder="A maths / reasoning question…", container=False, scale=5, lines=2)
                frun = gr.Button("Fuse", variant="primary", scale=1)
            active = gr.CheckboxGroup(GEN_NAMES, value=GEN_NAMES, label="Active generators (≥2)")
            with gr.Row():
                k2w = gr.Slider(0, 1, value=0.25, step=0.05, label="w · K2")
                limw = gr.Slider(0, 1, value=0.25, step=0.05, label="w · Limite")
                qw = gr.Slider(0, 1, value=0.25, step=0.05, label="w · Qwen3.5")
                r1w = gr.Slider(0, 1, value=0.25, step=0.05, label="w · R1")
            with gr.Row():
                rule = gr.Radio(["product (log-linear)", "mixture (linear)"], value="product (log-linear)", label="fusion rule")
                fmax = gr.Slider(64, 1024, value=512, step=32, label="max new tokens")
                ftemp = gr.Slider(0, 1.2, value=0.6, step=0.05, label="temperature (0 = greedy)")
                ftopp = gr.Slider(0.5, 1, value=0.95, step=0.01, label="top-p")
                fseed = gr.Number(value=0, precision=0, label="seed")
                fthink = gr.Slider(0, 1, value=0.75, step=0.05, label="budget forcing onset (share of budget)")
            fanswer = gr.Markdown()
            with gr.Row():
                freason = gr.Textbox(label="fused stream", lines=12, scale=3)
                fstats = gr.Dataframe(label="fusion diagnostics", scale=2)
            fouts = [fanswer, freason, fstats]
            finputs = [fprompt, active, k2w, limw, qw, r1w, rule, fmax, ftemp, ftopp, fseed, fthink]
            frun.click(fuse_models, finputs, fouts)
            gr.Examples(
                examples=[["How many positive divisors does 2024 have? Put the final answer in \\boxed{}."],
                          ["Find the remainder when 7^100 is divided by 13."]],
                inputs=[fprompt],
                outputs=fouts,
                fn=fuse_models,
                cache_examples=False,
            )

        with gr.Tab("🕸️ frame1 coupling graph"):
            gr.Markdown(
                "frame1 Round 1 couples 128 agents on a **ring** (λ₂ = 0.0024). 6in1 replaces it with a graph built from "
                "**real LFM2.5 vectors** of the agents' role cards: symmetric k-NN, heat-kernel weights `exp((cos−1)/τ)`, "
                "one bridge per agent to every other model family, Kruskal repair (frame1 module 25, ported to numpy — "
                "reproduces 1275 edges / λ₂ = 0.540 at k=8, τ=0.25)."
            )
            with gr.Row():
                gk = gr.Slider(1, 24, value=8, step=1, label="k")
                gtau = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="τ")
                gbtn = gr.Button("Rebuild 128-agent graph", variant="primary")
            with gr.Row():
                gstats = gr.Dataframe(label="ring vs LFM k-NN", scale=2)
                gfig = gr.Plot(label="agents (PCA of LFM vectors), black = cross-family edges", scale=3)
            gdyn = gr.LinePlot(x="step", y="value", color="series", title="numpy dynamics on both graphs (dispersion ↓, Kuramoto R ↑)", height=300)
            with gr.Accordion("frame1 MoonBit run: 128 agents × 640 steps (hexad_trace.csv)", open=False):
                tr = TRACE.melt(id_vars="step", var_name="series", value_name="value")
                gr.LinePlot(value=tr, x="step", y="value", color="series", height=320)
            with gr.Accordion("Your own population (LFM on GPU)", open=False):
                ctext = gr.Textbox(
                    lines=8,
                    label="one agent per line — `family | description`",
                    value="\n".join(
                        [
                            "math | proves inequalities and olympiad number theory",
                            "math | solves geometry with coordinates",
                            "code | writes Python data pipelines",
                            "code | debugs concurrency bugs in Rust",
                            "chat | translates between Cantonese and English",
                            "chat | summarises long documents politely",
                        ]
                    ),
                )
                with gr.Row():
                    ck = gr.Slider(1, 12, value=2, step=1, label="k")
                    ctau = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="τ")
                    cbtn = gr.Button("Embed + build graph")
                with gr.Row():
                    cstats = gr.Dataframe(scale=2)
                    cfig = gr.Plot(scale=3)
                cdyn = gr.LinePlot(x="step", y="value", color="series", height=260)
            gouts = [gstats, gfig, gdyn]
            gbtn.click(frame1_graph, [gk, gtau], gouts)
            cbtn.click(custom_graph, [ctext, ck, ctau], [cstats, cfig, cdyn])

        with gr.Tab("🩺 Diagnostics"):
            diag = gr.Markdown(diagnostics_md())
            with gr.Accordion("Shared alphabet (the fusion substrate)", open=True):
                gr.Markdown(pairwise_md())
                stab = gr.Dataframe(value=spine_table(), label="per-model mapping into the shared spine")
            gr.Markdown("### L3 baseline: fused system on licence-checked MATH")
            with gr.Row():
                bl_split = gr.Dropdown(["train", "test"], value="test", label="split")
                bl_n = gr.Slider(1, 40, value=6, step=1, label="problems")
                bl_off = gr.Slider(0, 14000, value=0, step=10, label="offset into corpus")
                bl_max = gr.Slider(128, 768, value=320, step=32, label="max new tokens")
                bl_temp = gr.Slider(0, 1, value=0.3, step=0.05, label="temperature")
                bl_btn = gr.Button("Run baseline", variant="primary")
            bl_md = gr.Markdown()
            bl_df = gr.Dataframe(label="per-problem result")
            bl_btn.click(run_baseline, [bl_n, bl_off, bl_max, bl_temp, bl_split], [bl_md, bl_df])

            gr.Markdown("### L3 distillation (LoRA on Qwen3.5-0.8B)")
            gr.Markdown("Trains the student to reproduce the fused law, teacher-forced, top-256 targets. "
                        "Each click = one chunk; it resumes from the last checkpoint.")
            with gr.Row():
                ds_steps = gr.Slider(1, 20, value=4, step=1, label="steps this chunk")
                ds_k = gr.Dropdown([64, 128, 256], value=256, label="top-k")
                ds_lr = gr.Slider(1e-5, 5e-4, value=1e-4, step=1e-5, label="learning rate")
                ds_max = gr.Slider(256, 2048, value=1024, step=64, label="max tokens (hard cap: 1024 很少觸及)")
                ds_lim = gr.Slider(0, 7500, value=1200, step=50, label="rows to sample from (0 = all)")
                ds_tn = gr.Slider(2, 4, value=2, step=1, label="teachers used")
                ds_fresh = gr.Checkbox(value=False, label="restart from scratch")
            with gr.Row():
                ds_btn = gr.Button("Run chunk", variant="primary")
                ds_ref = gr.Button("Refresh status")
            ds_out = gr.Markdown(distill_md())
            gr.Markdown("#### Evaluate the student against the fused teacher (§6 gates)")
            with gr.Row():
                ev_n = gr.Slider(2, 24, value=8, step=1, label="held-out problems")
                ev_off = gr.Slider(0, 4900, value=0, step=1, label="offset in test split")
                ev_btn = gr.Button("Evaluate", variant="secondary")
            ev_md = gr.Markdown()
            ev_df = gr.Dataframe(label="per problem")
            ev_btn.click(evaluate_student, [ev_n, ev_off], [ev_md, ev_df])
            vd_btn = gr.Button("VRAM diagnostic (what is resident?)", variant="secondary")
            vd_md = gr.Markdown()
            vd_btn.click(vram_diag, [], [vd_md])
            ds_btn.click(distill_step, [ds_steps, ds_k, ds_lr, ds_max, ds_lim, ds_fresh, ds_tn], ds_out)
            ds_ref.click(distill_md, [], ds_out)

            gr.Markdown("### L3 planning: target sparsity (measured, not assumed)")
            with gr.Row():
                sp_prompt = gr.Textbox(value="Find the remainder when 7^100 is divided by 13.", show_label=False, scale=4)
                sp_len = gr.Slider(64, 768, value=256, step=32, label="positions", scale=1)
                sp_btn = gr.Button("Measure", variant="primary", scale=1)
            sp_md = gr.Markdown()
            sp_df = gr.Dataframe(label="retained mass Z_k per target")
            sp_btn.click(measure_sparsity, [sp_prompt, sp_len], [sp_md, sp_df])

            sc_btn = gr.Button("Run LFM self-check (HF vs llama.cpp vectors)")
            sc_out = gr.Markdown()
            sc_btn.click(lfm_selfcheck, [], sc_out)

    demo.load(frame1_graph, None, gouts)

if __name__ == "__main__":
    demo.queue(max_size=16)
    demo.launch(theme=gr.themes.Citrus(), css=CSS, ssr_mode=False)
