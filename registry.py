"""registry.py — the ONE place the six organs are declared.

Before this module the same four (name, repo, revision) triples were written out in
`app.py` (as ModelSpec objects) and again in `audit_vocab.py` (as tuples for the
independent vocabulary oracle).  Two copies of a pinned revision is exactly the kind
of drift `tests/test_counts.py` exists to catch, so the pins now live here and both
sides import them.

`audit_vocab.SPECS` and `app.SPECS` are derived from this, which means a stray edit
to one of them is no longer possible: there is only one of them.
"""

from __future__ import annotations

from fuse import ModelSpec

# Pinned by revision, never by branch: a moving `main` on the Hub silently changes
# which vocabulary the spine is built from.
SPECS: list[ModelSpec] = [
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
        trust=True,
    ),
]

# Loading all four fills a t4-small (14.6 / 14.74 GiB) and leaves nothing for the
# student, the LFM embedder or any activation.  Only these two are loaded at import;
# `_ensure_teachers(names)` loads more on demand and frees what it drops.
WARM_NAMES = {"K2-Horizon-0.9B", "Limite-1B-Violetto"}

# Short labels used by the UI and by audit_vocab's report.
SHORT = {
    "K2-Horizon-0.9B": "K2",
    "Limite-1B-Violetto": "Limite",
    "Qwen3.5-0.8B": "Qwen3.5",
    "DeepSeek-R1-Distill-Qwen-1.5B": "R1",
}


def revision_pins() -> dict[str, str]:
    """`{name: revision}` — what `audit_vocab` and the tests compare against."""
    return {s.name: s.rev for s in SPECS}
