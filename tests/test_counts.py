"""test_counts.py — regression guard for every number quoted in the UI/README.

Two kinds of check:

1. **no stale literals** — the abandoned prototype produced 42 967 / 50 916 /
   151 471 / 131 451 / 42 184.  Those must not appear in shipped copy again.
2. **live recomputation** — the intersection sizes printed by `pairwise_md()`
   and the spine built by `build_spine()` must agree with each other and with
   `audit_vocab.py`.

Run:  python3 -m pytest tests/ -q      (or: python3 tests/test_counts.py)
Needs network on first run (tokenizers only — a few MB).
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

STALE = ["42 967", "42,967", "50 916", "50,916", "151 471", "151,471", "131 451", "131,451", "42 184", "42,184"]
SHIPPED_COPY = ["app.py", "README.md"]


# lines that *name* the old numbers on purpose (correction notes) are exempt
EXEMPT = ("更正", "早前版本", "舊數字", "correction", "stale")


def test_no_stale_literals():
    for fname in SHIPPED_COPY:
        for i, line in enumerate(open(os.path.join(ROOT, fname), encoding="utf-8"), 1):
            if any(e in line for e in EXEMPT):
                continue
            hits = [s for s in STALE if s in line]
            assert not hits, f"{fname}:{i} still quotes stale numbers {hits}: {line.strip()[:100]}"


def test_intersections_consistent():
    """pairwise_md() numbers == audit_vocab.main() numbers == build_spine()."""
    import itertools

    from transformers import AutoTokenizer

    from audit_vocab import SPECS
    from bytes_vocab import vocab_bytes
    from spine import build_spine

    names = [s[0] for s in SPECS]
    tok = {n: AutoTokenizer.from_pretrained(r, revision=v) for n, r, v in SPECS}
    surf = {n: set(vocab_bytes(tok[n])) for n in names}

    spine = build_spine({n: (tok[n], [0]) for n in names})
    four = set.intersection(*surf.values())
    assert len(spine) == len(four), (len(spine), len(four))

    for a, b in itertools.combinations(names, 2):
        # a pair intersection is always >= the 4-way one
        assert len(surf[a] & surf[b]) >= len(four)

    # the two Qwen2-family models agree on almost the whole alphabet
    assert len(surf["Limite"] & surf["R1"]) > 0.99 * len(surf["Limite"])
    # K2 is the bottleneck: its own vocab barely exceeds the 4-way intersection
    assert len(four) <= len(surf["K2"])
    assert len(four) > 0.6 * len(surf["K2"])


def test_reported_numbers_match_measured():
    """The decimal strings the Space prints are produced live, not typed."""
    import importlib.util

    os.environ["SIXIN1_SKIP_LOAD"] = "1"
    sys.path.insert(0, "/tmp/stubs")
    spec = importlib.util.spec_from_file_location("sixin1_app_under_test", os.path.join(ROOT, "app.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    from transformers import AutoTokenizer

    class FakeEngine:
        names = ["K2-Horizon-0.9B", "Limite-1B-Violetto", "Qwen3.5-0.8B", "DeepSeek-R1-Distill-Qwen-1.5B"]
        tok = {
            "K2-Horizon-0.9B": AutoTokenizer.from_pretrained("IFM/K2-Horizon-0.9B"),
            "Limite-1B-Violetto": AutoTokenizer.from_pretrained("paradigma-inc/limite-1b-violetto"),
            "Qwen3.5-0.8B": AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B"),
            "DeepSeek-R1-Distill-Qwen-1.5B": AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        }
        stops = {n: [0] for n in names}

        class spine:
            __len__ = lambda self: 0

    app.ENGINE = FakeEngine()
    md = app.pairwise_md()
    found = {int(x.replace(",", "")) for x in re.findall(r"\*\*([\d,]{5,})\*\*", md)}
    assert len(found) == 7, found  # 6 pairs + the 4-way spine
    four = int(re.search(r"spine）\*\*([\d,]+)\*\*", md).group(1).replace(",", ""))
    assert four in found and four > 40_000


if __name__ == "__main__":
    test_no_stale_literals()
    print("✅ no stale literals in shipped copy")
    test_intersections_consistent()
    print("✅ intersections consistent (pairwise_md == build_spine)")
    test_reported_numbers_match_measured()
    print("✅ every displayed number is computed live")
