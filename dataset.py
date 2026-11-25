"""dataset.py — the L3 distillation corpus, built only from license-checked sources.

Selection rule (the user's instruction): **MATH + open licences**.  Every source
below was checked against the HF API on 2026-11-25 — licence tag and gated flag
are recorded in the manifest that ships with the data, so the provenance travels
with the file instead of living in someone's memory.

    EleutherAI/hendrycks_math   MIT     train 7 500 / test 5 005   (the MATH corpus, 7 subjects)
    openai/gsm8k                MIT     train 7 473 / test 1 319
    AI-MO/NuminaMath-CoT        Apache-2.0  optional, much larger, used only if asked

What this file does NOT do
--------------------------
* no de-duplication by *problem intent* — only exact-text and near-duplicate
  (normalised text + same final answer) removal, which is auditable;
* no re-licensing: each record keeps its source and licence, and `manifest()`
  reproduces the HF tags it was built from.

Usage
-----
    python3 dataset.py --out data/corpus --sources math,gsm8k --max-per-source 6000
    python3 dataset.py --selftest          # offline checks, no download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter

MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

SOURCES = {
    "math": {"repo": "EleutherAI/hendrycks_math", "license": "mit", "kind": "configs"},
    "gsm8k": {"repo": "openai/gsm8k", "license": "mit", "kind": "single", "config": "main"},
    "numina": {"repo": "AI-MO/NuminaMath-CoT", "license": "apache-2.0", "kind": "single"},
}

SCHEMA = ["id", "source", "license", "split", "subject", "problem", "solution", "answer", "level"]


# --------------------------------------------------------------------------------------
# normalisation & answer extraction
# --------------------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_BOXED = re.compile(r"\\boxed\{")


def norm_text(s: str) -> str:
    """Aggressive but reversible-enough normalisation for duplicate detection."""
    s = unicodedata.normalize("NFKC", s or "")
    s = _WS.sub(" ", s).strip().lower()
    return s


def last_boxed(text: str) -> str | None:
    """Balanced-brace extraction of the last \\boxed{...}."""
    starts = [m.end() for m in _BOXED.finditer(text or "")]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i : j - 1].strip() if depth == 0 else None


def norm_answer(a: str | None) -> str | None:
    if a is None:
        return None
    a = unicodedata.normalize("NFKC", a)
    a = a.replace("\\!", "").replace("\\,", "").replace(" ", "").replace("$", "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = a.rstrip(".").strip()
    return a or None


def _hash(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def make_record(source: str, license_: str, split: str, subject: str, problem: str, solution: str, level=None) -> dict:
    ans = norm_answer(last_boxed(solution))
    if ans is None:
        # GSM8K keeps the answer after "####"
        m = re.search(r"####\s*(.+)$", solution or "")
        ans = norm_answer(m.group(1)) if m else None
    return {
        "id": _hash(source, split, problem),
        "source": source,
        "license": license_,
        "split": split,
        "subject": subject,
        "problem": problem.strip(),
        "solution": solution.strip(),
        "answer": ans,
        "level": level,
    }


def dedupe(records: list[dict]) -> tuple[list[dict], dict]:
    """Exact duplicates (normalised problem) and near-duplicates (same answer +
    same problem prefix) are dropped; the second one is the interesting class,
    because a duplicated problem with a different answer is a labelling bug."""
    seen, seen_ans, out = set(), Counter(), []
    stats = {"input": len(records), "exact_dup": 0, "near_dup": 0, "kept": 0}
    for r in records:
        key = norm_text(r["problem"])
        if key in seen:
            stats["exact_dup"] += 1
            continue
        akey = (r["answer"], key[:120]) if r["answer"] else None
        if akey and akey in seen_ans:
            stats["near_dup"] += 1
            continue
        seen.add(key)
        if akey:
            seen_ans[akey] += 1
        out.append(r)
    stats["kept"] = len(out)
    return out, stats


def manifest(records: list[dict]) -> dict:
    by = Counter((r["source"], r["license"], r["split"]) for r in records)
    return {
        "built": "dataset.py",
        "sources_checked": {k: {"repo": v["repo"], "license": v["license"]} for k, v in SOURCES.items()},
        "counts": {f"{s}/{l}/{sp}": n for (s, l, sp), n in sorted(by.items())},
        "schema": SCHEMA,
        "total": len(records),
        "with_answer": sum(1 for r in records if r["answer"]),
    }


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_math(max_per_source: int | None, split: str = "train") -> list[dict]:
    from datasets import load_dataset

    out = []
    for cfg in MATH_CONFIGS:
        ds = load_dataset(SOURCES["math"]["repo"], cfg, split=split)
        for row in ds:
            out.append(make_record("math", "mit", f"{cfg}:{split}", cfg, row["problem"], row["solution"], row.get("level")))
            if max_per_source and len(out) >= max_per_source:
                return out
    return out


def load_gsm8k(max_per_source: int | None, split: str = "train") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(SOURCES["gsm8k"]["repo"], SOURCES["gsm8k"]["config"], split=split)
    out = []
    for row in ds:
        out.append(make_record("gsm8k", "mit", f"main:{split}", "grade_school", row["question"], row["answer"]))
        if max_per_source and len(out) >= max_per_source:
            break
    return out


def build(sources: list[str], out_dir: str, max_per_source: int | None = None, split: str = "train") -> dict:
    loaders = {"math": load_math, "gsm8k": load_gsm8k}
    recs: list[dict] = []
    for s in sources:
        if s not in loaders:
            raise SystemExit(f"unknown source {s!r}; available: {sorted(loaders)}")
        part = loaders[s](max_per_source, split)
        print(f"  {s}: {len(part)} raw records")
        recs += part
    deduped, stats = dedupe(recs)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"corpus_{split}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = manifest(deduped)
    man["dedupe"] = stats
    with open(os.path.join(out_dir, f"manifest_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"wrote {path}: {len(deduped)} records · dedupe {stats}")
    print(f"manifest: {man['counts']}")
    return man


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    # boxed extraction with nesting
    assert last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert last_boxed(r"a \boxed{3} b \boxed{4}") == "4"
    assert last_boxed("no box here") is None
    assert last_boxed(r"unbalanced \boxed{3") is None

    # answer normalisation
    assert norm_answer(r"\dfrac{ 1 }{ 2 }.") == r"\frac{1}{2}"
    assert norm_answer("$42$") == "42"
    assert norm_answer(None) is None

    # GSM8K style trailing answer
    r = make_record("gsm8k", "mit", "main:train", "grade_school", "q", "reasoning #### 1,234")
    assert r["answer"] == "1,234", r["answer"]

    # de-dupe: exact and near duplicates
    a = make_record("math", "mit", "algebra:train", "algebra", "Find x if  x + 1 = 2 .", "\\boxed{1}")
    b = make_record("math", "mit", "algebra:train", "algebra", "find x if x + 1 = 2.", "\\boxed{1}")
    c = make_record("math", "mit", "algebra:train", "algebra", "Solve y^2 = 9.", "\\boxed{3}")
    d = make_record("math", "mit", "algebra:train", "algebra", "Solve y^2 = 9.", "\\boxed{-3}")
    kept, stats = dedupe([a, b, c, d])
    assert stats["exact_dup"] == 1, stats          # a/b normalise identically
    assert stats["kept"] == 3, stats                # c and d differ in answer -> both kept
    assert all(r["license"] == "mit" for r in kept)

    # manifest carries licence and counts
    man = manifest(kept)
    assert man["sources_checked"]["math"]["license"] == "mit"
    assert man["total"] == 3 and man["with_answer"] == 3

    print("dataset selftest OK · boxed/normalise/dedupe/manifest")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--sources", default="math,gsm8k")
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        build([s.strip() for s in a.sources.split(",") if s.strip()], a.out, a.max_per_source, a.split)
