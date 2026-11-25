"""audit_vocab.py — recompute every vocabulary intersection with the SHIPPED path.

Run:  python3 audit_vocab.py            (needs only `transformers` + tokenizers)

The Space builds its fusion alphabet with `bytes_vocab.vocab_bytes` (raw surface
bytes, GPT-2 byte table).  This script prints the same numbers, plus the k-way
intersections and a text-coverage probe, so the README can be checked against
code rather than against an earlier prototype.
"""

from __future__ import annotations

import itertools
import json

from transformers import AutoTokenizer

from bytes_vocab import vocab_bytes
from spine import build_spine

SPECS = [
    ("K2", "IFM/K2-Horizon-0.9B", "6b4f28e81b8a77c8b44465a93a452eea198d4ceb"),
    ("Limite", "paradigma-inc/limite-1b-violetto", "190ee8d9c6777b7c5f8e7592cc00caf495a90ca7"),
    ("Qwen3.5", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("R1", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"),
]

PROBES = {
    "zh+en+math": "證明：若 n 為正整數，則 n^3 - n 可被 6 整除。 Proof: n^3 - n = (n-1)n(n+1) is a product of three consecutive integers, so it is divisible by 2 and 3.",
    "cantonese": "用廣東話解釋乜嘢係代數連通度，佢同圖嘅連通性有咩關係？",
    "english-code": "def longest_palindrome(s: str) -> str:\n    \"\"\"return the longest palindromic substring\"\"\"\n    best = ''\n    for i in range(len(s)):\n        for j in range(i, len(s)):\n            t = s[i:j+1]\n            if t == t[::-1] and len(t) > len(best):\n                best = t\n    return best",
}


def main() -> dict:
    names = [s[0] for s in SPECS]
    tok, surf = {}, {}
    for n, repo, rev in SPECS:
        tok[n] = AutoTokenizer.from_pretrained(repo, revision=rev)
        surf[n] = vocab_bytes(tok[n])

    pairs = {}
    for a, b in itertools.combinations(names, 2):
        pairs[(a, b)] = len(set(surf[a]) & set(surf[b]))

    kway = {}
    for k in (3, 4):
        for combo in itertools.combinations(names, k):
            kway[combo] = len(set.intersection(*[set(surf[c]) for c in combo]))

    spine = build_spine({n: (tok[n], [0]) for n in names})
    inter4 = set.intersection(*[set(surf[c]) for c in names])

    coverage = {}
    for label, text in PROBES.items():
        row = {}
        for n in names:
            ids = tok[n].encode(text)
            keys = [tok[n].convert_ids_to_tokens(int(i)) for i in ids]
            from bytes_vocab import token_to_bytes

            bs = [token_to_bytes(k) for k in keys]
            row[n] = round(100.0 * sum(1 for b in bs if b in inter4) / max(len(bs), 1), 1)
        coverage[label] = row

    print("surface vocabulary (raw bytes)")
    for n in names:
        print(f"  {n:9s} {len(surf[n]):7,d}")
    print("\npairwise intersection")
    for (a, b), v in pairs.items():
        print(f"  {a:9s} ∩ {b:9s} {v:7,d}")
    print("\nk-way intersection")
    for combo, v in kway.items():
        print(f"  {' ∩ '.join(combo):40s} {v:7,d}")
    print(f"\nbuild_spine() |V| = {len(spine):,}")
    print("\ncoverage: share of each model's tokens that land in the 4-way set")
    for label, row in coverage.items():
        print(f"  {label:13s} " + "  ".join(f"{n}={v:5.1f}%" for n, v in row.items()))

    out = {
        "surface": {n: len(surf[n]) for n in names},
        "pairs": {" ∩ ".join(k): v for k, v in pairs.items()},
        "kway": {" ∩ ".join(k): v for k, v in kway.items()},
        "spine": len(spine),
        "coverage_pct": coverage,
    }
    json.dump(out, open("audit_vocab.json", "w"), indent=1, ensure_ascii=False)
    print("\nwrote audit_vocab.json")
    return out


if __name__ == "__main__":
    main()
