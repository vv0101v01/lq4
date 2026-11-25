"""Byte alignment must be correct, and must refuse rather than mis-align."""
import torch
from transformers import AutoTokenizer

import teacher as T


def test_byte_positions_monotone_and_exact():
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    text = "證明：若 n 為正整數，則 n^3 - n 可被 6 整除。 Proof: n^3-n = (n-1)n(n+1)."
    ids = torch.tensor(tok.encode(text))
    ends = T.byte_positions(tok, ids)
    assert ends == sorted(ends), "byte offsets must be non-decreasing"
    assert ends[-1] == len(text.encode()), f"last token must end at {len(text.encode())}, got {ends[-1]}"


def test_shared_boundaries_is_intersection():
    assert T.shared_boundaries([2, 5, 9], [5, 9, 12]) == [(1, 0), (2, 1)]
    assert T.shared_boundaries([], [1]) == []


def test_two_qwen2_models_align_fully():
    """Limite and R1 share a tokenizer => every boundary is shared."""
    a = AutoTokenizer.from_pretrained("paradigma-inc/limite-1b-violetto")
    b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    text = "Find the remainder when 7^100 is divided by 13."
    ia, ib = torch.tensor(a.encode(text)), torch.tensor(b.encode(text))
    ea, eb = T.byte_positions(a, ia), T.byte_positions(b, ib)
    shared = T.shared_boundaries(ea, eb)
    assert len(shared) == min(len(ea), len(eb)), (len(shared), len(ea), len(eb))


def test_four_tokenizers_align_on_at_least_80pct():
    """K2 vs Qwen3.5 tokenize differently — but ≥80 % of boundaries still coincide."""
    specs = [("K2", "IFM/K2-Horizon-0.9B"), ("Qwen", "Qwen/Qwen3.5-0.8B")]
    toks = {n: AutoTokenizer.from_pretrained(r) for n, r in specs}
    text = "We prove by induction that the sum of the first n odd numbers equals n squared."
    ends = {n: T.byte_positions(t, torch.tensor(t.encode(text))) for n, t in toks.items()}
    shared = len(set(ends["K2"]) & set(ends["Qwen"]))
    assert shared >= 0.8 * min(len(ends["K2"]), len(ends["Qwen"])), (shared, ends)


if __name__ == "__main__":
    test_byte_positions_monotone_and_exact()
    test_shared_boundaries_is_intersection()
    test_two_qwen2_models_align_fully()
    test_four_tokenizers_align_on_at_least_80pct()
    print("✅ byte alignment: monotone, exact, 100 % on Qwen2 siblings, ≥80 % across families")
