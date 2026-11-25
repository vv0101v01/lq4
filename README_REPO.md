# lq4 — 6-into-1 model fusion

Four small language models (K2-Horizon-0.9B, Limite-1B-Violetto, Qwen3.5-0.8B,
DeepSeek-R1-Distill-Qwen-1.5B) are combined into one system, in three rungs:

| rung | what it does | where |
|---|---|---|
| **L1 routing** | an embedding router picks 2 of 4 models per prompt | `app.py` (`/hexad_router`) |
| **L2 exact fusion** | per-token product of experts across 4 different tokenizers, aligned on **surface bytes** | `spine.py`, `fuse.py`, `teacher.py` |
| **L3 distillation** | one 0.8B student learns the fused distribution | `targets.py`, `train_distill.py` |

`frame1`'s agent graph is retained as a separate tab (`graph.py`).

## The four ideas that make it work

1. **Byte alignment, not token alignment.** Four tokenizers split the same text
   differently (K2 → 14 tokens where Qwen3.5 → 17). Token positions are not
   comparable; byte offsets are. Only positions where *every* model has a boundary
   are fused — measured yield 95.4 % pooled.
2. **A shared spine.** Every model's vocabulary is mapped onto the 42,207-token
   byte intersection of all four, so "the fused law" is a well-defined distribution.
3. **Sparse targets.** k = 256, chosen by measurement, not taste: the truncation
   floor is 0.030 nats (k=64 gives 0.063 and fails the 0.05 budget). Storage 126 GB
   for 20k×4k tokens instead of 6.9 TB.
4. **One temperature, one denominator.** See `targets.distill_kl` — three separate
   bugs lived there, all caught by the test "a perfect student must score 0".

## What is measured, and what is not

| claim | status |
|---|---|
| fusion runs on real weights (ZeroGPU) | ✅ live endpoint `/measure_sparsity` |
| byte-alignment yield 95.4 % | ✅ measured, 5 text domains |
| k = 256 | ✅ measured on real teacher laws |
| distillation loss falls 7.22 → 0.87 | ✅ 58 steps on a T4 |
| student matches the fused law (KL < 0.35) | ❌ **not yet evaluated** |
| 6-into-1 task accuracy | ❌ **not yet evaluated** |

L2 task accuracy is 1/6 (17 %) on MATH algebra — the fused system reliably derives
the right value and then *writes it in the wrong format* and restarts. That is
recorded in `DISTILL_PLAN.md` §8.3 rather than hidden.

## Run the tests

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install "transformers==5.15.1" numpy pandas matplotlib peft spaces
python run_tests.py          # 14 suites, CPU, no downloads
python run_tests.py --fast   # skip the statistical ones
```

CI runs the same command on Python 3.10 and 3.12, plus a hygiene job that fails on
committed tokens, orphan modules and unregistered selftests.

## Data

`dataset.py` builds the corpus from **MIT-licensed** sources only
(`EleutherAI/hendrycks_math`, `openai/gsm8k`), verifies the licence tag through the
HF API at build time, and writes a manifest so provenance travels with the file.
The corpus itself is not committed (regenerate with `python3 dataset.py --sources math`).
