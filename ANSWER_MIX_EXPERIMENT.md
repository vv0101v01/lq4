# Paired answer-loss trial

Two isolated trials on NVIDIA L4, each starting from `/data/ckpt/step_1826.pt`. The production checkpoint was not overwritten and neither branch was promoted. Teacher pair: K2-Horizon-0.9B and Limite-1B-Violetto; train limit 1200, max length 512, k=256, temperature 2, requested learning rate 2e-4. One training row with an exact match in the MATH test corpus was excluded before limiting. Results use the same first 8 MATH test rows, 1120 aligned token positions. These are a small evaluation sample; no task accuracy was measured.

| trial | steps per arm | answer CE weight | initial KL / top-1 | baseline KL / top-1 | mixed KL / top-1 |
|---|---:|---:|---|---|---|
| `/data/experiments/paired_224780568110` | 12 | 0.25 | 1.3504 / 53.21% | 1.3834 / 53.04% | 1.5120 / 45.00% |
| `/data/experiments/paired_45e9880f060d` | 36 | 0.05 | 1.3504 / 53.21% | 1.3665 / 52.95% | 1.3822 / 52.32% |

These short trials do not establish a long-run outcome. Neither mixed run outperformed its paired baseline on this small sample; do not promote them based on these results. Earlier calls that passed `offset_test=False` evaluated training rows, despite being described as held-out. They should not be compared as held-out results. To make a decision, evaluate more truly held-out questions and task accuracy, then consider a longer run if justified.

The CUDA ring convolution is not called by these training trials. It was separately tested on L4 against direct convolution, not on T4 and not as part of the training model.
