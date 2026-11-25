"""bench_gpu.py — the numbers that matter: FFT convolution for a BATCH of rings.

Run on the T4. Prints measured milliseconds only; if it runs on CPU it says so loudly,
because a CPU number here answers a question nobody asked.
"""
from __future__ import annotations

import sys

import torch

import ring_fft_gpu as G


def main() -> int:
    dev = G.device()
    print(f"device: {dev}")
    if dev.type != "cuda":
        print("!! running on CPU — these numbers do NOT answer the GPU question")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"   {p.name} · {p.total_memory / 1e9:.2f} GB · sm_{p.major}{p.minor}")

    print(f"\n{'N':>4} {'steps':>6} {'batch':>6} {'matmul ms':>11} {'fft ms':>9} "
          f"{'speedup':>8} {'max|diff|':>10}")
    for n in (8, 16, 32, 64, 128):
        r = G.bench(n, steps=200, batch=64, reps=20, dev=dev)
        print(f"{r['n']:4d} {r['steps']:6d} {r['batch']:6d} {r['matmul_ms']:11.4f} "
              f"{r['fft_ms']:9.4f} {r['speedup']:7.2f}x {r['max_abs_diff']:10.2e}")

    print("\nscaling with batch (N=128, 200 steps) — the agent population size")
    for batch in (1, 8, 64, 512, 4096):
        r = G.bench(128, steps=200, batch=batch, reps=10, dev=dev)
        print(f"  batch {batch:5d}: matmul {r['matmul_ms']:9.4f} ms · fft {r['fft_ms']:9.4f} ms "
              f"· {r['speedup']:6.2f}x")

    print("\nscaling with steps (N=128, batch=64)")
    for steps in (50, 200, 800):
        r = G.bench(128, steps=steps, batch=64, reps=10, dev=dev)
        print(f"  steps {steps:5d}: matmul {r['matmul_ms']:9.4f} ms · fft {r['fft_ms']:9.4f} ms "
              f"· {r['speedup']:6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
