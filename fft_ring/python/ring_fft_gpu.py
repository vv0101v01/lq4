"""ring_fft_gpu.py — cyclic convolution on the GPU with torch.fft.

Why torch.fft and not the hand-written kernel: torch.fft runs on the T4 the Space
already has, needs no nvcc, and is batched. The radix-8 kernel in csrc/ remains the
documented layout (and its logic is pinned by tests/test_radix.py) but the shipped
GPU path is torch.fft because it can be RUN here and THERE.

Nothing in this module is claimed to be fast. It reports what device it used and a
measured time; the caller decides.
"""
from __future__ import annotations

import torch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convolve(x: torch.Tensor, kernel: torch.Tensor, dev: torch.device | None = None) -> torch.Tensor:
    """Cyclic convolution along the last axis. x: (..., N) real or complex."""
    dev = dev or x.device
    x = x.to(dev)
    k = kernel.to(dev)
    n = x.shape[-1]
    if k.shape[-1] != n:
        raise ValueError(f"kernel {k.shape[-1]} != signal {n}")
    # real input -> rfft halves the work and the memory for the transform
    if not torch.is_complex(x):
        xc = torch.fft.rfft(x, dim=-1)
        kc = torch.fft.rfft(k, dim=-1)
        return torch.fft.irfft(xc * kc, n=n, dim=-1)
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * torch.fft.fft(k, dim=-1), dim=-1)


def diffuse(w: torch.Tensor, x0: torch.Tensor, steps: int, eta: float = 0.5,
            dev: torch.device | None = None) -> tuple[torch.Tensor, str]:
    """Ring diffusion on the GPU. Returns (per-step dispersion, device used).

    x0 is (N,) — one scalar per agent — or (N, d) for d features per agent. The ring
    is axis 0 in both cases, so the d-feature case is batched along axis 1.
    """
    dev = dev or device()
    n = int(w.shape[0])
    deg = w.sum(1).clamp_min(1e-30)
    kernel = (w / deg[:, None])[0]
    x = x0.to(dev).to(torch.float32)
    vec = x.ndim == 2
    state = x.T if vec else x                      # (d, N) or (N,)
    out = []
    for _ in range(steps + 1):
        dev_ = state - state.mean(-1, keepdim=True)
        out.append(dev_.norm(dim=0).mean() if vec else dev_.abs().mean())
        if _ < steps:
            conv = convolve(state, kernel)
            state = (1 - eta) * state + eta * conv
    return torch.stack(out), ("cuda" if dev.type == "cuda" else "cpu")


def bench(n: int, steps: int = 200, batch: int = 64, reps: int = 20,
          dev: torch.device | None = None) -> dict:
    """Measure FFT-convolution diffusion against matmul diffusion on the same device.

    `batch` runs the same recurrence for `batch` independent rings at once, which is
    the shape the agent population actually has.
    """
    import time

    dev = dev or device()
    w = torch.zeros(n, n, device=dev)
    idx = torch.arange(n, device=dev)
    w[idx, (idx + 1) % n] = 1.0
    w[(idx + 1) % n, idx] = 1.0
    p = w / w.sum(1, keepdim=True)
    kernel = p[0]

    x = torch.randn(batch, n, device=dev)

    def run_matmul():
        s = x.clone()
        for _ in range(steps):
            s = 0.6 * s + 0.4 * (s @ p.T)
        return s

    def run_fft():
        s = x.clone()
        for _ in range(steps):
            s = 0.6 * s + 0.4 * convolve(s, kernel, dev)
        return s

    def timeit(fn):
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    t_mat = timeit(run_matmul)
    t_fft = timeit(run_fft)
    a, b = run_matmul(), run_fft()
    return {
        "n": n, "steps": steps, "batch": batch, "device": str(dev),
        "matmul_ms": t_mat * 1e3, "fft_ms": t_fft * 1e3,
        "speedup": t_mat / t_fft,
        "max_abs_diff": float((a - b).abs().max()),
    }


def _selftest() -> None:
    torch.manual_seed(0)
    # logic only: FFT conv must equal the naive definition, on whatever device exists
    for n in (8, 16, 32, 64, 128):
        x = torch.randn(3, n, dtype=torch.float64)
        k = torch.randn(n, dtype=torch.float64)
        got = convolve(x, k)
        want = torch.empty_like(x)
        for r in range(x.shape[0]):
            for i in range(n):
                want[r, i] = sum(x[r, (i - j) % n] * k[j] for j in range(n))
        err = float((got - want).abs().max())
        assert err < 1e-9, (n, err)
    print(f"ring_fft_gpu: convolution == definition for N in 8..128 · device={device()}")


if __name__ == "__main__":
    _selftest()
