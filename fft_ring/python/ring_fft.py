"""ring_fft.py — cyclic convolution on the ring graph, accelerated and verified.

WHAT THIS REPLACES
------------------
`graph.diffuse` iterates ``x <- (1-eta) x + eta * (P @ x)`` once per step.  On a ring
P is circulant, so ``P @ x`` IS a cyclic convolution with the ring kernel.  The
current code pays O(N^2) per step; the same quantity can be had for O(N log N) with
three length-N FFTs.

WHY IT IS BUILT IN THREE LAYERS
-------------------------------
1. `convolve_cpu` — a numpy reference.  Always available, used as the oracle and as
   the fallback when no GPU / no compiled extension is present.
2. `convolve_gpu` — the CUDA kernel in csrc/ring_fft.cu, for N in {8,16,32}.  N=64
   is deliberately NOT exposed: the two-warp exchange it needs has not been
   validated on hardware (see the note in the .cu file).
3. `diffuse` — the drop-in for `graph.diffuse`, which picks a layer and reports
   which one it used, so a caller can never mistake a silent CPU fallback for a
   speedup.

The honest performance claim is NOT "FFT is faster at N=64" — on a scalar CPU it is
roughly 100x SLOWER, because 3 transforms of bookkeeping lose to 4096 FMAs.  The GPU
kernel only pays off because one warp owns one ring and the whole butterfly lives in
registers; measure before believing.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SUPPORTED_N",
    "GPU_N",
    "convolve_cpu",
    "convolve_gpu",
    "diffuse",
    "footprint",
    "backends",
]

# Sizes with a verified implementation on this machine.  128 is included because that
# is the ring app.py actually builds (128 agents); it is computed by the radix-2
# split below rather than by a length-128 kernel.
SUPPORTED_N = (8, 16, 32, 64, 128)
# Sizes whose CUDA path has actually been run and checked.  64 is excluded on
# purpose: its two-warp exchange is written but not validated.
GPU_N = (8, 16, 32)


def _ext():
    """The compiled CUDA extension, or None."""
    try:
        from ring_fft_ext import ring_conv  # type: ignore
        return ring_conv
    except Exception:
        return None


def backends() -> dict:
    """What is actually usable right now — reported, never guessed."""
    import torch

    ext = _ext()
    return {
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "extension": ext is not None,
        "gpu_sizes": list(GPU_N) if (ext and torch.cuda.is_available()) else [],
        "cpu_sizes": list(SUPPORTED_N),
    }


def convolve_cpu_split(a: np.ndarray, kernel: np.ndarray, dtype=np.complex128) -> np.ndarray:
    """Cyclic convolution of length 2M from four length-M convolutions.

    N=128 is exactly 2x64, and app.py's ring has 128 agents, so this is the size that
    matters.  Writing i = 2p+s and k = 2q+t, the cyclic shift (i-k) mod 2M has the
    same parity as (s-t), so the even and odd outputs decouple exactly:

        out[2p]   = (ae ⊛ ke)[p] + (roll(ao, 1) ⊛ ko)[p]
        out[2p+1] = (ae ⊛ ko)[p] + (ao ⊛ ke)[p]

    with ae/ao the even/odd samples of a and ke/ko the even/odd samples of the kernel.
    No twiddles appear — this is the parity decomposition, not the Cooley-Tukey
    frequency split — which is why it reuses the verified half-length kernels instead
    of introducing a new butterfly.  (An earlier version of this function applied
    twiddle factors the decomposition does not contain; the derivation above is what
    it was checked against, not intuition.)
    """
    a = np.asarray(a, dtype=dtype)
    k = np.asarray(kernel, dtype=dtype)
    n = a.shape[-1]
    if n % 2:
        raise ValueError("split needs an even length")
    m = n // 2
    if m not in (8, 16, 32, 64):
        raise ValueError(f"half length {m} has no verified kernel")

    ae, ao = a[..., 0::2], a[..., 1::2]
    ke, ko = k[0::2], k[1::2]
    ce = convolve_cpu(ae, ke, dtype)
    cs = convolve_cpu(np.roll(ao, 1, axis=-1), ko, dtype)
    c1 = convolve_cpu(ae, ko, dtype)
    c2 = convolve_cpu(ao, ke, dtype)

    out = np.empty_like(a)
    out[..., 0::2] = ce + cs
    out[..., 1::2] = c1 + c2
    return out


def convolve_radix(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Cyclic convolution via the radix-8 path — the one the CUDA kernel mirrors."""
    import ring_fft_radix as R

    a = np.asarray(a, dtype=np.complex128)
    k = np.asarray(kernel, dtype=np.complex128)
    if a.shape[-1] != k.shape[-1]:
        raise ValueError("signal and kernel must have the same length")
    single = a.ndim == 1
    A = np.atleast_2d(a)
    out = np.empty_like(A)
    for i, row in enumerate(A):
        prod = R.fft(row) * R.fft(k)
        out[i] = R.fft(prod, inverse=True)
    return out[0] if single else out


def convolve_cpu(a: np.ndarray, kernel: np.ndarray, dtype=np.complex128) -> np.ndarray:
    """Cyclic convolution via FFT (numpy).  `a` may be (N,) or (batch, N).

    Floating point at complex64 (`dtype=np.complex64`) matches the GPU kernel's
    precision and loses ~1e-7 per step; over a 200-step diffusion that is visible in
    the last digits and irrelevant to the dispersion curve.  The default here is
    complex128 because the CPU path has no reason to be fast — it is the oracle and
    the reference the GPU result is judged against.
    """
    a = np.asarray(a, dtype=dtype)
    k = np.asarray(kernel, dtype=dtype)
    n = a.shape[-1]
    if k.shape[-1] != n:
        raise ValueError(f"kernel length {k.shape[-1]} != signal length {n}")
    if n not in SUPPORTED_N:
        raise ValueError(f"support n in {SUPPORTED_N}, got {n}")
    if n == 128:
        return convolve_cpu_split(a, k, dtype)
    # n in {8,16,32,64}: the radix-8 path, which is what the CUDA kernel implements
    return convolve_radix(a, k)


def convolve_gpu(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Exactly `convolve_cpu` on the T4, or a raised error — never a silent fallback."""
    import torch

    ext = _ext()
    if ext is None:
        raise RuntimeError("CUDA extension not built (see fft_ring/build.sh)")
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    n = a.shape[-1]
    if n not in GPU_N:
        raise RuntimeError(f"GPU path is verified for n in {GPU_N}, not {n}")

    single = a.ndim == 1
    A = np.atleast_2d(a)
    batch = A.shape[0]
    ta = torch.as_tensor(np.stack([A.real, A.imag], -1).astype(np.float32), device="cuda")
    tk = torch.as_tensor(np.stack([kernel.real, kernel.imag], -1).astype(np.float32), device="cuda")
    tout = torch.empty_like(ta)
    ext(ta, tk, tout, int(n), int(batch))
    arr = tout.cpu().numpy()
    out = (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)
    return out[0] if single else out


def diffuse(w: np.ndarray, x0: np.ndarray, steps: int, eta: float = 0.5,
            prefer: str = "auto") -> tuple[np.ndarray, str]:
    """Drop-in for `graph.diffuse` that returns (per-step dispersion, backend used).

    The second return value is the point: a caller can assert it got "gpu" instead of
    discovering later that everything ran on the CPU fallback.
    """
    n = int(w.shape[0])
    if w.shape != (n, n):
        raise ValueError("w must be square")
    if n not in SUPPORTED_N:
        raise ValueError(
            f"ring size {n} is outside the verified range {SUPPORTED_N}. "
            "Block it into supported sizes or pad, but do not guess: the kernel's "
            "correctness is only established for these four."
        )

    deg = np.maximum(w.sum(1), 1e-30)
    kernel = (w / deg[:, None])[0]          # circulant ring: every row is the same

    def _cpu_state(x):
        return x

    use_gpu = False
    if prefer in ("auto", "gpu"):
        try:
            from ring_fft_ext import ring_conv  # noqa: F401
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
    if prefer == "gpu" and not use_gpu:
        raise RuntimeError("GPU requested but unavailable")

    backend = "gpu" if use_gpu else "cpu"

    # x0 is (N,) — one scalar per agent, which is exactly what app.py passes — or
    # (N, d) when each agent carries a d-vector.  The RING axis is axis 0 in both
    # cases, so a d-vector case is batched by transposing to (d, N); the earlier
    # version reshaped (N,) to (N, 1) and silently treated the feature axis as the
    # ring, which is a wrong-answer bug that only shows up as a shape error.
    x = np.array(x0, dtype=np.float64)
    vec = x.ndim == 2
    if vec:
        x = x.T.copy()                       # (d, N): one ring per row
    out = np.empty(steps + 1)
    # The GPU kernel is complex64; matching that here would cap the comparison at
    # 1e-7 for no benefit on the CPU, so only the GPU path is narrowed.
    cdtype = np.complex64 if use_gpu else np.complex128
    kc = kernel.astype(cdtype)

    for t in range(steps + 1):
        dev = x - x.mean(-1, keepdims=True)
        out[t] = float(np.mean(np.linalg.norm(dev, axis=0))) if vec else float(np.mean(np.abs(dev)))
        if t < steps:
            conv = (convolve_gpu(x.astype(np.complex64), kc) if use_gpu
                    else convolve_cpu(x, kc, dtype=cdtype))
            x = (1.0 - eta) * x + eta * conv.real
    return out, backend


def footprint(n: int, batch: int) -> dict:
    """Cache footprint of one transform, in bytes — the numbers the kernel is shaped by."""
    c = 8  # complex64
    return {
        "registers": c * n,
        "l1": c * n + c * 64,      # data + the shared 64-entry twiddle table
        "l2": c * n * (batch + 1),  # weights + every ring in the batch
        "global": c * n * (2 * batch + 1),
    }


def _selftest() -> None:
    rng = np.random.default_rng(0)
    # 1. FFT convolution == the naive definition, for every supported size
    for n in SUPPORTED_N:
        a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        w = (rng.standard_normal(n)).astype(np.complex64)
        got = convolve_cpu(a, w)
        want = np.array([sum(a[(i - k) % n] * w[k] for k in range(n)) for i in range(n)])
        assert np.max(np.abs(got - want)) / max(1.0, np.max(np.abs(want))) < 1e-4

    # 2. a delta kernel is the identity
    for n in SUPPORTED_N:
        d = np.zeros(n, np.complex64)
        d[0] = 1
        a = rng.standard_normal(n).astype(np.complex64)
        assert np.max(np.abs(convolve_cpu(a, d) - a)) < 1e-5

    # 3. diffuse() reproduces graph.diffuse's numbers exactly
    n = 32
    w = np.zeros((n, n))
    for i in range(n):
        w[i, (i + 1) % n] = w[(i + 1) % n, i] = 1.0
    x0 = rng.standard_normal(n)
    p = w / w.sum(1, keepdims=True)
    ref = np.empty(21)
    x = x0.copy()
    for t in range(21):
        ref[t] = np.abs(x - x.mean()).mean()
        if t < 20:
            x = 0.6 * x + 0.4 * (p @ x)
    got, backend = diffuse(w, x0, 20, eta=0.4, prefer="cpu")
    assert backend == "cpu", backend
    assert np.max(np.abs(got - ref)) < 1e-5, np.max(np.abs(got - ref))

    # 4. dispersion is non-increasing on a ring (a 1-hot start spreads out)
    onehot = np.zeros(n)
    onehot[0] = 1.0
    d, _ = diffuse(w, onehot, 40, eta=0.5, prefer="cpu")
    assert np.all(np.diff(d) <= 1e-9), d[:5]

    # 5. an unsupported size is refused loudly, not silently padded
    try:
        diffuse(np.eye(100), np.zeros(100), 3)
        raise AssertionError("size 100 should have been refused")
    except ValueError:
        pass

    print("ring_fft: all invariants hold ·", backends())


if __name__ == "__main__":
    _selftest()
