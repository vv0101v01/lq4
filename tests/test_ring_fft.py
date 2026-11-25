"""test_ring_fft.py — the accelerated ring diffusion must not change any number.

`graph.diffuse_fft` routes the ring through a cyclic-convolution kernel.  A wrong
convolution would not crash — it would produce a plausible-looking dispersion curve
and a wrong λ₂ story in the UI.  These tests pin equality against the live
`graph.diffuse` at machine precision for every size the kernel claims.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import graph  # noqa: E402
import ring_fft_adapter as A  # noqa: E402

SIZES = (8, 16, 32, 64, 128)


def test_available():
    assert A.available(), "ring_fft must be importable for the accelerated path to exist"


def test_matches_diffuse_exactly():
    if not A.available():
        return
    rng = np.random.default_rng(7)
    for n in SIZES:
        w = graph.ring_graph(n)
        x0 = rng.standard_normal(n)
        ref = graph.diffuse(w, x0, 120)
        got, backend = graph.diffuse_fft(w, x0, 120, prefer="cpu")
        err = float(np.max(np.abs(got - ref)))
        assert err < 1e-10, (n, err, backend)


def test_matches_diffuse_2d_features():
    if not A.available():
        return
    rng = np.random.default_rng(8)
    for n in (32, 128):
        w = graph.ring_graph(n)
        x0 = rng.standard_normal((n, 2))
        ref = graph.diffuse(w, x0, 80)
        got, _ = graph.diffuse_fft(w, x0, 80, prefer="cpu")
        assert float(np.max(np.abs(got - ref))) < 1e-10


def test_backend_is_named():
    """A caller must be able to tell the accelerated path from the fallback."""
    _, backend = graph.diffuse_fft(graph.ring_graph(32), np.zeros(32), 3, prefer="cpu")
    assert isinstance(backend, str) and backend
    assert "numpy" not in backend


def test_unsupported_size_falls_back_with_a_reason():
    """97 agents has no verified kernel; the fallback must say WHY, not pretend."""
    w = graph.ring_graph(97)
    ref = graph.diffuse(w, np.ones(97), 5)
    got, backend = graph.diffuse_fft(w, np.ones(97), 5)
    assert np.allclose(got, ref)
    assert "outside" in backend.lower() or "numpy" in backend.lower(), backend


def test_gpu_refuses_rather_than_silently_falling_back():
    """Asking for GPU without a CUDA path must raise, never return CPU numbers."""
    if not A.available():
        return
    import ring_fft

    rep = ring_fft.backends()
    if rep["cuda"] and rep["extension"]:
        return  # on real hardware this path is exercised by bench/gpu_bench.py
    try:
        ring_fft.convolve_gpu(np.zeros(16, np.complex64), np.ones(16, np.complex64))
        raise AssertionError("convolve_gpu should have refused")
    except RuntimeError:
        pass


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL {name}: {e}")
    print("failed" if fails else "passed")
    raise SystemExit(1 if fails else 0)
