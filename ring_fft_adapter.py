"""ring_fft_adapter.py — the one place 6in1 talks to the FFT ring convolver.

Kept as a thin adapter so `graph.py` keeps its zero-hard-dependency character: if the
`fft_ring` package is not on the path (the Space does not install it), everything
falls back to the original numpy loop and says so.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG = os.environ.get("RING_FFT_PATH", os.path.join(os.path.dirname(__file__), "..", "fft_ring", "python"))


def _mod():
    if _PKG not in sys.path:
        sys.path.insert(0, os.path.abspath(_PKG))
    try:
        import ring_fft  # noqa: F401
        return ring_fft
    except Exception:
        return None


def available() -> bool:
    return _mod() is not None


def accelerate(w: np.ndarray, x0: np.ndarray, steps: int, eta: float = 0.5,
               prefer: str = "auto") -> tuple[np.ndarray, str]:
    """`(dispersion, backend)` — raises rather than quietly mis-reporting."""
    m = _mod()
    if m is None:
        raise RuntimeError("ring_fft unavailable")
    return m.diffuse(w, x0, steps, eta=eta, prefer=prefer)


def backend_report() -> dict:
    """What the UI shows in the Diagnostics tab: measured, not assumed."""
    m = _mod()
    if m is None:
        return {"available": False}
    out = {"available": True}
    out.update(m.backends())
    return out
