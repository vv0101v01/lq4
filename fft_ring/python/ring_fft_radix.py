"""ring_fft_radix.py — mixed-radix FFT, radix 8 / 4 / 2.

The point of the radix-8 butterfly: eight points live in ONE thread's registers, so a
length-64 transform needs only 8 threads = 8 lanes of a single warp. The radix-2
version needed 64 lanes (two warps + a shared-memory exchange).

DIT decomposition, N = R*M:
    n = R*r + t          t = 0..R-1,  r = 0..M-1
    X[q*M + p] = sum_t  W_N^(t*p) * W_R^(t*q) * Y_t[p]
where Y_t = DFT_M of the t-th decimated subsequence. The radix-4 butterfly is the
x2+x4 form: two radix-2 stages with the +-j multiply folded in.
"""
from __future__ import annotations

import numpy as np

RADICES = (8, 4, 2)


def _factors(n: int) -> list[int]:
    """Factor n into stages using the largest radix that divides exactly."""
    out = []
    while n > 1:
        for r in RADICES:
            if n % r == 0:
                out.append(r)
                n //= r
                break
        else:
            raise ValueError(f"{n} has no factor in {RADICES}")
    return out


def _dft(v: np.ndarray) -> np.ndarray:
    """Direct DFT of a small array — the oracle the butterflies are checked against."""
    n = len(v)
    W = np.exp(-2j * np.pi * np.arange(n) / n)
    return np.array([np.sum(v[r] * W[(r * k) % n] for r in range(n)) for k in range(n)])


def _butterfly2(a, b, tw):
    t = b * tw
    return a + t, a - t


def _butterfly4(v, tw, inverse=False):
    """Radix-4: two radix-2 stages, the second folded with +-j.

    v: 4 complex values. tw: 4 twiddles W_N^(j*stride) for this stage.
    """
    a0, a1, a2, a3 = v[0] * tw[0], v[1] * tw[1], v[2] * tw[2], v[3] * tw[3]
    b0, b1 = a0 + a2, a0 - a2
    b2, b3 = a1 + a3, (a1 - a3) * (1j if inverse else -1j)
    return np.array([b0 + b2, b1 + b3, b0 - b2, b1 - b3])


def _butterfly8(v, tw, inverse=False):
    """Radix-8: 8 inputs -> 8 outputs, built from radix-4 + radix-2.

    8 = 2 * 4: split even/odd, two radix-4 transforms, then combine with W_8^k.
    """
    v = np.asarray(v, dtype=np.complex128) * tw
    even = _butterfly4(np.array([v[0], v[2], v[4], v[6]]), np.ones(4, complex), inverse)
    odd = _butterfly4(np.array([v[1], v[3], v[5], v[7]]), np.ones(4, complex), inverse)
    s = 1j if inverse else -1j
    w = np.array([1.0, np.exp(s * 2 * np.pi * 1 / 8), np.exp(s * 2 * np.pi * 2 / 8),
                  np.exp(s * 2 * np.pi * 3 / 8)])
    out = np.empty(8, dtype=np.complex128)
    for k in range(4):
        t = w[k] * odd[k]
        out[k] = even[k] + t
        out[k + 4] = even[k] - t
    return out


BUTTERFLY = {2: None, 4: _butterfly4, 8: _butterfly8}


def fft(x, inverse: bool = False) -> np.ndarray:
    """Mixed-radix DIT.  Handles n in {8,16,32,64} and anything 2/4/8-smooth."""
    x = np.asarray(x, dtype=np.complex128).copy()
    n = len(x)
    stages = _factors(n)
    # The sign of every twiddle (including the -1j inside the radix-4) is driven by
    # `inverse`, so conjugating the input as well would apply the reverse transform
    # twice.  One mechanism only.
    _fft_recursive(x, len(x), stages, n, inverse)
    if inverse:
        x = x / n
    return x


def _fft_recursive(x, size, stages, n0, inverse):
    """x is a strided slice view; `size` its length; `n0` the full transform length."""
    if size == 1:
        return
    r = stages[0]
    m = size // r
    # gather the r decimated subsequences
    sub = [x[t::r].copy() for t in range(r)]
    for t in range(r):
        _fft_recursive(sub[t], m, stages[1:], n0, inverse)
    # combine with twiddles and the r-point butterfly
    sgn = 1j if inverse else -1j
    for p in range(m):
        tw = np.array([np.exp(sgn * 2 * np.pi * t * p / size) for t in range(r)])
        col = np.array([sub[t][p] for t in range(r)])
        if r == 2:
            a, b = _butterfly2(col[0], col[1], tw[1])
            res = np.array([a, b])
        else:
            res = BUTTERFLY[r](col, tw, inverse)
        for q in range(r):
            x[q * m + p] = res[q]


def fft_warp64(x):
    """Length-64 transform with EXACTLY the thread mapping the CUDA kernel uses.

    lane t (t = 0..7) owns the 8 decimated samples x[t::8], runs an 8-point DFT in
    registers, then the lanes exchange through shuffles for the final radix-8 stage.
    This file is the executable definition of that mapping — if the kernel disagrees,
    the kernel is wrong.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    if n != 64:
        raise ValueError("warp64 is the 8-lane mapping for n=64")
    # step 1: each lane's private 8-point DFT
    Y = np.empty((8, 8), dtype=np.complex128)   # Y[t] = DFT8 of lane t's samples
    for t in range(8):
        Y[t] = _butterfly8(x[t::8], np.ones(8, complex))
    # step 2: cross-lane radix-8 combine — this is the shuffle step
    out = np.empty(64, dtype=np.complex128)
    for p in range(8):
        tw = np.array([np.exp(-2j * np.pi * t * p / 64) for t in range(8)])
        col = Y[:, p] * tw
        res = _butterfly8(col, np.ones(8, complex))
        for q in range(8):
            out[q * 8 + p] = res[q]
    return out


def _selftest():
    rng = np.random.default_rng(0)
    for n in (8, 16, 32, 64):
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        got = fft(x)
        want = np.fft.fft(x)
        e = np.max(np.abs(got - want)) / max(1.0, np.max(np.abs(want)))
        assert e < 1e-10, (n, e)
        back = fft(got, inverse=True)
        assert np.max(np.abs(back - x)) < 1e-10, n
    x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    e = np.max(np.abs(fft_warp64(x) - np.fft.fft(x)))
    assert e < 1e-10, e
    print("radix 8/4/2 · N in {8,16,32,64} == np.fft · warp64 (8-lane) == np.fft")


if __name__ == "__main__":
    _selftest()
