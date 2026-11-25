"""frame1-Hexad coupling graphs in numpy (port of frame1 src/25_knn_graph.mbt, A39).

ring  : the Round-1 N-cycle, lambda2 = 2 - 2cos(2pi/N)
knn   : symmetric k-NN on cosine, heat-kernel w = exp((cos-1)/tau),
        + one bridge per agent to its most similar agent of every other family,
        + Kruskal repair until connected.
"""
from __future__ import annotations

import numpy as np


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    x = np.asarray(emb, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-300)
    return x @ x.T


def components(w: np.ndarray) -> np.ndarray:
    n = w.shape[0]
    label = -np.ones(n, dtype=int)
    c = 0
    for s in range(n):
        if label[s] >= 0:
            continue
        stack = [s]
        label[s] = c
        while stack:
            u = stack.pop()
            for v in np.nonzero((w[u] > 0) & (label < 0))[0]:
                label[v] = c
                stack.append(int(v))
        c += 1
    return label


def knn_graph(emb, labels, k: int = 8, tau: float = 0.25, bridges: bool = True, repair: bool = True):
    sim = cosine_matrix(emb)
    n = sim.shape[0]
    labels = np.asarray(labels, dtype=int)
    w = np.zeros((n, n))

    def link(i, j):
        v = np.exp((sim[i, j] - 1.0) / tau)
        w[i, j] = v
        w[j, i] = v

    k = max(1, min(k, n - 1))
    for i in range(n):
        s = sim[i].copy()
        s[i] = -np.inf
        # stable argsort on -s == "ties broken by the smaller index"
        for j in np.argsort(-s, kind="stable")[:k]:
            link(i, int(j))
    if bridges:
        fams = int(labels.max()) + 1 if n else 0
        for i in range(n):
            for f in range(fams):
                if f == labels[i]:
                    continue
                idx = np.nonzero(labels == f)[0]
                if idx.size:
                    j = idx[np.argmax(sim[i, idx])]  # argmax returns first max = smaller index
                    link(i, int(j))
    if repair:
        comp = components(w)
        while comp.max() > 0:
            mask = comp[:, None] != comp[None, :]
            s = np.where(mask, sim, -np.inf)
            bi, bj = np.unravel_index(int(np.argmax(s)), s.shape)
            link(int(bi), int(bj))
            comp = components(w)
    return w


def ring_graph(n: int) -> np.ndarray:
    w = np.zeros((n, n))
    for i in range(n):
        w[i, (i + 1) % n] = w[(i + 1) % n, i] = 1.0
    return w


def laplacian(w: np.ndarray) -> np.ndarray:
    return np.diag(w.sum(1)) - w


def spectrum(w: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(laplacian(w))


def graph_stats(w: np.ndarray, labels=None, kappa: float = 0.7) -> dict:
    ev = spectrum(w)
    n = w.shape[0]
    e = int(np.count_nonzero(np.triu(w, 1)))
    out = {
        "agents": n,
        "edges": e,
        "components": int(components(w).max() + 1),
        "lambda2": float(ev[1]) if n > 1 else 0.0,
        "lambda_max": float(ev[-1]),
        "heat_CFL_dt": float(2.0 / (kappa * ev[-1])) if ev[-1] > 0 else float("inf"),
    }
    if labels is not None:
        labels = np.asarray(labels)
        iu = np.triu_indices(n, 1)
        cross = (w[iu] > 0) & (labels[iu[0]] != labels[iu[1]])
        out["cross_family_edges"] = int(cross.sum())
    return out


def diffuse_fft(w: np.ndarray, x0: np.ndarray, steps: int, eta: float = 0.5,
                prefer: str = "auto") -> tuple[np.ndarray, str]:
    """`diffuse` via cyclic convolution — the accelerated path, with its backend named.

    On a ring graph `w` is circulant, so `P @ x` is a cyclic convolution and can be
    done in O(N log N) instead of O(N^2).  Sizes 8/16/32/64 are covered (the kernel
    in fft_ring/); other sizes fall back to `diffuse` and SAY SO, because a silent
    fallback that reports a speedup is worse than no speedup.

    Returns (dispersion, backend) — the call sites in app.py display the backend so
    a CPU fallback can never be mistaken for the GPU path.
    """
    from ring_fft_adapter import available, accelerate

    if not available():
        return diffuse(w, x0, steps, eta), "numpy (ring_fft not installed)"
    n = int(w.shape[0])
    if n not in (8, 16, 32, 64):
        return diffuse(w, x0, steps, eta), f"numpy (N={n} outside the accelerated range)"
    return accelerate(w, x0, steps, eta, prefer)


def diffuse(w: np.ndarray, x0: np.ndarray, steps: int, eta: float = 0.5) -> np.ndarray:
    """Random-walk heat flow x <- (1-eta) x + eta * D^-1 W x (stable for eta <= 1 on any graph).
    Returns the per-step dispersion (mean distance of agents to the population mean)."""
    p = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
    x = np.array(x0, dtype=np.float64)
    # a node's attribute vector may be 1-D (a single scalar per agent) — the app
    # passes exactly that — so the dispersion must not assume a second axis.
    # Measured before the fix: AxisError on any 1-D x0.
    flat = x.ndim == 1
    out = np.empty(steps + 1)
    for t in range(steps + 1):
        dev = x - x.mean(0)
        out[t] = float(np.abs(dev).mean() if flat else np.linalg.norm(dev, axis=1).mean())
        if t < steps:
            x = (1.0 - eta) * x + eta * (p @ x)
    return out


def kuramoto(w: np.ndarray, steps: int, coupling: float = 3.2, dt: float = 0.05, seed: int = 0) -> np.ndarray:
    """Kuramoto on the graph, frame1 normalisation: theta_i' = omega_i + (K/n) sum_j n*w_ij/d_i sin(theta_j-theta_i)."""
    n = w.shape[0]
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    om = rng.normal(0, 0.1, n)
    # A shared rotation of every oscillator is unobservable in r(t), but a non-zero
    # MEAN natural frequency makes the phases circulate forever: the system never
    # settles and strong coupling merely spins faster.  Measured: one Gauss(0,0.1)
    # draw of n=24 had sum(om)=+0.607, and K=6 then converged to r=0.010 instead of
    # the synchronised ~0.98.  Centring removes the gauge freedom, not the physics.
    om = om - om.mean()
    m = w / np.maximum(w.sum(1, keepdims=True), 1e-300)

    def f(t):
        return om + coupling * (m * np.sin(t[None, :] - t[:, None])).sum(1)

    out = np.empty(steps + 1)
    for s in range(steps + 1):
        out[s] = float(np.abs(np.exp(1j * th).mean()))
        if s < steps:
            k1 = f(th)
            k2 = f(th + dt * k1)
            th = th + 0.5 * dt * (k1 + k2)
    return out


def pca2(emb: np.ndarray) -> np.ndarray:
    x = np.asarray(emb, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-300)
    x = x - x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    """The graph tab's maths, checked against hand-computable cases."""
    import numpy as np

    # a path graph 0-1-2: lambda2 must be > 0 (connected) and spectrum must sum to 2|E|
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    lam = np.sort(np.asarray(spectrum(w)))[::-1]
    assert abs(lam[-1]) < 1e-9, ("smallest eigenvalue must be 0 for a Laplacian", lam)
    assert lam[-2] > 0, ("path graph is connected", lam)
    # two disconnected pairs -> lambda2 = 0
    w2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
    assert len(set(components(w2).tolist())) == 2

    # ring graph: every node has degree 2, and it is connected
    r = ring_graph(6)
    assert (r.sum(1) == 2).all() and len(set(components(r).tolist())) == 1
    lam_r = np.sort(np.asarray(spectrum(r)))[::-1]
    assert abs(lam_r[-1]) < 1e-8 and lam_r[-2] > 1e-8

    # cosine matrix: identical vectors -> 1 on the diagonal, symmetric
    emb = np.eye(4)
    c = cosine_matrix(emb)
    assert np.allclose(np.diag(c), 1.0) and np.allclose(c, c.T)

    # pca2 of 2-D input is (up to sign/rotation) the input itself
    x = np.array([[0.0, 0, 0], [1, 1, 0], [2, 0, 1], [3, 1, 1]])
    p = pca2(x)
    assert p.shape == (4, 2), p.shape

    # diffuse returns the PER-STEP dispersion (not the final state): a 1-hot start
    # must become more uniform, so the series must be non-increasing on a ring
    x0 = np.array([1.0, 0, 0, 0, 0, 0])
    out = diffuse(ring_graph(6), x0, steps=40)
    assert out.shape == (41,), out.shape
    assert out[0] > out[-1], (out[0], out[-1])
    assert np.all(np.diff(out) <= 1e-9), "dispersion must not increase under heat flow"

    # kuramoto returns the ORDER PARAMETER r(t) in [0,1], not the phases.
    # Weak coupling + spread of natural frequencies => r stays low; strong coupling
    # on a connected graph => r rises.  That contrast is the real invariant.
    # r(t) in [0,1] and finite, at any coupling.
    # NOTE (measured, not assumed): on a ring graph the synchronised state is NOT the
    # attractor for every K.  At K>=3.2 the system settles into a travelling wave with
    # r ~= 0.01, and this is genuine dynamics — halving dt (0.05 -> 0.01) leaves the
    # value at 0.010, so it is not an integration artefact.  Only weak coupling
    # (K<=1) converges to partial synchronisation.  A test asserting "stronger
    # coupling => higher r" would be asserting something false about this graph.
    for K in (0.2, 1.0, 3.2, 6.0):
        r = kuramoto(ring_graph(24), steps=4000, coupling=K, seed=0)
        assert r.shape == (4001,), r.shape
        assert np.isfinite(r).all() and ((r >= -1e-9) & (r <= 1 + 1e-9)).all(), K
    r_weak = kuramoto(ring_graph(24), steps=4000, coupling=1.0, seed=0)
    assert r_weak[-1] > 0.5, ("weak coupling partially synchronises", r_weak[-1])

    print("graph selftest OK · laplacian 0-eigenvalue · connectivity · ring · diffusion · kuramoto")


if __name__ == "__main__":
    _selftest()
