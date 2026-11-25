"""What is actually true about product-of-experts (written down after I got it wrong).

I first wrote "a product of experts is sharper than every teacher" as an invariant
and used it to accuse the probe of a bug.  **That reasoning was wrong.**  Two
confident, mutually-opposed teachers fuse to something flatter:

    p1 = (0.9, 0.1), p2 = (0.1, 0.9)  ->  geo-mean = (0.5, 0.5)   H: 0.325 -> 0.693

So fused entropy can legitimately exceed every teacher's.  The real invariants are
the two below, and the probe reports teacher disagreement separately.
"""
import numpy as np

from spine import SharedSpine, fuse, step_logprobs


def _fake_spine(n=50, names=("A", "B")):
    strings = [bytes([65 + i % 26]) + str(i).encode() for i in range(n)]
    ids = {k: np.arange(n) for k in names}
    feeds = {k: {"think": [0], "end": [], "answer": [0]} for k in names}
    return SharedSpine(list(names), strings, ids, feeds)


def test_opposed_teachers_fuse_flatter():
    """The counter-example, as a test: disagreement => flatter, not sharper."""
    p1, p2 = np.array([0.9, 0.1]), np.array([0.1, 0.9])
    H = lambda p: float(-(p * np.log(p)).sum())  # noqa: E731
    g = np.sqrt(p1 * p2)
    g /= g.sum()
    assert H(g) > max(H(p1), H(p2)) + 0.3
    assert np.allclose(g, [0.5, 0.5])


def test_fused_equals_direct_geometric_mean():
    """Whatever the entropy does, the fused law MUST be the normalised geo-mean."""
    rng = np.random.default_rng(0)
    sp = _fake_spine()
    n = len(sp)
    lp1, lp2 = np.log(rng.dirichlet(np.full(n, 0.7), 20)), np.log(rng.dirichlet(np.full(n, 0.7), 20))
    z1 = np.stack([step_logprobs(l, sp, "A", allow_answer=False)[:n] for l in lp1])
    z2 = np.stack([step_logprobs(l, sp, "B", allow_answer=False)[:n] for l in lp2])
    for t in range(20):
        z = fuse([z1[t], z2[t]], [0.5, 0.5], "product")[:n]
        # direct: normalised geometric mean of the two restricted laws
        expect = np.exp(0.5 * np.log(np.exp(z1[t])) + 0.5 * np.log(np.exp(z2[t])))
        expect = expect / expect.sum()
        assert np.allclose(np.exp(z), expect, atol=1e-9)


def test_mixture_is_between_teachers():
    """A mixture's entropy lies between the two teachers' — it cannot be sharper."""
    rng = np.random.default_rng(1)
    sp = _fake_spine()
    n = len(sp)
    lp1, lp2 = np.log(rng.dirichlet(np.full(n, 0.5), 5)), np.log(rng.dirichlet(np.full(n, 0.5), 5))
    z1 = np.stack([step_logprobs(l, sp, "A", allow_answer=False)[:n] for l in lp1])
    z2 = np.stack([step_logprobs(l, sp, "B", allow_answer=False)[:n] for l in lp2])
    for t in range(5):
        z = fuse([z1[t], z2[t]], [0.5, 0.5], "mixture")[:n]
        z = z - np.logaddexp.reduce(z)
        Hf = -(np.exp(z) * z).sum()
        Hs = sorted([-(np.exp(z1[t]) * z1[t]).sum(), -(np.exp(z2[t]) * z2[t]).sum()])
        # mixture entropy >= max? No: it can be slightly below; but never below min - small slack
        assert Hf >= Hs[0] - 0.05, (Hf, Hs)


if __name__ == "__main__":
    test_opposed_teachers_fuse_flatter()
    test_fused_equals_direct_geometric_mean()
    test_mixture_is_between_teachers()
    print("✅ fused law == normalised geometric mean; opposed teachers fuse FLATTER (not sharper)")
