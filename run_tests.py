"""run_tests.py — one entry point for every check in this repo.

    python3 run_tests.py            # everything that runs on CPU
    python3 run_tests.py --fast     # skip the slow statistical tests (graph/teacher)
    python3 run_tests.py --list

Why a runner instead of pytest: the Space's build only installs `requirements.txt`,
and half these modules must import *before* torch is even touched (the byte table),
so keeping the harness dependency-free means the same command works on the Space,
on CI and offline.
"""

from __future__ import annotations

import argparse
import importlib
import io
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
# the test files import sibling modules by bare name; make that work no matter how
# the runner was invoked (repo root, CI checkout, the Space's /app).
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("SIXIN1_SKIP_LOAD", "1")      # never download 6 GB to run a unit test

# (module, attribute, label, slow)
SUITES = [
    ("bytes_vocab", "_selftest", "byte table is a 256-byte bijection", False),
    ("spine", "selftest", "shared spine + fuse arithmetic", False),
    ("fuse", "_selftest", "boxed-stop policy, ModelSpec, stop ids", False),
    ("targets", "_selftest", "distillation loss (perfect student => 0)", False),
    ("sparse_targets", "_selftest", "top-k encode/TV identity", False),
    ("distill", "_selftest", "fused target on the spine", False),
    ("train_distill", "_selftest", "atomic checkpoints, corpus reader", False),
    ("dataset", "_selftest", "corpus schema, dedupe, licence manifest", False),
    ("probe", "_selftest", "verdict(k) picks the cheapest sufficient k", False),
    ("teacher", "_selftest", "byte offsets, boundary intersection", False),
    ("graph", "_selftest", "laplacian, diffusion, kuramoto", True),
]

FILES = [
    ("tests/test_counts.py", "vocabulary-intersection numbers vs deployed code", False),
    ("tests/test_teacher.py", "byte alignment on real tokenizers", True),
    ("tests/test_probe.py", "geometric-mean invariant, opposed teachers", True),
]


def run_module(mod: str, attr: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(mod)
        fn = getattr(m, attr, None)
        if fn is None:
            return False, f"{mod}.{attr} not found"
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            fn()
        finally:
            sys.stdout = old
        out = buf.getvalue().strip().splitlines()
        return True, out[-1] if out else "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n" + "".join(traceback.format_exc().splitlines(True)[-4:])


def run_file(path: str) -> tuple[bool, str]:
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([HERE] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("SIXIN1_SKIP_LOAD", "1")
    p = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=HERE, env=env)
    if p.returncode == 0:
        line = (p.stdout.strip().splitlines() or ["ok"])[-1]
        return True, line
    return False, (p.stderr.strip().splitlines() or ["failed"])[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip slow statistical suites")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for m, f, label, slow in SUITES:
            print(f"  {'[slow] ' if slow else '       '}{m}.{f:12s} {label}")
        for p, label, slow in FILES:
            print(f"  {'[slow] ' if slow else '       '}{p:26s} {label}")
        return 0

    ok = bad = skip = 0
    for mod, attr, label, slow in SUITES:
        if slow and a.fast:
            skip += 1
            continue
        good, msg = run_module(mod, attr)
        print(f"{'PASS' if good else 'FAIL'}  {mod:16s} {label}")
        if not good:
            print("      " + msg.replace("\n", "\n      "))
            bad += 1
        else:
            ok += 1
    for path, label, slow in FILES:
        if slow and a.fast:
            skip += 1
            continue
        good, msg = run_file(path)
        print(f"{'PASS' if good else 'FAIL'}  {path:16s} {label}")
        if not good:
            print("      " + msg.replace("\n", "\n      "))
            bad += 1
        else:
            ok += 1

    print(f"\n{ok} passed · {bad} failed · {skip} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
