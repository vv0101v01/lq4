#!/usr/bin/env python3
"""selftest.py — run every check that does not need a GPU.  Exit code is the verdict."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ring_fft  # noqa: E402

if __name__ == "__main__":
    ring_fft._selftest()
    print("\nnow run ./build.sh for the C++ checks, and bench/gpu_bench.py on the T4.")
