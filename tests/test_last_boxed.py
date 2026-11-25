"""test_last_boxed.py — the answer extractor the pass@1 scorer depends on.

`last_boxed` was duplicated in app.py and dataset.py; dataset.py is superseded and
its copy is in attic/, so this file pins the behaviour of the one that ships.
"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, "/tmp/stubs")


def _app():
    os.environ["SIXIN1_SKIP_LOAD"] = "1"
    spec = importlib.util.spec_from_file_location("app_under_test_lb", os.path.join(ROOT, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def last_boxed(text):
    from pathlib import Path

    src = Path(os.path.join(ROOT, "app.py")).read_text()
    start = src.index("def last_boxed(")
    end = src.index("\n\n\n", start)
    ns = {"re": __import__("re")}
    exec(src[start:end], ns)  # noqa: S102 - extracting one pure function, no side effects
    return ns["last_boxed"](text)


def test_simple():
    assert last_boxed(r"the answer is \boxed{42}") == "42"


def test_takes_the_last_one():
    assert last_boxed(r"\boxed{1} then \boxed{7}") == "7"


def test_nested_braces_balance():
    assert last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"


def test_unbalanced_is_none_not_a_crash():
    assert last_boxed(r"\boxed{ unfinished") is None


def test_no_boxed():
    assert last_boxed("no answer here") is None
    assert last_boxed("") is None
    assert last_boxed(None) is None


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
    print(f"{'FAILED' if fails else 'passed'}")
    raise SystemExit(1 if fails else 0)
