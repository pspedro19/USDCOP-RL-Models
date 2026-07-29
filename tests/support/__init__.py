"""Shared helpers for the test suite (no tests live here — pytest collects `test_*.py`).

`tests/fixtures/` holds shared DATA; `tests/support/` holds shared CODE. A primitive that
two locks depend on lives here exactly once: two copies of the same measuring tape means
one of them is lying (K-035), and the copy nobody re-reads is the one that rots.
"""
