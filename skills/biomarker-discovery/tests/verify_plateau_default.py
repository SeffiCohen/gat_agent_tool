#!/usr/bin/env python3
"""verify_plateau_default.py — confirm the documented plateau_patience default is 5.

Reads ``references/iteration_loop.md`` and ``SKILL.md``, asserts:
  - The current default ``plateau_patience`` is documented as ``5``.
  - The old default ``3`` is no longer documented as the active default.
  - ``pop_per_iter`` default is ``200`` (not ``10``).

This is a quick text-grep style guard against the user's spec
(early stopping = 5 iterations, pop_per_iter = 200) silently regressing
back to the old defaults during a refactor.

Exit 0 on all-pass, 1 on first failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent.parent
_ITERATION_LOOP = _SKILL_DIR / "references" / "iteration_loop.md"
_SKILL_MD = _SKILL_DIR / "SKILL.md"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _check_iteration_loop() -> None:
    if not _ITERATION_LOOP.exists():
        _fail(f"iteration_loop.md missing at {_ITERATION_LOOP}")
    body = _ITERATION_LOOP.read_text()

    # Affirmative check: "default `5`" must appear somewhere with plateau_patience.
    # Look for the phrase like "plateau_patience` (default `5`)" or
    # "plateau_patience` ... default `5`".
    if not re.search(
        r"plateau_patience[^.]{0,80}default\s*[`'\"]?5[`'\"]?",
        body, re.IGNORECASE,
    ):
        _fail("iteration_loop.md should document plateau_patience default as 5; "
              "the regex `plateau_patience.*default.*5` did not match")

    # Negative check: must NOT document "default 3" for plateau_patience.
    if re.search(
        r"plateau_patience[^.]{0,80}default\s*[`'\"]?3[`'\"]?",
        body, re.IGNORECASE,
    ):
        _fail("iteration_loop.md still documents plateau_patience default as 3 "
              "— REGRESSION (should be 5 per user spec).")

    # pop_per_iter should be 200, not 10.
    if not re.search(
        r"pop_per_iter[^.]{0,80}default\s*[`'\"]?200[`'\"]?",
        body, re.IGNORECASE,
    ):
        _fail("iteration_loop.md should document pop_per_iter default as 200")
    if re.search(
        r"pop_per_iter[^.]{0,80}default\s*[`'\"]?10[`'\"]?[\s.,)]",
        body, re.IGNORECASE,
    ):
        _fail("iteration_loop.md still documents pop_per_iter default as 10 "
              "— REGRESSION (should be 200 per user spec).")


def _check_skill_md() -> None:
    if not _SKILL_MD.exists():
        _fail(f"SKILL.md missing at {_SKILL_MD}")
    body = _SKILL_MD.read_text()

    # Walk SKILL.md line-by-line so we don't have to span markdown table
    # delimiters with regex. For each table row mentioning --plateau-patience
    # or --pop-per-iter, check that the same line documents the new default.
    plateau_seen = False
    pop_seen = False
    for line in body.splitlines():
        if "--plateau-patience" in line:
            plateau_seen = True
            if "default 5" not in line:
                _fail(
                    f"SKILL.md row for --plateau-patience does not say "
                    f"'default 5'. Line: {line.strip()!r}"
                )
            if "default 3" in line:
                _fail(
                    f"SKILL.md row for --plateau-patience still says "
                    f"'default 3' — REGRESSION. Line: {line.strip()!r}"
                )
        if "--pop-per-iter" in line:
            pop_seen = True
            if "default 200" not in line:
                _fail(
                    f"SKILL.md row for --pop-per-iter does not say "
                    f"'default 200'. Line: {line.strip()!r}"
                )
            if re.search(r"default\s+10[^0]", line):
                _fail(
                    f"SKILL.md row for --pop-per-iter still says 'default 10' "
                    f"— REGRESSION. Line: {line.strip()!r}"
                )
    if not plateau_seen:
        _fail("SKILL.md does not mention --plateau-patience anywhere")
    if not pop_seen:
        _fail("SKILL.md does not mention --pop-per-iter anywhere")


def main() -> int:
    _check_iteration_loop()
    _check_skill_md()
    print("PASS verify_plateau_default: plateau_patience=5 and "
          "pop_per_iter=200 are documented as defaults in both "
          "iteration_loop.md and SKILL.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
