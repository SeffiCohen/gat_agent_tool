#!/usr/bin/env python3
"""verify_initial_prompt.py — static-asset checks on assets/initial_prompt.md.

Runs without GAT, without torch. Asserts:
  - File exists.
  - Contains all 14 canonical CBC features as ``lab_<name>_last`` lines.
  - Does NOT contain regression-banned strings (``0.000001``, ``epsilon``,
    ``divide-by-zero``, ``stabilization``) — guards against the dropped
    constraint sneaking back in.
  - Contains the verbatim "200 mathematical expressions" header.
  - Contains item 1 verbatim: "Use only +, -, *, / and parentheses."
  - Contains the ``{disease_name}`` placeholder exactly once.
  - Format-substitutes cleanly with a sample disease name.

Exit 0 on all-pass, 1 on first failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ASSET = Path(__file__).resolve().parent.parent / "assets" / "initial_prompt.md"

# The 14-feature canonical CBC panel — must all appear as `lab_<name>_last`.
_CBC_FEATURES = [
    "lab_BASO_pct_last",
    "lab_EOS_pct_last",
    "lab_HB_last",
    "lab_HCT_last",
    "lab_LYMpct_last",
    "lab_MCH_last",
    "lab_MCHC_last",
    "lab_MCV_last",
    "lab_MONOpct_last",
    "lab_NEUTpct_last",
    "lab_PLT_last",
    "lab_RBC_last",
    "lab_RDW_last",
    "lab_WBC_last",
]

# Strings that MUST NOT appear — regression guard against the user's original
# verbatim items 5 and 6 (epsilon / stabilization) sneaking back in. The GAT
# parser rejects all numeric literals; if any of these reappear in the prompt,
# the LLM will produce expressions that all fail validation.
_BANNED = [
    "0.000001",
    "epsilon",
    "divide-by-zero",
    "stabilization",
    "stabilized",
    # extra defensive:
    "1e-6",
    "1e-06",
]

# Verbatim phrases the prompt MUST contain.
_REQUIRED = [
    "200 mathematical expressions",
    "Use only +, -, *, / and parentheses.",
    "Use only the variables listed above; do not invent new variables.",
    "{disease_name}",
    "ONE line containing ONLY the expression",
    "Return a list of 200 expressions",
]


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> int:
    if not _ASSET.exists():
        _fail(f"asset file missing: {_ASSET}")

    body = _ASSET.read_text()

    # Check feature presence.
    for feat in _CBC_FEATURES:
        if feat not in body:
            _fail(f"missing CBC feature: {feat}")

    # Check banned strings.
    for banned in _BANNED:
        if banned in body:
            _fail(f"BANNED string present (should have been dropped per user "
                  f"directive on epsilon constraint): {banned!r}")

    # Check required phrases.
    for required in _REQUIRED:
        if required not in body:
            _fail(f"missing required phrase: {required!r}")

    # {disease_name} should appear exactly once (single substitution point).
    n_disease = body.count("{disease_name}")
    if n_disease != 1:
        _fail(f"expected 1 {{disease_name}} placeholder, found {n_disease}")

    # No {history_block} in the initial prompt — that's iter 2+ only.
    if "{history_block}" in body:
        _fail("initial_prompt.md must NOT contain {history_block} — that "
              "placeholder belongs in iteration_prompt.md (iter 2+)")

    # Format-substitute test.
    try:
        rendered = body.format(disease_name="RheumatoidArthritis")
    except (KeyError, IndexError) as exc:
        _fail(f"format substitution failed (extra unescaped braces?): {exc}")
    if "{" in rendered or "}" in rendered:
        # If after substitution any literal { or } remains, the asset has
        # an unescaped brace that isn't a real placeholder — corrupts a
        # downstream `.format()` call.
        # (Allow them in code-fence-like contexts? No — there are no fences here.)
        leftovers = [line for line in rendered.splitlines()
                     if "{" in line or "}" in line]
        _fail(f"unescaped braces after format(): {leftovers[:3]}")
    if "RheumatoidArthritis" not in rendered:
        _fail("format() did not substitute {disease_name}")

    print("PASS verify_initial_prompt: all 14 CBC features present, no banned "
          "strings, all required phrases present, {disease_name} substitutes cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
