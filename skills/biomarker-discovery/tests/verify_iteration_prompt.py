#!/usr/bin/env python3
"""verify_iteration_prompt.py — static-asset checks on assets/iteration_prompt.md.

Same DSL guards as verify_initial_prompt.py, plus:
  - Must contain ``{history_block}`` placeholder (iter 2+ feedback channel).
  - Must contain "Propose 200 NEW expressions".
  - Must contain "{disease_name}".
  - Format-substitutes cleanly with sample values.

Exit 0 on all-pass, 1 on first failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ASSET = Path(__file__).resolve().parent.parent / "assets" / "iteration_prompt.md"

_CBC_FEATURES = [
    "lab_BASO_pct_last", "lab_EOS_pct_last", "lab_HB_last", "lab_HCT_last",
    "lab_LYMpct_last", "lab_MCH_last", "lab_MCHC_last", "lab_MCV_last",
    "lab_MONOpct_last", "lab_NEUTpct_last", "lab_PLT_last", "lab_RBC_last",
    "lab_RDW_last", "lab_WBC_last",
]

_BANNED = [
    "0.000001",
    "epsilon",
    "divide-by-zero",
    "stabilization",
    "stabilized",
    "1e-6",
    "1e-06",
]

_REQUIRED = [
    "{disease_name}",
    "{history_block}",
    "Propose 200 NEW expressions",
    "Use only +, -, *, / and parentheses.",
    "Previously scored by the GAT",
    "ONE line containing ONLY the expression",
]


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> int:
    if not _ASSET.exists():
        _fail(f"asset file missing: {_ASSET}")

    body = _ASSET.read_text()

    for feat in _CBC_FEATURES:
        if feat not in body:
            _fail(f"missing CBC feature: {feat}")

    for banned in _BANNED:
        if banned in body:
            _fail(f"BANNED string present: {banned!r}")

    for required in _REQUIRED:
        if required not in body:
            _fail(f"missing required phrase: {required!r}")

    n_disease = body.count("{disease_name}")
    if n_disease != 1:
        _fail(f"expected 1 {{disease_name}} placeholder, found {n_disease}")
    n_history = body.count("{history_block}")
    if n_history != 1:
        _fail(f"expected 1 {{history_block}} placeholder, found {n_history}")

    # Format-substitute test with both placeholders.
    sample_history = (
        "  0.652   lab_PLT_last / lab_LYMpct_last\n"
        "  0.640   lab_NEUTpct_last / lab_LYMpct_last\n"
    )
    try:
        rendered = body.format(disease_name="RheumatoidArthritis",
                               history_block=sample_history)
    except (KeyError, IndexError) as exc:
        _fail(f"format substitution failed: {exc}")
    if "RheumatoidArthritis" not in rendered:
        _fail("format() did not substitute {disease_name}")
    if sample_history.strip() not in rendered:
        _fail("format() did not substitute {history_block}")

    print("PASS verify_iteration_prompt: all 14 CBC features present, no banned "
          "strings, {disease_name} + {history_block} substitute cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
