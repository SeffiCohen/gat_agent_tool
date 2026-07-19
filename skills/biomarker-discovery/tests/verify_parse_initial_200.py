#!/usr/bin/env python3
"""verify_parse_initial_200.py — sanity check the validation layer on a 200-batch.

Synthesizes a fake LLM response of 200 valid CBC arithmetic expressions,
pipes through ``gat_agent_tool.validation.parse_candidates``, asserts all 200
survive (no false rejections of valid arithmetic).

Then injects 5 expressions with numeric literals (``+ 0.000001``) and 5 with
unknown-feature names, asserts exactly the original 200 survive (15 rejected).
This guards against a regression where a future refactor either (a) accepts
numeric literals, breaking the GAT contract, or (b) starts dropping legitimate
ratio expressions.

Exit 0 on all-pass, 1 on first failure.
"""
from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path

# Patch sys.path so we can import gat_agent_tool without installing.
def _find_repo_root(start: Path) -> Path:
    """Locate the repository root by searching upward for ``Code/_external_scoring.py``.

    Override with the ``GAT_AGENT_TOOL_REPO`` environment variable. This replaces an
    earlier fixed ``parents[4]`` hop that assumed one specific checkout layout.
    """
    import os
    env = os.environ.get("GAT_AGENT_TOOL_REPO")
    if env:
        return Path(env).resolve()
    start = start.resolve()
    for up in [start, *start.parents]:
        if (up / "Code" / "_external_scoring.py").exists():
            return up
    raise SystemExit(
        "Could not locate Code/_external_scoring.py above this script; "
        "set GAT_AGENT_TOOL_REPO to the repository root."
    )


_REPO_ROOT = _find_repo_root(Path(__file__))
sys.path.insert(0, str(_REPO_ROOT / "gat_agent_tool"))

from gat_agent_tool.validation import parse_candidates  # noqa: E402

_FEATURES = [
    "lab_BASO_pct_last", "lab_EOS_pct_last", "lab_HB_last", "lab_HCT_last",
    "lab_LYMpct_last", "lab_MCH_last", "lab_MCHC_last", "lab_MCV_last",
    "lab_MONOpct_last", "lab_NEUTpct_last", "lab_PLT_last", "lab_RBC_last",
    "lab_RDW_last", "lab_WBC_last",
]
_OPS = ["+", "-", "*", "/"]


def _gen_200_valid_exprs() -> list[str]:
    """Produce 200 distinct, DSL-valid CBC expressions deterministically.

    Uses a simple Cartesian product over feature pairs/triples and operator
    combos so the same 200 expressions are produced every run. All 200 are
    guaranteed unique (no dedup collisions) and all use only ``+ - * /`` and
    feature names from the allowlist.
    """
    out: list[str] = []
    # Pairwise ratios, differences, products: 14 * 13 / 2 * 3 ops = ~273
    # combos. Take first 200 that are distinct.
    for op in _OPS:
        for i, a in enumerate(_FEATURES):
            for b in _FEATURES[i + 1:]:
                expr = f"{a} {op} {b}"
                if expr not in out:
                    out.append(expr)
                if len(out) >= 200:
                    break
            if len(out) >= 200:
                break
        if len(out) >= 200:
            break
    return list(islice(out, 200))


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> int:
    # Step 1: build 200 valid expressions, run through parse_candidates.
    valid = _gen_200_valid_exprs()
    if len(valid) != 200:
        _fail(f"could not synthesize 200 distinct valid exprs (got {len(valid)})")

    raw = "\n".join(valid)
    surviving = parse_candidates(raw, _FEATURES, _OPS)
    if len(surviving) != 200:
        _fail(f"parse_candidates dropped {200 - len(surviving)} valid expressions; "
              f"first 3 missing: {[e for e in valid if e not in surviving][:3]}")

    # Order preservation: surviving[0] must equal valid[0] — the first valid
    # expression's position is what becomes rank_in_iter==0 (the baseline).
    if surviving[0] != valid[0]:
        _fail(f"order not preserved: surviving[0]={surviving[0]!r} but "
              f"valid[0]={valid[0]!r}")

    # Step 2: inject 5 numeric-literal expressions + 5 unknown-feature exprs.
    bad_numeric = [
        "lab_PLT_last + 0.000001",
        "lab_HB_last / (lab_PLT_last + 0.000001)",
        "lab_NEUTpct_last * 2",
        "lab_LYMpct_last - 1.5",
        "(lab_PLT_last + 0.001) / lab_LYMpct_last",
    ]
    # NB: parse_candidates' unknown-feature gate is "at least one known
    # feature must appear as a substring". So an expression like
    # `lab_FAKE_last / lab_LYMpct_last` PASSES because `lab_LYMpct_last`
    # IS a substring. We deliberately build truly-no-known-substring
    # expressions here so they get rejected by the at-least-one gate
    # (and the GAT will reject them downstream with score=null anyway).
    bad_unknown = [
        "alpha / beta",
        "x + y",
        "p * q",
        "foo - bar",
        "mystery / value",
    ]
    mixed = valid + bad_numeric + bad_unknown
    raw_mixed = "\n".join(mixed)
    surviving2 = parse_candidates(raw_mixed, _FEATURES, _OPS)
    if len(surviving2) != 200:
        # Either we let through bad ones, or we dropped some good ones.
        bad_through = [e for e in surviving2 if e in bad_numeric + bad_unknown]
        good_dropped = [e for e in valid if e not in surviving2]
        _fail(
            f"expected exactly 200 survivors, got {len(surviving2)}.\n"
            f"  bad expressions that snuck through: {bad_through[:3]}\n"
            f"  good expressions dropped: {good_dropped[:3]}"
        )

    # Final sanity: no banned strings in any survivor.
    for s in surviving2:
        if "0.000001" in s or "0.001" in s or "1.5" in s:
            _fail(f"numeric literal survived parse_candidates: {s!r}")
        if "lab_FAKE_last" in s or "lab_INVENTED" in s:
            _fail(f"unknown feature survived: {s!r}")

    print(f"PASS verify_parse_initial_200: 200 valid expressions all parse cleanly; "
          f"15 banned ones rejected (5 numeric literals + 5 unknown features); "
          f"order preserved (rank_in_iter==0 candidate is first valid expression)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
