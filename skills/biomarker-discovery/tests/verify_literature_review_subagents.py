#!/usr/bin/env python3
"""verify_literature_review_subagents.py — guard the parallel-Opus-4.8 Workflow pattern.

Phase 1 fans out 3 research agents via the ``Workflow`` tool, each running on
**Opus 4.8 at max effort** (``model: "opus"`` + ``effort: "max"`` on the
``agent()`` call). This verifier reads ``references/literature_review.md`` and
``SKILL.md`` and asserts:
  - Both reference the ``Workflow`` tool (the fan-out mechanism).
  - Both pin Opus 4.8 (``model: "opus"`` resolving to the latest Opus) at
    ``max`` effort.
  - literature_review.md documents the ``parallel()`` / ``agent()`` fan-out and
    the 3 distinct research angles (mechanism / established / preprints) over
    the bio-research MCP tools (Open Targets / PubMed / bioRxiv).
  - Neither file still pins the stale "Opus 4.7" version string.

Exit 0 on all-pass, 1 on first failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent.parent
_LITREV = _SKILL_DIR / "references" / "literature_review.md"
_SKILL_MD = _SKILL_DIR / "SKILL.md"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _check_litrev() -> None:
    if not _LITREV.exists():
        _fail(f"literature_review.md missing at {_LITREV}")
    body = _LITREV.read_text()

    # Required substrings — the Workflow + Opus-4.8-max fan-out pattern.
    must_contain = [
        ("Workflow",                   "the Workflow tool name"),
        ("parallel(",                  "the parallel() fan-out in the Workflow script"),
        ("agent(",                     "the agent() call in the Workflow script"),
        ('model: "opus"',              'the prose model: "opus" override'),
        ("'opus'",                     "the script-form model: 'opus' override"),
        ('effort: "max"',              'the prose effort: "max" tier'),
        ("'max'",                      "the script-form effort: 'max' tier"),
        ("Opus 4.8",                   "the explicit Opus 4.8 version string"),
        ("general-purpose",            "the general-purpose agentType/subagent_type"),
        ("Agent 1 — Mechanism",        "Agent 1 section heading (mechanism)"),
        ("Agent 2 — Established",      "Agent 2 section heading (established ratios)"),
        ("Agent 3 — Recent preprints", "Agent 3 section heading (preprints)"),
        ("Open Targets",               "Open Targets reference (Agent 1)"),
        ("PubMed",                     "PubMed reference (Agent 2)"),
        ("bioRxiv",                    "bioRxiv reference (Agent 3)"),
    ]
    for needle, why in must_contain:
        if needle not in body:
            _fail(f"literature_review.md missing {why}: expected substring "
                  f"{needle!r} not found")

    # Banned substrings — a regression to the stale model pin.
    must_not_contain = [
        ("Opus 4.7", "stale Opus 4.7 version pin (should be Opus 4.8)"),
    ]
    for needle, why in must_not_contain:
        if needle in body:
            _fail(f"literature_review.md contains banned substring {needle!r} "
                  f"({why})")

    # Sanity: the doc should be non-trivial in length (the Workflow flow is
    # substantially documented).
    if len(body) < 4000:
        _fail(f"literature_review.md is suspiciously short ({len(body)} chars). "
              f"The parallel Workflow flow needs detailed documentation.")


def _check_skill_md() -> None:
    if not _SKILL_MD.exists():
        _fail(f"SKILL.md missing at {_SKILL_MD}")
    body = _SKILL_MD.read_text()

    # Phase 1 section in SKILL.md should also reflect the Workflow + Opus-4.8 approach.
    must_contain = [
        ("Workflow",        "Phase 1 description mentioning the Workflow tool"),
        ('model: "opus"',   'explicit model: "opus" reference in Phase 1'),
        ('effort: "max"',   'explicit effort: "max" reference in Phase 1'),
        ("Opus 4.8",        "explicit Opus 4.8 version string in Phase 1"),
        ("parallel",        "parallel execution language in Phase 1"),
    ]
    for needle, why in must_contain:
        if needle not in body:
            _fail(f"SKILL.md missing {why}: expected substring {needle!r} "
                  f"not found")

    if "Opus 4.7" in body:
        _fail("SKILL.md contains banned substring 'Opus 4.7' "
              "(should be Opus 4.8)")


def main() -> int:
    _check_litrev()
    _check_skill_md()
    print("PASS verify_literature_review_subagents: parallel Opus-4.8-max "
          "Workflow pattern documented in both literature_review.md and "
          "SKILL.md (3 agents: mechanism / established / preprints; "
          'Workflow parallel() fan-out; model: "opus" + effort: "max" = '
          "Opus 4.8 max; agentType: general-purpose)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
