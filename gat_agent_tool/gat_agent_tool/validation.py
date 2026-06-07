"""gat_agent_tool.validation — pure-Python helpers for parsing and validating
expression strings emitted by an LLM.

These helpers are pipeline-agnostic (no torch, no torch_geometric, no repo deps)
so they can be unit-tested trivially and reused by the HF driver, the OpenAI
driver, and the MCP server alike.

Rules enforced (align with the BD_Paper pipeline's constraints):
  - Expression must contain at least one arithmetic operator.
  - Expression must reference at least one known feature name.
  - No numeric literals — the GAT pipeline rejects numeric constants, so we
    catch them at parse time rather than wasting a GAT call on a guaranteed
    None score.

The parser is forgiving about LLM formatting noise: strips markdown bullets,
numbered lists, trailing comments, and fenced code blocks.
"""
from __future__ import annotations

import re
from typing import List


# ---------------------------------------------------------------------------
# Numbering / bullet prefixes an LLM might emit despite "no numbering" prompts.
# Intentionally does NOT strip a leading "-" or "*" because those can also be
# unary-minus or multiplication in a well-formed expression.
# ---------------------------------------------------------------------------
_PREFIX_PATTERNS = [
    re.compile(r"^\d+\s*[\.):\s]+"),       # "1. " / "1) " / "1: "
    re.compile(r"^\(\d+\)\s*"),            # "(1) "
    re.compile(r"^[a-zA-Z]\s*[\.)]\s*"),   # "a. " / "A) "
    re.compile(r"^[•·]\s*"),               # bullets
]


def strip_leading_numbering_or_bullets(text: str) -> str:
    """Remove common list/bullet prefixes an LLM may inject despite instructions.

    Preserves leading ``-`` and ``*`` since those are valid expression tokens
    (unary minus, multiplication).
    """
    s = text.lstrip()
    changed = True
    while changed:
        changed = False
        for pat in _PREFIX_PATTERNS:
            m = pat.match(s)
            if m:
                s = s[m.end():].lstrip()
                changed = True
    return s


# Regex that matches a digit that ISN'T part of an identifier. We run this on
# the "residual" expression — i.e., after stripping out all known feature names
# (so `lab_cd_4` doesn't trip the guard). If a digit still appears, it must be
# a literal constant, which the pipeline forbids.
_NUMERIC_LITERAL_RE = re.compile(r"(?<![A-Za-z_])\d")


def looks_like_numeric_literal(expr: str, allowed_features: List[str]) -> bool:
    """True if ``expr`` contains a digit that's not part of a known feature name.

    The pipeline rejects numeric literals. Feature names may legitimately contain
    digits (e.g. ``lab_cd_4``, ``lab_5HT_mean``), so we can't just regex for any
    digit — we strip the known feature names first, then check what remains.
    """
    stripped = expr
    # Longest first so ``lab_alpha_10`` is removed before ``lab_alpha_1``.
    for f in sorted(allowed_features, key=len, reverse=True):
        stripped = stripped.replace(f, "")
    return bool(_NUMERIC_LITERAL_RE.search(stripped))


def _drop_fenced_code_block(text: str) -> str:
    """If the LLM wraps its output in ```` ``` ```` blocks, keep only the block body."""
    m = re.search(r"```(?:[a-zA-Z]*\n)?(.*?)```", text, flags=re.DOTALL)
    return m.group(1) if m else text


def parse_candidates(
    raw_text: str,
    allowed_features: List[str],
    operators: List[str],
) -> List[str]:
    """Parse an LLM's raw text into a list of candidate expressions.

    Splits on newlines, strips each line, filters out lines that fail any rule,
    and deduplicates while preserving order.

    Parameters
    ----------
    raw_text : str
        The LLM's raw output (may contain markdown, numbering, or prose).
    allowed_features : List[str]
        Case-sensitive feature names that are legal to reference.
    operators : List[str]
        Arithmetic operators that must appear somewhere in each expression.
    """
    body = _drop_fenced_code_block(raw_text)
    op_class = "[" + re.escape("".join(operators)) + "]"
    op_re = re.compile(op_class)

    seen = set()
    out: List[str] = []
    for raw_line in body.splitlines():
        s = raw_line.strip().strip("`#;")
        if not s:
            continue
        s = strip_leading_numbering_or_bullets(s)
        s = s.split("#", 1)[0].strip()  # drop trailing "  # comment"
        if not s:
            continue
        if not op_re.search(s):
            continue  # no operator -> not an expression
        if not any(f in s for f in allowed_features):
            continue  # no known feature -> not a valid candidate
        if looks_like_numeric_literal(s, allowed_features):
            continue  # numeric constants forbidden
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out
