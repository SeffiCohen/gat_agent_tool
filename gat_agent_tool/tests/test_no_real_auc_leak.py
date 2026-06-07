"""Invariant: the iterative LLM<->GAT loop never leaks the silent real
univariate AUC into any LLM prompt.

The invariant is anchored at Code/iterative_llm_gat.py:386 (the
`_evaluate_expression` call site) and the prompt-history block immediately
following. If any of these tests fail, the silent-AUC experimental control
is broken and every downstream metric in skill_iter0_vs_final.csv is
invalid.

Tests:
  1. Static check: no prompt-builder function body contains 'real_auc'.
  2. Dynamic check: a synthetic history with real_auc=0.755 must not show
     "0.755" or "real_auc" anywhere in the rendered prompt.
  3. Static check: the skill's finalize.py never feeds real_auc into a
     prompt-construction code path.

Adversarial probe (manual): intentionally append
    prompt += f"\nreal_auc={real_auc}"
to one of the prompt-builders, rerun pytest, and confirm the test fails.
Then revert. This proves the test catches the bug it was written to catch.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ITER_FILE = REPO / "Code" / "iterative_llm_gat.py"


def _load_iter_module():
    spec = importlib.util.spec_from_file_location("iterative_llm_gat", ITER_FILE)
    if str(REPO / "Code") not in sys.path:
        sys.path.insert(0, str(REPO / "Code"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _function_source(name: str) -> str:
    tree = ast.parse(ITER_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise KeyError(name)


def test_iterative_llm_gat_file_exists():
    """Sanity: the file we anchor on must exist."""
    assert ITER_FILE.exists(), f"Anchor file missing: {ITER_FILE}"


def test_prompt_builders_never_reference_real_auc():
    """Static AST check: no prompt-builder function body contains 'real_auc'.

    Searched function names cover the typical iter-loop builder vocabulary.
    Functions that don't exist in this checkout are silently skipped.

    The canonical prompt-builder in this repo is `build_prompt` (line 90
    of Code/iterative_llm_gat.py). It receives a `history` argument whose
    dicts contain a `real_auc` key, but its body must never read that key.
    """
    candidates = (
        "build_prompt",
        "_render_history",
        "_build_prompt",
        "_initial_prompt",
        "_build_iteration_prompt",
        "_history_block",
        "_format_history",
        "_construct_prompt",
        "render_prompt",
    )
    found_any = False
    for fn in candidates:
        try:
            src = _function_source(fn)
        except KeyError:
            continue
        found_any = True
        # The function may MENTION real_auc in a docstring (describing the
        # invariant) but must not READ h["real_auc"] or substitute its value.
        # Strip the docstring before checking.
        tree = ast.parse(src)
        func = tree.body[0]
        body_no_doc = func.body
        if (body_no_doc and isinstance(body_no_doc[0], ast.Expr)
                and isinstance(body_no_doc[0].value, ast.Constant)
                and isinstance(body_no_doc[0].value.value, str)):
            body_no_doc = body_no_doc[1:]
        body_src = "\n".join(ast.unparse(s) for s in body_no_doc)
        assert "real_auc" not in body_src, (
            f"{fn} body references 'real_auc' -- silent-AUC invariant LEAK risk.\n"
            f"Offending body:\n{body_src[:500]}"
        )
    if not found_any:
        pytest.skip(
            "No prompt-builder function with a known name was found in "
            f"{ITER_FILE} -- consider updating the candidate list."
        )


def test_rendered_prompt_excludes_real_auc_values():
    """Dynamic check: a synthetic history with real_auc=0.755 must not show
    '0.755' or 'real_auc' anywhere in the rendered prompt.

    Loading iterative_llm_gat.py imports torch_geometric, which can fail
    in CI environments. We extract `build_prompt` via a fresh module
    namespace that only carries the typing/regex/etc imports it needs,
    bypassing the heavyweight torch_geometric import.
    """
    src = ITER_FILE.read_text()

    # Extract only what `build_prompt` needs -- it's pure-Python and depends
    # on no third-party packages.
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in (
            "typing",
        ):
            keep.append(node)
        elif isinstance(node, ast.Import) and any(
            a.name in ("re",) for a in node.names
        ):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in (
            "build_prompt",
            "_disease_name",
        ):
            keep.append(node)
    new_tree = ast.Module(body=keep, type_ignores=[])
    ns: dict = {}
    exec(compile(ast.fix_missing_locations(new_tree), str(ITER_FILE), "exec"), ns)

    builder = ns.get("build_prompt")
    if builder is None:
        pytest.skip(
            "build_prompt not found in iterative_llm_gat.py; rename "
            "invalidated the test. Update the test."
        )

    history = [
        {
            "iter": 1,
            "expression": "lab_NEUTpct_last/lab_LYMpct_last",
            "gat_score": 0.81234,
            "real_auc": 0.75567,
        },
        {
            "iter": 2,
            "expression": "(lab_PLT_last+lab_HB_last)/lab_RDW_last",
            "gat_score": 0.84211,
            "real_auc": 0.79900,
        },
    ]

    # Try several signatures (refactor-tolerant)
    prompt = None
    for kwargs in (
        dict(
            disease_name="rheumatoid arthritis",
            allowed_features=[
                "lab_PLT_last",
                "lab_NEUTpct_last",
                "lab_LYMpct_last",
                "lab_HB_last",
                "lab_RDW_last",
            ],
            operators=["+", "-", "*", "/"],
            history=history,
            pop_per_iter=200,
            max_history_shown=20,
        ),
        dict(
            disease_name="rheumatoid arthritis",
            allowed_features=[
                "lab_PLT_last",
                "lab_NEUTpct_last",
                "lab_LYMpct_last",
                "lab_HB_last",
                "lab_RDW_last",
            ],
            operators=["+", "-", "*", "/"],
            history=history,
            pop_per_iter=200,
        ),
        dict(history=history, disease_name="rheumatoid arthritis"),
    ):
        try:
            prompt = builder(**kwargs)
            break
        except TypeError:
            continue

    if prompt is None:
        pytest.skip(
            "Could not call build_prompt with any known signature; "
            "update the test."
        )

    # GAT scores SHOULD be present in some form
    assert any(s in prompt for s in ("0.812", "0.81")), (
        "GAT score missing from prompt -- expected '0.812' substring. "
        "Test signature likely wrong; rendered prompt was:\n" + prompt[:500]
    )
    # Real AUCs MUST NOT be present in any form
    forbidden_substrings = ["0.755", "0.799", "0.7556", "0.7990"]
    leaked = [s for s in forbidden_substrings if s in prompt]
    assert not leaked, (
        f"LEAK: real_auc value tokens {leaked} appear in prompt. "
        "Silent-AUC invariant broken. Prompt:\n" + prompt[:800]
    )
    # The literal token 'real_auc' may not appear either
    assert "real_auc" not in prompt, (
        "LEAK: literal token 'real_auc' appears in prompt:\n" + prompt[:800]
    )


def test_skill_finalize_does_not_feed_real_auc_into_prompt():
    """The skill's finalize.py reads real_auc from CSV, but only after the
    loop ends. Confirm via static scan that 'real_auc' does not appear inside
    any prompt-construction code path of finalize.py.
    """
    skill_finalize = REPO / ".claude/skills/biomarker-discovery/scripts/finalize.py"
    if not skill_finalize.exists():
        pytest.skip(f"Skill finalize.py not vendored on this checkout: {skill_finalize}")
    text = skill_finalize.read_text()
    # If 'prompt' appears, check the next few hundred characters do not
    # mention real_auc -- a coarse but defensible heuristic for "is real_auc
    # ever assembled into prompt-shaped string".
    if "prompt" in text:
        pieces = text.split("prompt")
        for piece in pieces[1:]:
            window = piece[:500]
            assert "real_auc" not in window, (
                "finalize.py may feed real_auc into a prompt; window:\n" + window
            )
