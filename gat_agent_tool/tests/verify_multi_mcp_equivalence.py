#!/usr/bin/env python3
"""Diff IMPLEMENTER vs ORACLE multi-disease MCP server implementations.

Both `gat_agent_tool/multi_mcp_server.py` (IMPLEMENTER) and
`gat_agent_tool/_multi_mcp_server_oracle.py` (ORACLE) expose the same
module-level helpers (`_require_registry`, `_register_tools`, `main`) and
register the same 5 tools onto a FastMCP-shaped object. This script registers
both against a `FakeMCP`, drives matched pairs of tools through the same
inputs, and asserts identical output dicts.

Run after both files exist:

    cd /path/to/gat-agent-tool/gat_agent_tool
    python tests/verify_multi_mcp_equivalence.py

Exit code 0 = all PASS; 1 = any FAIL.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Add the sub-project root to sys.path so 'gat_agent_tool' resolves.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBPROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _SUBPROJECT_ROOT not in sys.path:
    sys.path.insert(0, _SUBPROJECT_ROOT)


# ---------------------------------------------------------------------------
# Import both implementations.
# Bail out cleanly if either is missing — the verifier may have been launched
# before one of the parallel agents finished writing its file.
# ---------------------------------------------------------------------------
try:
    import gat_agent_tool.multi_mcp_server as impl_mod
except Exception as e:  # noqa: BLE001
    print(f"FAIL [import_impl]: could not import IMPLEMENTER's "
          f"gat_agent_tool.multi_mcp_server: {e!r}")
    sys.exit(1)

try:
    import gat_agent_tool._multi_mcp_server_oracle as oracle_mod  # type: ignore
except Exception as e:  # noqa: BLE001
    print(f"FAIL [import_oracle]: could not import ORACLE's "
          f"gat_agent_tool._multi_mcp_server_oracle: {e!r}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test infrastructure: a FastMCP-shaped fake that records tool registrations,
# and a stand-in for GatScorerRegistry that records every call so we can
# diff behavior without spinning up torch.
# ---------------------------------------------------------------------------
class FakeMCP:
    """Minimal stand-in for `mcp.server.fastmcp.FastMCP`.

    Exposes the same `.tool()` decorator pattern. Both implementations are
    free to register tools however they like — by decorator, by dict + loop,
    whatever — as long as the result is a populated `.tools` mapping.
    """

    def __init__(self) -> None:
        self.tools: Dict[str, Any] = {}

    def tool(self):  # noqa: D401 — matches FastMCP signature.
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


class FakeRegistry:
    """Drop-in stand-in for `gat_agent_tool.registry.GatScorerRegistry`.

    Records each `.score()` / `.score_batch()` call so we can confirm both
    implementations pass through to the registry the same way. Note: each
    impl gets its OWN FakeRegistry instance so call-recording stays disjoint.
    """

    def __init__(
        self,
        diseases: list = None,
        fail_score: bool = False,
        score_returns: float = 0.7,
        score_returns_none: bool = False,
    ) -> None:
        self._diseases = list(diseases or ["242", "714"])
        self._fail = fail_score
        self._sret = score_returns
        self._snone = score_returns_none
        self.score_calls: list = []
        self.batch_calls: list = []

    def list_diseases(self) -> List[str]:
        return list(self._diseases)

    def score(self, did: str, expr: str):
        self.score_calls.append((did, expr))
        if self._fail:
            raise RuntimeError("simulated failure")
        if self._snone:
            return None
        return self._sret

    def score_batch(self, did: str, exprs):
        self.batch_calls.append((did, list(exprs)))
        if self._fail:
            raise RuntimeError("simulated failure")
        return [None if self._snone else self._sret] * len(exprs)

    def feature_names(self, did: str) -> List[str]:
        if did not in self._diseases:
            raise KeyError(did)
        return [f"lab_X_last_for_{did}"]

    def info(self, did: str) -> Dict[str, Any]:
        if did not in self._diseases:
            raise KeyError(did)
        return {
            "disease_id": did,
            "model_type": "Fake",
            "num_features": 1,
            "num_operators": 4,
            "max_depth": 3,
            "graph_format_version": 2,
            "has_auc_transform": False,
            "checkpoint_path": f"/fake/{did}.pt",
            "device": "cpu",
        }


def install_registry(
    mod,
    reg: FakeRegistry,
    gat_root: str = "/fake/gat",
    ckpt_fn: str = "best_by_loss.pt",
) -> None:
    """Set the module-level globals on either impl or oracle.

    Both impls resolve `_registry` from the module global at tool-call time
    (via `_require_registry()`), so install BEFORE registering tools is
    sufficient. We still call `_register_tools` *after* installation to
    handle the alternative pattern (dict-of-closures) cleanly.
    """
    mod._registry = reg
    if hasattr(mod, "_gat_root_for_display"):
        mod._gat_root_for_display = gat_root
    if hasattr(mod, "_checkpoint_filename_for_display"):
        mod._checkpoint_filename_for_display = ckpt_fn


def setup_pair(
    impl_reg: FakeRegistry,
    oracle_reg: FakeRegistry,
    gat_root: str = "/fake/gat",
    ckpt_fn: str = "best_by_loss.pt",
):
    """Install registries on both modules, register tools onto fresh FakeMCPs,
    and return the two `(mcp, reg, mod)` triples ready for tool calls."""
    install_registry(impl_mod, impl_reg, gat_root=gat_root, ckpt_fn=ckpt_fn)
    install_registry(oracle_mod, oracle_reg, gat_root=gat_root, ckpt_fn=ckpt_fn)
    impl_mcp = FakeMCP()
    oracle_mcp = FakeMCP()
    impl_mod._register_tools(impl_mcp)
    oracle_mod._register_tools(oracle_mcp)
    return impl_mcp, oracle_mcp


# ---------------------------------------------------------------------------
# Diff harness in the style of verify_orchestrator_helpers.py /
# verify_compare_picks.py: PASS/FAIL prints, totals, exit 1 on any FAIL.
# ---------------------------------------------------------------------------
PASS: list = []
FAIL: list = []


def check(name: str, expected: Any, actual: Any) -> None:
    if expected == actual:
        PASS.append(name)
        print(f"[{name}] PASS")
    else:
        FAIL.append((name, expected, actual))
        print(f"[{name}] FAIL: expected={expected!r} actual={actual!r}")


def check_true(name: str, cond: bool, msg: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"[{name}] PASS")
    else:
        FAIL.append((name, True, False))
        print(f"[{name}] FAIL: {msg}")


def dict_equal(a: Any, b: Any) -> bool:
    return a == b


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

# S1: list_diseases output matches
def s1_list_diseases() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"])
    oracle_reg = FakeRegistry(diseases=["242", "714"])
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_out = impl_mcp.tools["list_diseases"]()
    oracle_out = oracle_mcp.tools["list_diseases"]()

    check("S1.list_diseases", oracle_out, impl_out)
    check_true(
        "S1.impl_returns_dict",
        isinstance(impl_out, dict),
        f"impl returned {type(impl_out).__name__}",
    )
    check_true(
        "S1.oracle_returns_dict",
        isinstance(oracle_out, dict),
        f"oracle returned {type(oracle_out).__name__}",
    )


# S2: get_feature_list for known disease matches
def s2_get_feature_list_known() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"])
    oracle_reg = FakeRegistry(diseases=["242", "714"])
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_out = impl_mcp.tools["get_feature_list"]("714")
    oracle_out = oracle_mcp.tools["get_feature_list"]("714")

    check("S2.get_feature_list_known", oracle_out, impl_out)

    expected_keys = {
        "disease_id",
        "features",
        "operators",
        "max_depth",
        "model_type",
        "rules",
        "score_range",
        "higher_is_better",
    }
    check("S2.impl_keyset", expected_keys, set(impl_out.keys()))
    check("S2.oracle_keyset", expected_keys, set(oracle_out.keys()))

    # Spec checks (apply to both since they should be equal anyway)
    check("S2.disease_id", "714", impl_out.get("disease_id"))
    check("S2.disease_id_oracle", "714", oracle_out.get("disease_id"))
    check("S2.score_range", [0.5, 1.0], impl_out.get("score_range"))
    check("S2.score_range_oracle", [0.5, 1.0], oracle_out.get("score_range"))
    check("S2.rules_len", 5, len(impl_out.get("rules", [])))
    check("S2.rules_len_oracle", 5, len(oracle_out.get("rules", [])))


# S3: get_feature_list for unknown disease matches
def s3_get_feature_list_unknown() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"])
    oracle_reg = FakeRegistry(diseases=["242", "714"])
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_out = impl_mcp.tools["get_feature_list"]("nonexistent")
    oracle_out = oracle_mcp.tools["get_feature_list"]("nonexistent")

    check("S3.get_feature_list_unknown", oracle_out, impl_out)
    for key in ("error", "available", "disease_id"):
        check_true(
            f"S3.impl_has_{key}",
            key in impl_out,
            f"impl missing {key!r}: {sorted(impl_out)}",
        )
        check_true(
            f"S3.oracle_has_{key}",
            key in oracle_out,
            f"oracle missing {key!r}: {sorted(oracle_out)}",
        )


# S4: get_model_info for known disease matches
def s4_get_model_info_known() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"])
    oracle_reg = FakeRegistry(diseases=["242", "714"])
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_out = impl_mcp.tools["get_model_info"]("714")
    oracle_out = oracle_mcp.tools["get_model_info"]("714")

    check("S4.get_model_info_known", oracle_out, impl_out)

    expected_keys = {
        "disease_id",
        "model_type",
        "num_features",
        "num_operators",
        "max_depth",
        "graph_format_version",
        "has_auc_transform",
        "checkpoint_path",
        "device",
    }
    check("S4.impl_keyset", expected_keys, set(impl_out.keys()))
    check("S4.oracle_keyset", expected_keys, set(oracle_out.keys()))


# S5: get_model_info for unknown disease matches
def s5_get_model_info_unknown() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"])
    oracle_reg = FakeRegistry(diseases=["242", "714"])
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_out = impl_mcp.tools["get_model_info"]("nonexistent")
    oracle_out = oracle_mcp.tools["get_model_info"]("nonexistent")

    check("S5.get_model_info_unknown", oracle_out, impl_out)
    for key in ("error", "available", "disease_id"):
        check_true(
            f"S5.impl_has_{key}",
            key in impl_out,
            f"impl missing {key!r}: {sorted(impl_out)}",
        )
        check_true(
            f"S5.oracle_has_{key}",
            key in oracle_out,
            f"oracle missing {key!r}: {sorted(oracle_out)}",
        )


# S6: score_expression success matches
def s6_score_expression_success() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"], score_returns=0.7)
    oracle_reg = FakeRegistry(diseases=["242", "714"], score_returns=0.7)
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_out = impl_mcp.tools["score_expression"]("714", "lab_A + lab_B")
    oracle_out = oracle_mcp.tools["score_expression"]("714", "lab_A + lab_B")

    check("S6.score_expression_success", oracle_out, impl_out)

    # Both should have invoked .score("714", "lab_A + lab_B") exactly once on
    # their respective FakeRegistry. They each have their own registry, so
    # diff the recorded calls per-impl rather than across.
    check("S6.impl_score_calls", [("714", "lab_A + lab_B")], impl_reg.score_calls)
    check("S6.oracle_score_calls", [("714", "lab_A + lab_B")], oracle_reg.score_calls)
    check("S6.score_value_impl", 0.7, impl_out.get("score"))
    check("S6.score_value_oracle", 0.7, oracle_out.get("score"))


# S7: score_expression with score=None (parse failure) matches
def s7_score_expression_none() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"], score_returns_none=True)
    oracle_reg = FakeRegistry(diseases=["242", "714"], score_returns_none=True)
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_out = impl_mcp.tools["score_expression"]("714", "garbage")
    oracle_out = oracle_mcp.tools["score_expression"]("714", "garbage")

    check("S7.score_expression_none", oracle_out, impl_out)
    check("S7.impl_score_is_None", None, impl_out.get("score"))
    check("S7.oracle_score_is_None", None, oracle_out.get("score"))
    check(
        "S7.impl_error_msg",
        "could not parse expression into a graph",
        impl_out.get("error"),
    )
    check(
        "S7.oracle_error_msg",
        "could not parse expression into a graph",
        oracle_out.get("error"),
    )


# S8: score_expression for unknown disease matches
def s8_score_expression_unknown_disease() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"])
    oracle_reg = FakeRegistry(diseases=["242", "714"])
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_out = impl_mcp.tools["score_expression"]("nonexistent", "lab_X")
    oracle_out = oracle_mcp.tools["score_expression"]("nonexistent", "lab_X")

    check("S8.score_expression_unknown", oracle_out, impl_out)
    for key in ("disease_id", "expression", "score", "error", "available"):
        check_true(
            f"S8.impl_has_{key}",
            key in impl_out,
            f"impl missing {key!r}: {sorted(impl_out)}",
        )
        check_true(
            f"S8.oracle_has_{key}",
            key in oracle_out,
            f"oracle missing {key!r}: {sorted(oracle_out)}",
        )
    check("S8.impl_score_is_None", None, impl_out.get("score"))
    check("S8.oracle_score_is_None", None, oracle_out.get("score"))
    # Critically: the registry's .score() must NOT have been called.
    check("S8.impl_no_score_call", [], impl_reg.score_calls)
    check("S8.oracle_no_score_call", [], oracle_reg.score_calls)


# S9: score_expression catches internal exception matches
def s9_score_expression_internal_failure() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"], fail_score=True)
    oracle_reg = FakeRegistry(diseases=["242", "714"], fail_score=True)
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    impl_raised = False
    oracle_raised = False
    try:
        impl_out = impl_mcp.tools["score_expression"]("714", "lab_A")
    except Exception as e:  # noqa: BLE001
        impl_raised = True
        FAIL.append(("S9.impl_propagates_exception", "no_exception", repr(e)))
        print(f"[S9.impl_propagates_exception] FAIL: impl let exception escape: {e!r}")
        return
    try:
        oracle_out = oracle_mcp.tools["score_expression"]("714", "lab_A")
    except Exception as e:  # noqa: BLE001
        oracle_raised = True
        FAIL.append(("S9.oracle_propagates_exception", "no_exception", repr(e)))
        print(f"[S9.oracle_propagates_exception] FAIL: oracle let exception escape: {e!r}")
        return

    check("S9.score_expression_internal_failure", oracle_out, impl_out)
    check("S9.impl_score_is_None", None, impl_out.get("score"))
    check("S9.oracle_score_is_None", None, oracle_out.get("score"))
    check("S9.impl_error_msg", "simulated failure", impl_out.get("error"))
    check("S9.oracle_error_msg", "simulated failure", oracle_out.get("error"))


# S10: score_expressions preserves order and length
def s10_score_expressions_batch() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"], score_returns=0.7)
    oracle_reg = FakeRegistry(diseases=["242", "714"], score_returns=0.7)
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    inputs = ["a", "b", "c", "d", "e"]
    impl_out = impl_mcp.tools["score_expressions"]("714", inputs)
    oracle_out = oracle_mcp.tools["score_expressions"]("714", inputs)

    check("S10.score_expressions_batch", oracle_out, impl_out)
    check("S10.impl_length", 5, len(impl_out))
    check("S10.oracle_length", 5, len(oracle_out))
    check("S10.impl_order", inputs, [d["expression"] for d in impl_out])
    check("S10.oracle_order", inputs, [d["expression"] for d in oracle_out])

    # Element-wise equality is implied by S10.score_expressions_batch above,
    # but spell it out so we get a precise FAIL location if it diverges.
    for i, (a, b) in enumerate(zip(impl_out, oracle_out)):
        check(f"S10.elem_{i}", b, a)


# S11: score_expressions for unknown disease returns list of error dicts
def s11_score_expressions_unknown_disease() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"])
    oracle_reg = FakeRegistry(diseases=["242", "714"])
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    inputs = ["x", "y", "z"]
    impl_out = impl_mcp.tools["score_expressions"]("nonexistent", inputs)
    oracle_out = oracle_mcp.tools["score_expressions"]("nonexistent", inputs)

    check("S11.score_expressions_unknown", oracle_out, impl_out)
    check("S11.impl_length", 3, len(impl_out))
    check("S11.oracle_length", 3, len(oracle_out))
    for i, d in enumerate(impl_out):
        for key in ("disease_id", "expression", "score", "error", "available"):
            check_true(
                f"S11.impl_elem_{i}_has_{key}",
                key in d,
                f"impl elem {i} missing {key!r}: {sorted(d)}",
            )
    for i, d in enumerate(oracle_out):
        for key in ("disease_id", "expression", "score", "error", "available"):
            check_true(
                f"S11.oracle_elem_{i}_has_{key}",
                key in d,
                f"oracle elem {i} missing {key!r}: {sorted(d)}",
            )
    # Critically: score_batch must NOT have been called.
    check("S11.impl_no_batch_call", [], impl_reg.batch_calls)
    check("S11.oracle_no_batch_call", [], oracle_reg.batch_calls)


# S12: _require_registry raises RuntimeError when unconfigured
def s12_require_registry_unconfigured() -> None:
    # Save current values, then nuke.
    impl_saved = impl_mod._registry
    oracle_saved = oracle_mod._registry
    impl_mod._registry = None
    oracle_mod._registry = None

    try:
        impl_raised_kind = None
        oracle_raised_kind = None

        try:
            impl_mod._require_registry()
        except RuntimeError:
            impl_raised_kind = "RuntimeError"
        except Exception as e:  # noqa: BLE001
            impl_raised_kind = type(e).__name__

        try:
            oracle_mod._require_registry()
        except RuntimeError:
            oracle_raised_kind = "RuntimeError"
        except Exception as e:  # noqa: BLE001
            oracle_raised_kind = type(e).__name__

        check("S12.impl_raises_RuntimeError", "RuntimeError", impl_raised_kind)
        check("S12.oracle_raises_RuntimeError", "RuntimeError", oracle_raised_kind)
    finally:
        impl_mod._registry = impl_saved
        oracle_mod._registry = oracle_saved


# S13: tool registration count matches
def s13_tool_count() -> None:
    impl_reg = FakeRegistry(diseases=["242", "714"])
    oracle_reg = FakeRegistry(diseases=["242", "714"])
    impl_mcp, oracle_mcp = setup_pair(impl_reg, oracle_reg)

    expected_names = {
        "list_diseases",
        "get_feature_list",
        "get_model_info",
        "score_expression",
        "score_expressions",
    }
    check("S13.impl_tool_count", 5, len(impl_mcp.tools))
    check("S13.oracle_tool_count", 5, len(oracle_mcp.tools))
    check("S13.impl_tool_names", expected_names, set(impl_mcp.tools.keys()))
    check("S13.oracle_tool_names", expected_names, set(oracle_mcp.tools.keys()))
    check("S13.tool_names_match", set(oracle_mcp.tools.keys()), set(impl_mcp.tools.keys()))


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("S1", s1_list_diseases),
    ("S2", s2_get_feature_list_known),
    ("S3", s3_get_feature_list_unknown),
    ("S4", s4_get_model_info_known),
    ("S5", s5_get_model_info_unknown),
    ("S6", s6_score_expression_success),
    ("S7", s7_score_expression_none),
    ("S8", s8_score_expression_unknown_disease),
    ("S9", s9_score_expression_internal_failure),
    ("S10", s10_score_expressions_batch),
    ("S11", s11_score_expressions_unknown_disease),
    ("S12", s12_require_registry_unconfigured),
    ("S13", s13_tool_count),
]


for label, fn in SCENARIOS:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        FAIL.append((f"{label}.exception", "no_exception", repr(e)))
        print(f"[{label}.exception] FAIL: scenario raised {type(e).__name__}: {e!r}")


total = len(PASS) + len(FAIL)
print(f"\n{len(PASS)}/{total} PASS — multi-MCP server implementations agree"
      if not FAIL
      else f"\n{len(PASS)}/{total} PASS")
if FAIL:
    print(f"{len(FAIL)} FAIL — multi-MCP server implementations DIFFER")
    for name, expected, actual in FAIL:
        print(f"  - {name}: expected={expected!r}  actual={actual!r}")

sys.exit(1 if FAIL else 0)
