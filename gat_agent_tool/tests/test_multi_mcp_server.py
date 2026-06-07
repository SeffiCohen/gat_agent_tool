"""Tests for the multi-disease MCP server tool registrations and CLI surface.

The MCP server itself depends on FastMCP, which is fine for `_register_tools`
(we pass a FakeMCP that captures decorated functions instead of routing them
through MCP) but problematic for `main()`. We restrict `main()` testing to the
argparse-layer concerns (mutually-exclusive flags, env-var fallback) — those
exit before any FastMCP import path is touched. End-to-end startup is left to
an integration test.

Run with:
    python gat_agent_tool/tests/test_multi_mcp_server.py
or:
    python -m pytest gat_agent_tool/tests/test_multi_mcp_server.py
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

# Tests may be run directly (python gat_agent_tool/tests/test_multi_mcp_server.py),
# so make sure the sub-project root is on sys.path — the Python package lives at
# <sub-project-root>/gat_agent_tool/ and must resolve as a top-level import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBPROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _SUBPROJECT_ROOT not in sys.path:
    sys.path.insert(0, _SUBPROJECT_ROOT)


class FakeMCP:
    """Captures `@mcp.tool()`-decorated functions into a name-keyed dict.

    Mirrors just enough of FastMCP's surface for `_register_tools(mcp)` to run.
    """

    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


class FakeRegistry:
    """In-memory stand-in for `GatScorerRegistry`.

    Records calls so tests can assert pass-through, supports a `_fail_score`
    flag for the exception-handling test.
    """

    def __init__(self, diseases=None, gat_root: str = "/fake/gat"):
        self._diseases = list(diseases or ["242", "714"])
        self._gat_root = gat_root
        self.score_calls: list = []
        self.score_batch_calls: list = []
        self._fail_score = False

    # Read-side -----------------------------------------------------------
    def list_diseases(self):
        return list(self._diseases)

    def feature_names(self, did):
        if did not in self._diseases:
            raise KeyError(did)
        return [f"lab_X_last_for_{did}"]

    def info(self, did):
        if did not in self._diseases:
            raise KeyError(did)
        return {
            "model_type": "Fake",
            "num_features": 1,
            "num_operators": 4,
            "max_depth": 3,
            "graph_format_version": 2,
            "has_auc_transform": False,
            "checkpoint_path": f"/fake/{did}.pt",
            "device": "cpu",
        }

    # Score-side ----------------------------------------------------------
    def score(self, did, expr):
        self.score_calls.append((did, expr))
        if did not in self._diseases:
            raise KeyError(did)
        if self._fail_score:
            raise RuntimeError("simulated failure")
        return 0.7

    def score_batch(self, did, exprs):
        self.score_batch_calls.append((did, list(exprs)))
        if did not in self._diseases:
            raise KeyError(did)
        if self._fail_score:
            raise RuntimeError("simulated failure")
        return [0.7] * len(exprs)


def _install_registry(reg: "FakeRegistry | None"):
    """Set the module-global `_registry` and return the imported module."""
    import gat_agent_tool.multi_mcp_server as mod

    mod._registry = reg
    return mod


def _build_tools(reg: "FakeRegistry | None"):
    """Install `reg`, register tools onto a fresh FakeMCP, return (mod, mcp)."""
    mod = _install_registry(reg)
    mcp = FakeMCP()
    mod._register_tools(mcp)
    return mod, mcp


class MultiMcpServerToolTests(unittest.TestCase):
    """Tests against `_register_tools` + a captured FakeMCP."""

    def test_list_diseases_returns_registry_state(self):
        reg = FakeRegistry(diseases=["242", "714"], gat_root="/fake/gat")
        _, mcp = _build_tools(reg)
        result = mcp.tools["list_diseases"]()
        self.assertEqual(set(result["diseases"]), {"242", "714"})
        self.assertEqual(result["gat_root"], "/fake/gat")

    def test_get_feature_list_includes_disease_id(self):
        reg = FakeRegistry(diseases=["242", "714"])
        _, mcp = _build_tools(reg)
        result = mcp.tools["get_feature_list"]("714")
        self.assertEqual(result["disease_id"], "714")
        self.assertEqual(result["features"], ["lab_X_last_for_714"])
        required = {
            "disease_id", "features", "operators", "max_depth",
            "model_type", "rules", "score_range", "higher_is_better",
        }
        self.assertTrue(
            required.issubset(set(result.keys())),
            f"missing keys: {required - set(result.keys())}",
        )

    def test_unknown_disease_returns_error_dict_not_exception(self):
        reg = FakeRegistry(diseases=["242", "714"])
        _, mcp = _build_tools(reg)
        try:
            result = mcp.tools["get_feature_list"]("nonexistent")
        except Exception as exc:  # noqa: BLE001 — ensures graceful path
            self.fail(f"unknown disease must NOT raise; got {exc!r}")
        self.assertIsInstance(result.get("error"), str)
        self.assertTrue(result["error"], "error message must be non-empty")
        self.assertEqual(set(result["available"]), {"242", "714"})

    def test_score_expression_passes_disease_id_through(self):
        reg = FakeRegistry(diseases=["242", "714"])
        _, mcp = _build_tools(reg)
        result = mcp.tools["score_expression"]("714", "lab_A + lab_B")
        self.assertEqual(reg.score_calls, [("714", "lab_A + lab_B")])
        self.assertEqual(result["disease_id"], "714")
        self.assertEqual(result["expression"], "lab_A + lab_B")
        self.assertEqual(result["score"], 0.7)
        self.assertIsNone(result["error"])

    def test_score_expression_score_is_none_returns_error(self):
        # Build a registry whose .score() returns None to model a parse failure.
        reg = FakeRegistry(diseases=["242", "714"])
        original_score = reg.score

        def score_returns_none(did, expr):
            reg.score_calls.append((did, expr))
            return None

        reg.score = score_returns_none  # type: ignore[assignment]
        try:
            _, mcp = _build_tools(reg)
            result = mcp.tools["score_expression"]("714", "garbage @@@")
            self.assertIsNone(result["score"])
            self.assertEqual(result["error"], "could not parse expression into a graph")
        finally:
            reg.score = original_score  # type: ignore[assignment]

    def test_score_expressions_preserves_order_and_length(self):
        reg = FakeRegistry(diseases=["242", "714"])
        _, mcp = _build_tools(reg)
        inputs = [
            "lab_A + lab_B",
            "lab_C - lab_D",
            "lab_E * lab_F",
            "lab_G / lab_H",
            "(lab_I + lab_J) * lab_K",
        ]
        result = mcp.tools["score_expressions"]("714", inputs)
        self.assertEqual(len(result), 5)
        self.assertEqual([d["expression"] for d in result], inputs)

    def test_score_expression_catches_internal_exception(self):
        reg = FakeRegistry(diseases=["242", "714"])
        reg._fail_score = True
        _, mcp = _build_tools(reg)
        try:
            result = mcp.tools["score_expression"]("714", "lab_A + lab_B")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"internal failure must NOT propagate; got {exc!r}")
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"], "simulated failure")

    def test_cli_mutually_exclusive_root_and_config(self):
        from gat_agent_tool.multi_mcp_server import main

        argv = ["gat-agent-multi-mcp", "--gat-root", "/x", "--registry-config", "/y"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 2)

    def test_cli_requires_one_of_root_config_or_env(self):
        from gat_agent_tool.multi_mcp_server import main

        argv = ["gat-agent-multi-mcp"]
        env = {k: v for k, v in os.environ.items() if k != "GAT_AGENT_REGISTRY_ROOT"}
        with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                main()

    def test_require_registry_raises_when_unconfigured(self):
        mod = _install_registry(None)
        with self.assertRaises(RuntimeError):
            mod._require_registry()


if __name__ == "__main__":
    unittest.main(verbosity=2)
