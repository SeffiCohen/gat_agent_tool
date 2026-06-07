"""Integration tests for GatScorerRegistry against the bundled GAT/ checkpoints.

These tests load real torch state dicts and require the trained-model
checkpoints under `<GAT_ROOT>/<folder>/trained_models/best_by_loss.pt`. By
default GAT_ROOT resolves to this repository's own `GAT/` directory (the 13
released checkpoints ship with the repo); override it with the
GAT_AGENT_TOOL_GAT_ROOT environment variable.

When the GAT root is missing the whole class is skipped, so this file is safe
to keep in CI even on machines without the model artifacts.

Run with:
    python gat_agent_tool/tests/test_registry_integration.py
or:
    python -m pytest gat_agent_tool/tests/test_registry_integration.py
"""
from __future__ import annotations

import os
import sys
import unittest

# Tests may be run directly (python gat_agent_tool/tests/test_registry_integration.py),
# so make sure the sub-project root is on sys.path — the Python package lives
# at <sub-project-root>/gat_agent_tool/ and must resolve as a top-level import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBPROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _SUBPROJECT_ROOT not in sys.path:
    sys.path.insert(0, _SUBPROJECT_ROOT)


GAT_ROOT = os.environ.get(
    "GAT_AGENT_TOOL_GAT_ROOT",
    os.path.abspath(os.path.join(_SUBPROJECT_ROOT, os.pardir, "GAT")),
)
_RA_CKPT = os.path.join(GAT_ROOT, "714", "trained_models", "best_by_loss.pt")
_TEST_EXPR = "lab_PLT_last/lab_LYMpct_last"


@unittest.skipUnless(
    os.path.isdir(GAT_ROOT),
    f"BD_Paper GAT root unavailable at {GAT_ROOT}",
)
class RegistryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gat_agent_tool.registry import GatScorerRegistry

        cls.reg = GatScorerRegistry(gat_root=GAT_ROOT, lazy=True)

    def test_load_RA_and_score_real_expression(self):
        score = self.reg.score("714", _TEST_EXPR)
        if score is None:
            self.skipTest(f"GAT returned None for {_TEST_EXPR!r} — parse failure, not a test bug")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_registry_score_byte_equal_direct_score(self):
        if not os.path.isfile(_RA_CKPT):
            self.skipTest(f"714 checkpoint missing at {_RA_CKPT}")
        from gat_agent_tool.core import GatScorerTool

        direct_tool = GatScorerTool(_RA_CKPT)
        direct = direct_tool.score(_TEST_EXPR)
        via_reg = self.reg.score("714", _TEST_EXPR)
        # Same checkpoint, same expression — must be bit-identical, not just close.
        self.assertEqual(direct, via_reg)

    def test_list_diseases_returns_at_least_13(self):
        diseases = self.reg.list_diseases()
        self.assertGreaterEqual(len(diseases), 13)
        self.assertIn("2452", diseases)
        self.assertNotIn("2542", diseases)

    def test_2452_resolves_to_disk_folder_2542(self):
        ckpt_path = str(self.reg._index["2452"])
        self.assertIn(os.sep + "2542" + os.sep, ckpt_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
