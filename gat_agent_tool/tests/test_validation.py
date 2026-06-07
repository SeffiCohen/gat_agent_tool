"""Pure-Python tests for the validation helpers.

Run with:
    python -m pytest gat_agent_tool/tests
or:
    python gat_agent_tool/tests/test_validation.py   (uses unittest)

These tests deliberately avoid any torch / torch_geometric / LLM imports so
they can run in a minimal environment — the full package's heavy deps are only
needed for the core GatScorerTool and the consumer drivers.
"""
from __future__ import annotations

import os
import sys
import unittest

# Tests may be run directly (python gat_agent_tool/tests/test_validation.py), so
# make sure the sub-project root is on sys.path — the Python package lives at
# <sub-project-root>/gat_agent_tool/ and must resolve as a top-level import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBPROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _SUBPROJECT_ROOT not in sys.path:
    sys.path.insert(0, _SUBPROJECT_ROOT)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Import lazily so an import error shows up inside a test, not at
        # collection time — keeps the failure mode clear.
        from gat_agent_tool.validation import (
            looks_like_numeric_literal,
            parse_candidates,
            strip_leading_numbering_or_bullets,
        )
        self.looks = looks_like_numeric_literal
        self.parse = parse_candidates
        self.strip = strip_leading_numbering_or_bullets
        self.features = [
            "lab_RDW_mean",
            "lab_RBC_max",
            "lab_WBC_mean",
            "lab_NEUT_mean",
            "lab_cd_4",      # feature name legitimately contains a digit
            "lab_5HT_last",  # feature name starts with a digit after the underscore
        ]
        self.operators = ["+", "-", "*", "/"]

    # -- looks_like_numeric_literal --------------------------------------------
    def test_plain_valid_expression(self):
        self.assertFalse(self.looks("(lab_RDW_mean + lab_RBC_max) / lab_WBC_mean", self.features))

    def test_numeric_constant_rejected(self):
        self.assertTrue(self.looks("lab_RDW_mean + 0.5", self.features))
        self.assertTrue(self.looks("lab_RDW_mean * 2", self.features))
        self.assertTrue(self.looks("3 * lab_RBC_max", self.features))

    def test_digit_inside_feature_name_allowed(self):
        self.assertFalse(self.looks("lab_cd_4 * lab_RBC_max", self.features))
        self.assertFalse(self.looks("lab_5HT_last + lab_WBC_mean", self.features))

    def test_mixed_constant_and_digit_feature(self):
        # Constant present even though a digit-bearing feature is also present.
        self.assertTrue(self.looks("lab_cd_4 * 2 + lab_RBC_max", self.features))

    # -- strip_leading_numbering_or_bullets ------------------------------------
    def test_strip_numbering(self):
        self.assertEqual(self.strip("1. foo + bar"), "foo + bar")
        self.assertEqual(self.strip("1) foo + bar"), "foo + bar")
        self.assertEqual(self.strip("(1) foo + bar"), "foo + bar")
        self.assertEqual(self.strip("a. foo + bar"), "foo + bar")
        self.assertEqual(self.strip("• foo + bar"), "foo + bar")

    def test_strip_preserves_expression_operators(self):
        # Leading unary minus must survive; leading * is ambiguous but we
        # preserve it too (the GAT parse will catch genuinely malformed input).
        self.assertEqual(self.strip("- lab_A - lab_B"), "- lab_A - lab_B")
        self.assertEqual(self.strip("* lab_A * lab_B"), "* lab_A * lab_B")

    # -- parse_candidates ------------------------------------------------------
    def test_parse_realistic_llm_output(self):
        raw = (
            "1. (lab_RDW_mean + lab_RBC_max) / lab_WBC_mean\n"
            "- lab_NEUT_mean - lab_WBC_mean\n"
            "lab_RDW_mean + 0.5               # should be rejected (literal)\n"
            "some prose without operators or features\n"
            "(lab_cd_4 * lab_RBC_max)\n"
            "1. (lab_RDW_mean + lab_RBC_max) / lab_WBC_mean   # dup\n"
        )
        got = self.parse(raw, self.features, self.operators)
        self.assertEqual(len(got), 3, f"expected 3 unique kept, got {got!r}")
        self.assertIn("(lab_RDW_mean + lab_RBC_max) / lab_WBC_mean", got)
        self.assertIn("- lab_NEUT_mean - lab_WBC_mean", got)
        self.assertIn("(lab_cd_4 * lab_RBC_max)", got)

    def test_parse_handles_code_fences(self):
        raw = (
            "Here are my proposals:\n"
            "```\n"
            "lab_RDW_mean + lab_RBC_max\n"
            "lab_WBC_mean * lab_NEUT_mean\n"
            "```\n"
        )
        got = self.parse(raw, self.features, self.operators)
        self.assertEqual(len(got), 2)

    def test_parse_drops_trailing_comments(self):
        raw = "lab_RDW_mean + lab_RBC_max   # neutrophil-related ratio"
        got = self.parse(raw, self.features, self.operators)
        self.assertEqual(got, ["lab_RDW_mean + lab_RBC_max"])

    def test_parse_requires_operator_and_feature(self):
        raw = (
            "lab_RDW_mean\n"              # no operator
            "+ - * / () ^\n"              # no feature
            "lab_UNKNOWN + lab_OTHER\n"   # unknown features
            "lab_RDW_mean + lab_RBC_max\n"  # OK
        )
        got = self.parse(raw, self.features, self.operators)
        self.assertEqual(got, ["lab_RDW_mean + lab_RBC_max"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
