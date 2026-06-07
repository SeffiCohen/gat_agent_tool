"""Tests for the GatScorerRegistry multi-disease wrapper.

These tests use a FakeScorer (monkeypatched in for `gat_agent_tool.core.GatScorerTool`)
to avoid loading torch / real GAT checkpoints. The registry's auto-discovery,
caching, lazy/eager construction, thread-safety, and error semantics are all
exercised against zero-byte placeholder checkpoint files in a tempdir.

Run with:
    python gat_agent_tool/tests/test_registry.py
or:
    python -m pytest gat_agent_tool/tests/test_registry.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

# Tests may be run directly (python gat_agent_tool/tests/test_registry.py), so
# make sure the sub-project root is on sys.path — the Python package lives at
# <sub-project-root>/gat_agent_tool/ and must resolve as a top-level import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBPROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _SUBPROJECT_ROOT not in sys.path:
    sys.path.insert(0, _SUBPROJECT_ROOT)


# Disk folder layout mirrors the real BD_Paper/GAT/ tree. Note "2542" — the
# disk-folder-override maps it to the advertised disease_id "2452".
_DISK_FOLDERS = [
    "242", "250", "2542", "277", "340", "555", "556",
    "5790", "696", "7100", "7101", "7102", "714",
]
_EXPECTED_DISEASE_IDS = {
    "242", "250", "277", "340", "555", "556",
    "696", "714", "2452", "5790", "7100", "7101", "7102",
}


class FakeScorer:
    """Drop-in stand-in for `gat_agent_tool.core.GatScorerTool`.

    Records every constructor invocation into a class-level list so tests can
    assert lazy / cache / threading behavior without spinning up torch.
    """

    calls: list = []   # list of (checkpoint_path, kwargs) tuples

    def __init__(self, checkpoint_path, *, device="auto", **kw):
        FakeScorer.calls.append((checkpoint_path, {"device": device, **kw}))
        self.checkpoint_path = checkpoint_path
        self.device = device
        # Disease-specific marker — folder name (e.g. "714", "2542") sits two
        # parents up from the .pt file: <root>/<folder>/trained_models/best_by_loss.pt
        from pathlib import Path as _P
        self.feature_names = [f"lab_X_last_{_P(str(checkpoint_path)).parent.parent.name}"]
        self.operators = ["+", "-", "*", "/"]
        self.max_depth = 3
        self.model_type = "FakeModel"

    def score(self, expr):
        return 0.7

    def score_batch(self, exprs):
        return [0.7] * len(exprs)

    def info(self):
        return type("I", (), {
            "max_depth": self.max_depth,
            "model_type": self.model_type,
            "num_features": 1,
            "num_operators": 4,
            "graph_format_version": 2,
            "has_auc_transform": False,
            "checkpoint_path": self.checkpoint_path,
        })()

    @classmethod
    def reset(cls):
        cls.calls = []


def _build_fake_gat_root(tmpdir: str, folders=_DISK_FOLDERS, *, add_dsstore=True) -> Path:
    """Create a fake GAT root mirroring `<root>/<folder>/trained_models/best_by_loss.pt`.

    Returns the root Path. Files are zero-byte; the registry only needs them to
    exist for auto-discovery — the FakeScorer doesn't read their contents.
    """
    root = Path(tmpdir)
    for folder in folders:
        ckpt_dir = root / folder / "trained_models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "best_by_loss.pt").touch()
    if add_dsstore:
        # macOS metadata file at the top level — must be skipped by discovery.
        (root / ".DS_Store").touch()
    return root


@patch("gat_agent_tool.core.GatScorerTool", FakeScorer)
class GatScorerRegistryTests(unittest.TestCase):
    """Tests for the synchronous parts of the registry."""

    def setUp(self) -> None:
        # Clear global state on the fake so individual tests don't leak.
        FakeScorer.reset()

    def test_autodiscovery_finds_all_13(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            reg = GatScorerRegistry(gat_root=root, lazy=True)
            self.assertEqual(set(reg.list_diseases()), _EXPECTED_DISEASE_IDS)
            # `.DS_Store` must NOT have leaked into the advertised set.
            self.assertNotIn(".DS_Store", reg.list_diseases())

    def test_disk_folder_override_2452_to_2542(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            reg = GatScorerRegistry(gat_root=root, lazy=True)
            ckpt_path = str(reg._index["2452"])
            self.assertTrue(
                ckpt_path.endswith(os.path.join("2542", "trained_models", "best_by_loss.pt")),
                f"expected 2452 to resolve to a 2542/ checkpoint, got {ckpt_path!r}",
            )

    def test_lazy_does_not_construct_scorer(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            FakeScorer.reset()
            reg = GatScorerRegistry(gat_root=root, lazy=True)
            _ = reg.list_diseases()
            self.assertEqual(FakeScorer.calls, [])

    def test_get_constructs_once_then_caches(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            FakeScorer.reset()
            reg = GatScorerRegistry(gat_root=root, lazy=True)
            t1 = reg.get("714")
            t2 = reg.get("714")
            self.assertEqual(len(FakeScorer.calls), 1)
            self.assertIs(t1, t2)

    def test_unknown_disease_id_raises_keyerror(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            reg = GatScorerRegistry(gat_root=root, lazy=True)
            with self.assertRaises(KeyError) as cm:
                reg.get("999")
            msg = str(cm.exception)
            self.assertIn("999", msg)
            self.assertIn("available", msg)

    def test_unload_evicts_then_reconstructs(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            FakeScorer.reset()
            reg = GatScorerRegistry(gat_root=root, lazy=True)
            t1 = reg.get("714")
            reg.unload("714")
            t2 = reg.get("714")
            self.assertEqual(len(FakeScorer.calls), 2)
            self.assertIsNot(t1, t2)

    def test_explicit_registry_config_overrides_autodiscovery(self):
        from gat_agent_tool.registry import GatScorerRegistry

        # No gat_root, just an explicit mapping. The registry should advertise
        # exactly the supplied disease_ids — no filesystem scan happens.
        reg = GatScorerRegistry(registry_config={"X": "/some/path"}, lazy=True)
        self.assertEqual(reg.list_diseases(), ["X"])

    def test_thread_safety_init_runs_once(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            FakeScorer.reset()
            reg = GatScorerRegistry(gat_root=root, lazy=True)

            n = 8
            barrier = threading.Barrier(n)
            results: list = [None] * n

            def worker(i: int) -> None:
                barrier.wait()
                results[i] = reg.get("714")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(FakeScorer.calls), 1)
            # All workers must observe the *same* cached instance.
            first = results[0]
            for r in results[1:]:
                self.assertIs(r, first)

    def test_constructor_rejects_both_root_and_config(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with self.assertRaises(ValueError):
            GatScorerRegistry(
                gat_root=Path("/x"),
                registry_config={"X": "/y"},
            )

    def test_constructor_rejects_neither_root_nor_config(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with self.assertRaises(ValueError):
            GatScorerRegistry()

    def test_eager_loads_all_at_init(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            FakeScorer.reset()
            _ = GatScorerRegistry(gat_root=root, lazy=False)
            self.assertEqual(len(FakeScorer.calls), 13)

    def test_score_delegates_to_scorer(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            reg = GatScorerRegistry(gat_root=root, lazy=True)
            self.assertEqual(reg.score("714", "lab_X_last + lab_Y_last"), 0.7)
            self.assertEqual(
                reg.score_batch("714", ["a", "b", "c"]),
                [0.7, 0.7, 0.7],
            )

    def test_feature_names_per_disease(self):
        from gat_agent_tool.registry import GatScorerRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_fake_gat_root(tmp)
            reg = GatScorerRegistry(gat_root=root, lazy=True)
            f714 = reg.feature_names("714")
            f250 = reg.feature_names("250")
            # Both are non-empty lists, but disease-specific (the FakeScorer
            # bakes the checkpoint path tail into the feature name).
            self.assertTrue(f714 and f250)
            self.assertNotEqual(f714, f250)


if __name__ == "__main__":
    unittest.main(verbosity=2)
