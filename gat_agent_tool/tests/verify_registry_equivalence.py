#!/usr/bin/env python3
"""Diff IMPLEMENTER vs ORACLE GatScorerRegistry implementations.

Both `gat_agent_tool/registry.py` (IMPLEMENTER) and
`gat_agent_tool/_registry_oracle.py` (ORACLE) export a `GatScorerRegistry`
class with identical public API. This script drives matched pairs through the
same matrix of inputs and asserts they produce identical outputs.

Run after both files exist:

    cd /path/to/gat-agent-tool/gat_agent_tool
    python tests/verify_registry_equivalence.py

Exit code 0 = all PASS; 1 = any FAIL.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

# Add the sub-project root to sys.path so 'gat_agent_tool' resolves
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
    from gat_agent_tool.registry import GatScorerRegistry as ImplReg
except Exception as e:  # noqa: BLE001
    print(f"FAIL [import_impl]: could not import IMPLEMENTER's "
          f"gat_agent_tool.registry.GatScorerRegistry: {e!r}")
    sys.exit(1)

try:
    from gat_agent_tool._registry_oracle import (  # type: ignore
        GatScorerRegistry as OracleReg,
    )
except Exception as e:  # noqa: BLE001
    print(f"FAIL [import_oracle]: could not import ORACLE's "
          f"gat_agent_tool._registry_oracle.GatScorerRegistry: {e!r}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test infrastructure mirroring tests/test_registry.py — re-defined here so
# this script is standalone (no dependency on the test module).
# ---------------------------------------------------------------------------
class FakeScorer:
    """Drop-in stand-in for `gat_agent_tool.core.GatScorerTool`.

    Records every constructor invocation into a class-level list so we can
    diff lazy / cache / threading behavior without spinning up torch.
    """

    calls: list = []  # list of (checkpoint_path, device) tuples

    def __init__(self, checkpoint_path, *, device="auto", **kw):
        FakeScorer.calls.append((checkpoint_path, device))
        self.checkpoint_path = checkpoint_path
        self.device = device
        # Disease-specific marker — the tail of the checkpoint path embeds the
        # disk folder, so two diseases must produce different feature lists.
        self.feature_names = [f"lab_X_last_{str(checkpoint_path)[-30:]}"]
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


_DISK_FOLDERS = [
    "242", "250", "2542", "277", "340", "555", "556",
    "5790", "696", "7100", "7101", "7102", "714",
]


def make_fake_gat_root() -> str:
    """Build a tempdir mirroring BD_Paper/GAT/ layout. Returns absolute path."""
    tmpdir = tempfile.mkdtemp(prefix="fake_gat_")
    for f in _DISK_FOLDERS:
        d = Path(tmpdir) / f / "trained_models"
        d.mkdir(parents=True, exist_ok=True)
        (d / "best_by_loss.pt").write_bytes(b"")
    # macOS metadata file at the top level — must be skipped by discovery.
    (Path(tmpdir) / ".DS_Store").write_bytes(b"")
    return tmpdir


# ---------------------------------------------------------------------------
# Diff harness in the style of verify_orchestrator_helpers.py /
# verify_compare_picks.py: PASS/FAIL prints, totals, exit 1 on any FAIL.
# ---------------------------------------------------------------------------
PASS: list = []
FAIL: list = []


def check(name, expected, actual):
    if expected == actual:
        PASS.append(name)
        print(f"[{name}] PASS")
    else:
        FAIL.append((name, expected, actual))
        print(f"[{name}] FAIL: oracle={expected!r} impl={actual!r}")


def check_true(name, cond, msg=""):
    if cond:
        PASS.append(name)
        print(f"[{name}] PASS")
    else:
        FAIL.append((name, True, False))
        print(f"[{name}] FAIL: {msg}")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

# S1: list_diseases identical with auto-discovery
def s1_list_diseases_autodiscovery():
    root = make_fake_gat_root()
    impl = ImplReg(gat_root=root, lazy=True)
    oracle = OracleReg(gat_root=root, lazy=True)
    impl_list = impl.list_diseases()
    oracle_list = oracle.list_diseases()
    check("S1.list_diseases_autodiscovery", oracle_list, impl_list)
    # Both should include 2452 (overridden) and exclude 2542 (the on-disk name).
    check_true(
        "S1.contains_2452_impl",
        "2452" in impl_list,
        f"impl list missing 2452: {impl_list}",
    )
    check_true(
        "S1.excludes_2542_impl",
        "2542" not in impl_list,
        f"impl list still contains raw 2542: {impl_list}",
    )
    check_true(
        "S1.contains_2452_oracle",
        "2452" in oracle_list,
        f"oracle list missing 2452: {oracle_list}",
    )
    check_true(
        "S1.excludes_2542_oracle",
        "2542" not in oracle_list,
        f"oracle list still contains raw 2542: {oracle_list}",
    )


# S2: list_diseases identical with explicit registry_config
def s2_list_diseases_explicit():
    cfg = {"714": "/some/path", "250": "/other/path"}
    impl = ImplReg(registry_config=cfg, lazy=True)
    oracle = OracleReg(registry_config=cfg, lazy=True)
    check(
        "S2.list_diseases_explicit",
        oracle.list_diseases(),
        impl.list_diseases(),
    )


# S3: get() returns equivalent scorer for same disease (FakeScorer patched)
def s3_get_returns_equivalent_scorer():
    root = make_fake_gat_root()

    with patch("gat_agent_tool.core.GatScorerTool", FakeScorer):
        FakeScorer.reset()
        impl = ImplReg(gat_root=root, lazy=True)
        impl_scorer = impl.get("714")
        impl_calls = list(FakeScorer.calls)

        FakeScorer.reset()
        oracle = OracleReg(gat_root=root, lazy=True)
        oracle_scorer = oracle.get("714")
        oracle_calls = list(FakeScorer.calls)

        check(
            "S3.get_checkpoint_path",
            oracle_scorer.checkpoint_path,
            impl_scorer.checkpoint_path,
        )
        check("S3.get_call_count", len(oracle_calls), len(impl_calls))
        # Both should have called FakeScorer with the same checkpoint path.
        check(
            "S3.get_constructor_arg",
            oracle_calls[0][0] if oracle_calls else None,
            impl_calls[0][0] if impl_calls else None,
        )


# S4: KeyError messages match for unknown disease_id
def s4_keyerror_messages_match():
    root = make_fake_gat_root()
    impl = ImplReg(gat_root=root, lazy=True)
    oracle = OracleReg(gat_root=root, lazy=True)

    impl_msg = None
    oracle_msg = None

    try:
        impl.get("999")
    except KeyError as e:
        impl_msg = str(e)
    except Exception as e:  # noqa: BLE001
        FAIL.append(("S4.impl_raises_KeyError", "KeyError", type(e).__name__))
        print(f"[S4.impl_raises_KeyError] FAIL: got {type(e).__name__}={e!r}")
        return

    try:
        oracle.get("999")
    except KeyError as e:
        oracle_msg = str(e)
    except Exception as e:  # noqa: BLE001
        FAIL.append(("S4.oracle_raises_KeyError", "KeyError", type(e).__name__))
        print(f"[S4.oracle_raises_KeyError] FAIL: got {type(e).__name__}={e!r}")
        return

    if impl_msg is None or oracle_msg is None:
        FAIL.append(("S4.both_raise_KeyError", "raised", "did not raise"))
        print(
            f"[S4.both_raise_KeyError] FAIL: impl_raised={impl_msg is not None} "
            f"oracle_raised={oracle_msg is not None}"
        )
        return

    check_true("S4.impl_raises_KeyError", True)
    check_true("S4.oracle_raises_KeyError", True)

    # Substring checks first (per spec — these must always hold)
    for tok in ("999", "available"):
        check_true(
            f"S4.impl_msg_contains_{tok}",
            tok in impl_msg,
            f"missing {tok!r} in impl message: {impl_msg!r}",
        )
        check_true(
            f"S4.oracle_msg_contains_{tok}",
            tok in oracle_msg,
            f"missing {tok!r} in oracle message: {oracle_msg!r}",
        )

    # Stronger byte-equal check.
    check("S4.keyerror_msg_equal", oracle_msg, impl_msg)


# S5: ValueError on both-None and both-set constructor
def s5_value_errors():
    # both None
    impl_err = None
    oracle_err = None
    try:
        ImplReg()
    except ValueError as e:
        impl_err = str(e)
    except Exception as e:  # noqa: BLE001
        FAIL.append(("S5.impl_neither_ValueError", "ValueError", type(e).__name__))
        print(
            f"[S5.impl_neither_ValueError] FAIL: got {type(e).__name__}={e!r}"
        )
    try:
        OracleReg()
    except ValueError as e:
        oracle_err = str(e)
    except Exception as e:  # noqa: BLE001
        FAIL.append(("S5.oracle_neither_ValueError", "ValueError", type(e).__name__))
        print(
            f"[S5.oracle_neither_ValueError] FAIL: got {type(e).__name__}={e!r}"
        )
    check_true(
        "S5.impl_neither_raises_ValueError",
        impl_err is not None,
        "impl did not raise ValueError when called with no args",
    )
    check_true(
        "S5.oracle_neither_raises_ValueError",
        oracle_err is not None,
        "oracle did not raise ValueError when called with no args",
    )

    # both set
    impl_err2 = None
    oracle_err2 = None
    try:
        ImplReg(gat_root=Path("/x"), registry_config={"X": "/y"})
    except ValueError as e:
        impl_err2 = str(e)
    except Exception as e:  # noqa: BLE001
        FAIL.append(("S5.impl_both_ValueError", "ValueError", type(e).__name__))
        print(
            f"[S5.impl_both_ValueError] FAIL: got {type(e).__name__}={e!r}"
        )
    try:
        OracleReg(gat_root=Path("/x"), registry_config={"X": "/y"})
    except ValueError as e:
        oracle_err2 = str(e)
    except Exception as e:  # noqa: BLE001
        FAIL.append(("S5.oracle_both_ValueError", "ValueError", type(e).__name__))
        print(
            f"[S5.oracle_both_ValueError] FAIL: got {type(e).__name__}={e!r}"
        )
    check_true(
        "S5.impl_both_raises_ValueError",
        impl_err2 is not None,
        "impl did not raise ValueError when both gat_root + registry_config set",
    )
    check_true(
        "S5.oracle_both_raises_ValueError",
        oracle_err2 is not None,
        "oracle did not raise ValueError when both gat_root + registry_config set",
    )


# S6: disk_folder_override behavior identical
def s6_disk_folder_override():
    root = make_fake_gat_root()

    # Default override (2452 -> 2542)
    impl = ImplReg(gat_root=root, lazy=True)
    oracle = OracleReg(gat_root=root, lazy=True)

    impl_list = impl.list_diseases()
    oracle_list = oracle.list_diseases()

    impl_has_2452 = "2452" in impl_list
    oracle_has_2452 = "2452" in oracle_list
    check("S6.default_override_2452_present", oracle_has_2452, impl_has_2452)

    # Path under 2452 must end in 2542/trained_models/best_by_loss.pt.
    # Coerce to str — IMPL stores _index values as str, ORACLE may store
    # them as pathlib.Path. We diff via str() so cosmetic-only storage type
    # doesn't show up as a behavioral difference.
    impl_2452_path = impl._index.get("2452")
    oracle_2452_path = oracle._index.get("2452")
    impl_2452_str = str(impl_2452_path) if impl_2452_path is not None else None
    oracle_2452_str = str(oracle_2452_path) if oracle_2452_path is not None else None
    suffix = os.path.join("2542", "trained_models", "best_by_loss.pt")

    impl_endswith = bool(impl_2452_str) and impl_2452_str.endswith(suffix)
    oracle_endswith = bool(oracle_2452_str) and oracle_2452_str.endswith(suffix)
    check("S6.default_override_path_endswith_2542", oracle_endswith, impl_endswith)
    check_true(
        "S6.impl_2452_resolves_to_2542",
        impl_endswith,
        f"impl _index['2452']={impl_2452_path!r} does not end with {suffix!r}",
    )
    check_true(
        "S6.oracle_2452_resolves_to_2542",
        oracle_endswith,
        f"oracle _index['2452']={oracle_2452_path!r} does not end with {suffix!r}",
    )

    # Empty override — both should advertise 2542 (the raw on-disk folder),
    # and neither should advertise 2452.
    impl2 = ImplReg(gat_root=root, lazy=True, disk_folder_override={})
    oracle2 = OracleReg(gat_root=root, lazy=True, disk_folder_override={})
    impl2_list = impl2.list_diseases()
    oracle2_list = oracle2.list_diseases()
    check("S6.empty_override_list", oracle2_list, impl2_list)
    check_true(
        "S6.impl_empty_override_has_2542",
        "2542" in impl2_list and "2452" not in impl2_list,
        f"impl empty-override list wrong: {impl2_list}",
    )
    check_true(
        "S6.oracle_empty_override_has_2542",
        "2542" in oracle2_list and "2452" not in oracle2_list,
        f"oracle empty-override list wrong: {oracle2_list}",
    )


# S7: lazy = True does not import core / construct any scorer
def s7_lazy_does_not_construct():
    root = make_fake_gat_root()

    with patch("gat_agent_tool.core.GatScorerTool", FakeScorer):
        FakeScorer.reset()
        impl = ImplReg(gat_root=root, lazy=True)
        _ = impl.list_diseases()
        impl_calls = list(FakeScorer.calls)

        FakeScorer.reset()
        oracle = OracleReg(gat_root=root, lazy=True)
        _ = oracle.list_diseases()
        oracle_calls = list(FakeScorer.calls)

        check("S7.lazy_impl_no_calls", [], impl_calls)
        check("S7.lazy_oracle_no_calls", [], oracle_calls)
        check("S7.lazy_call_count_match", len(oracle_calls), len(impl_calls))


# S8: eager = False vs eager = True call count
def s8_eager_call_counts():
    root = make_fake_gat_root()

    with patch("gat_agent_tool.core.GatScorerTool", FakeScorer):
        # Eager — both must construct 13 scorers.
        FakeScorer.reset()
        _ = ImplReg(gat_root=root, lazy=False)
        impl_eager = len(FakeScorer.calls)

        FakeScorer.reset()
        _ = OracleReg(gat_root=root, lazy=False)
        oracle_eager = len(FakeScorer.calls)

        check("S8.eager_impl_call_count", 13, impl_eager)
        check("S8.eager_oracle_call_count", 13, oracle_eager)
        check("S8.eager_match", oracle_eager, impl_eager)

        # Lazy — both must have 0 calls until .get() is invoked.
        FakeScorer.reset()
        _ = ImplReg(gat_root=root, lazy=True)
        impl_lazy = len(FakeScorer.calls)

        FakeScorer.reset()
        _ = OracleReg(gat_root=root, lazy=True)
        oracle_lazy = len(FakeScorer.calls)

        check("S8.lazy_impl_call_count", 0, impl_lazy)
        check("S8.lazy_oracle_call_count", 0, oracle_lazy)
        check("S8.lazy_match", oracle_lazy, impl_lazy)


# S9: unload + re-get reconstructs in both
def s9_unload_then_reget():
    root = make_fake_gat_root()

    with patch("gat_agent_tool.core.GatScorerTool", FakeScorer):
        FakeScorer.reset()
        impl = ImplReg(gat_root=root, lazy=True)
        _ = impl.get("714")
        impl.unload("714")
        _ = impl.get("714")
        impl_calls = len(FakeScorer.calls)

        FakeScorer.reset()
        oracle = OracleReg(gat_root=root, lazy=True)
        _ = oracle.get("714")
        oracle.unload("714")
        _ = oracle.get("714")
        oracle_calls = len(FakeScorer.calls)

        check("S9.impl_reconstructs_after_unload", 2, impl_calls)
        check("S9.oracle_reconstructs_after_unload", 2, oracle_calls)
        check("S9.match", oracle_calls, impl_calls)


# S10: thread safety constructs once in both
def s10_thread_safety():
    root = make_fake_gat_root()

    def run_thread_test(reg_cls):
        FakeScorer.reset()
        reg = reg_cls(gat_root=root, lazy=True)
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
        return len(FakeScorer.calls), results

    with patch("gat_agent_tool.core.GatScorerTool", FakeScorer):
        impl_calls, impl_results = run_thread_test(ImplReg)
        oracle_calls, oracle_results = run_thread_test(OracleReg)

        check("S10.impl_constructs_once_under_threads", 1, impl_calls)
        check("S10.oracle_constructs_once_under_threads", 1, oracle_calls)
        check("S10.match_call_count", oracle_calls, impl_calls)

        # All workers in each registry must observe the same cached instance.
        impl_all_same = all(r is impl_results[0] for r in impl_results)
        oracle_all_same = all(r is oracle_results[0] for r in oracle_results)
        check_true(
            "S10.impl_all_workers_share_cache",
            impl_all_same,
            "impl workers got different scorer instances",
        )
        check_true(
            "S10.oracle_all_workers_share_cache",
            oracle_all_same,
            "oracle workers got different scorer instances",
        )


# S11: info() returns dict shape match
def s11_info_dict_shape():
    root = make_fake_gat_root()

    with patch("gat_agent_tool.core.GatScorerTool", FakeScorer):
        FakeScorer.reset()
        impl = ImplReg(gat_root=root, lazy=True)
        oracle = OracleReg(gat_root=root, lazy=True)

        impl_info = impl.info("714")
        oracle_info = oracle.info("714")

        check_true(
            "S11.impl_info_is_dict",
            isinstance(impl_info, dict),
            f"impl.info('714') not dict: {type(impl_info).__name__}",
        )
        check_true(
            "S11.oracle_info_is_dict",
            isinstance(oracle_info, dict),
            f"oracle.info('714') not dict: {type(oracle_info).__name__}",
        )

        if isinstance(impl_info, dict) and isinstance(oracle_info, dict):
            check(
                "S11.info_keyset_match",
                set(oracle_info.keys()),
                set(impl_info.keys()),
            )
            check_true(
                "S11.impl_info_has_disease_id",
                "disease_id" in impl_info,
                f"impl info missing 'disease_id' key: {sorted(impl_info)}",
            )
            check_true(
                "S11.oracle_info_has_disease_id",
                "disease_id" in oracle_info,
                f"oracle info missing 'disease_id' key: {sorted(oracle_info)}",
            )
            check(
                "S11.info_disease_id_value",
                oracle_info.get("disease_id"),
                impl_info.get("disease_id"),
            )


# S12: feature_names per disease identical
def s12_feature_names():
    root = make_fake_gat_root()

    with patch("gat_agent_tool.core.GatScorerTool", FakeScorer):
        FakeScorer.reset()
        impl = ImplReg(gat_root=root, lazy=True)

        FakeScorer.reset()
        oracle = OracleReg(gat_root=root, lazy=True)

        # Determinism within each registry — call twice, must match.
        impl_a = impl.feature_names("714")
        impl_b = impl.feature_names("714")
        check("S12.impl_self_consistent", impl_a, impl_b)

        oracle_a = oracle.feature_names("714")
        oracle_b = oracle.feature_names("714")
        check("S12.oracle_self_consistent", oracle_a, oracle_b)

        # Cross-impl: both must produce identical feature names because
        # the underlying checkpoint path is the same.
        check("S12.cross_impl_feature_names", oracle_a, impl_a)


# S13: score / score_batch return values identical
def s13_score_and_score_batch():
    root = make_fake_gat_root()

    with patch("gat_agent_tool.core.GatScorerTool", FakeScorer):
        FakeScorer.reset()
        impl = ImplReg(gat_root=root, lazy=True)
        FakeScorer.reset()
        oracle = OracleReg(gat_root=root, lazy=True)

        impl_score = impl.score("714", "lab_X")
        oracle_score = oracle.score("714", "lab_X")
        check("S13.score_impl_value", 0.7, impl_score)
        check("S13.score_oracle_value", 0.7, oracle_score)
        check("S13.score_match", oracle_score, impl_score)

        impl_batch = impl.score_batch("714", ["a", "b"])
        oracle_batch = oracle.score_batch("714", ["a", "b"])
        check("S13.score_batch_impl_value", [0.7, 0.7], impl_batch)
        check("S13.score_batch_oracle_value", [0.7, 0.7], oracle_batch)
        check("S13.score_batch_match", oracle_batch, impl_batch)


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("S1", s1_list_diseases_autodiscovery),
    ("S2", s2_list_diseases_explicit),
    ("S3", s3_get_returns_equivalent_scorer),
    ("S4", s4_keyerror_messages_match),
    ("S5", s5_value_errors),
    ("S6", s6_disk_folder_override),
    ("S7", s7_lazy_does_not_construct),
    ("S8", s8_eager_call_counts),
    ("S9", s9_unload_then_reget),
    ("S10", s10_thread_safety),
    ("S11", s11_info_dict_shape),
    ("S12", s12_feature_names),
    ("S13", s13_score_and_score_batch),
]


for label, fn in SCENARIOS:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        FAIL.append((f"{label}.exception", "no_exception", repr(e)))
        print(f"[{label}.exception] FAIL: scenario raised {type(e).__name__}: {e!r}")


total = len(PASS) + len(FAIL)
print(f"\n{len(PASS)}/{total} PASS")
if FAIL:
    print(f"{len(FAIL)} FAIL — registry implementations DIFFER")
    for name, expected, actual in FAIL:
        print(f"  - {name}: expected={expected!r}  actual={actual!r}")
else:
    print("registry implementations agree")

sys.exit(1 if FAIL else 0)
