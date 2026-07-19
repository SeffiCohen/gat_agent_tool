#!/usr/bin/env python3
"""Diff IMPLEMENTER vs ORACLE silent_real_auc implementations.

Both:

  * IMPL   — ``.claude/skills/biomarker-discovery/scripts/silent_real_auc.py``
  * ORACLE — ``.claude/skills/biomarker-discovery/scripts/_silent_real_auc_oracle.py``

are CLI wrappers around ``Code/_external_scoring.{eval_expression,univariate_auc,
bootstrap_auc_ci}`` that fill the ``real_auc`` column of an iterative-loop CSV.
This script drives both through identical inputs and asserts byte-equal
``real_auc`` columns (with a ~1e-9 numeric tolerance for floating-point noise),
identical exit codes on error paths, and identical stdout discipline (the silent
invariant — no per-expression AUCs are allowed to leak to stdout).

Run after both implementations exist::

    python .claude/skills/biomarker-discovery/tests/verify_silent_real_auc.py

Exit 0 = all PASS; exit 1 = any FAIL; exit 2 = blocked (one or both files
missing — verifier cannot run yet).

Style follows ``Code/verify_compare_outputs.py`` and
``gat_agent_tool/tests/verify_registry_equivalence.py``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
SCRIPTS = os.path.join(SKILL_ROOT, "scripts")
IMPL = os.path.join(SCRIPTS, "silent_real_auc.py")
ORACLE = os.path.join(SCRIPTS, "_silent_real_auc_oracle.py")


# ---------------------------------------------------------------------------
# Bail out cleanly if either file is missing — the verifier may have been
# launched before one of the parallel agents finished writing its file.
# ---------------------------------------------------------------------------
_missing = []
if not os.path.exists(IMPL):
    _missing.append(f"IMPLEMENTER ({IMPL})")
if not os.path.exists(ORACLE):
    _missing.append(f"ORACLE ({ORACLE})")
if _missing:
    print("BLOCKED: missing implementations:")
    for m in _missing:
        print(f"  - {m}")
    print("verifier cannot run until both files exist.")
    sys.exit(2)


# ---------------------------------------------------------------------------
# PASS/FAIL harness.
# ---------------------------------------------------------------------------
PASS: list = []
FAIL: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"[{name}] PASS{(' ' + detail) if detail else ''}")
    else:
        FAIL.append((name, detail))
        print(f"[{name}] FAIL {detail}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_test_data(tmp: str, n: int = 100, seed: int = 0) -> str:
    """Build a small parquet with two lab features and one binary target."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "lab_X_last": rng.standard_normal(n),
        "lab_Y_last": rng.standard_normal(n) + 0.5,
        "icd_TEST": (rng.random(n) > 0.7).astype(int),
    })
    p = os.path.join(tmp, "test.parquet")
    df.to_parquet(p)
    return p


def run(script: str, **kwargs) -> subprocess.CompletedProcess:
    """Subprocess-call a script with --kebab-case flags built from kwargs.

    Booleans become bare flags; everything else is stringified after a flag.
    """
    cmd = [sys.executable, script]
    for k, v in kwargs.items():
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            cmd.append(flag)
        elif v is False or v is None:
            continue
        else:
            cmd += [flag, str(v)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def read_real_auc(csv_path: str) -> np.ndarray:
    """Return the ``real_auc`` column as a float array (NaN preserved)."""
    df = pd.read_csv(csv_path)
    return df["real_auc"].to_numpy(dtype=np.float64)


def auc_columns_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """True iff the two real_auc arrays match elementwise (NaN==NaN counts)."""
    if a.shape != b.shape:
        return False
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        return False
    finite = ~a_nan
    if not finite.any():
        return True
    return bool(np.all(np.abs(a[finite] - b[finite]) < tol))


def make_csv(tmp: str, name: str, df: pd.DataFrame) -> str:
    p = os.path.join(tmp, name)
    df.to_csv(p, index=False)
    return p


# AUC-looking float regex used by the silent-invariant check.
_AUC_RE = re.compile(r"\b0\.[5-9]\d{2,}\b")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def s1_empty_real_auc_fill_all() -> None:
    """All five rows are valid+empty -> both implementations fill all five."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet = make_test_data(tmp, n=100, seed=1)
        df = pd.DataFrame({
            "expression": [
                "lab_X_last + lab_Y_last",
                "lab_X_last - lab_Y_last",
                "lab_X_last * lab_Y_last",
                "lab_X_last / lab_Y_last",
                "(lab_X_last + lab_Y_last) * lab_X_last",
            ],
            "is_valid": [True, True, True, True, True],
            "real_auc": [np.nan] * 5,
        })
        csv_impl = make_csv(tmp, "impl.csv", df)
        csv_oracle = make_csv(tmp, "oracle.csv", df)

        r_impl = run(
            IMPL,
            csv=csv_impl, parquet=parquet, target_col="icd_TEST",
        )
        r_oracle = run(
            ORACLE,
            csv=csv_oracle, parquet=parquet, target_col="icd_TEST",
        )

        check("S1.impl_exit_zero", r_impl.returncode == 0,
              f"rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}")
        check("S1.oracle_exit_zero", r_oracle.returncode == 0,
              f"rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}")

        a = read_real_auc(csv_impl)
        b = read_real_auc(csv_oracle)
        check("S1.impl_filled_all", bool(np.all(np.isfinite(a))),
              f"impl real_auc={a}")
        check("S1.oracle_filled_all", bool(np.all(np.isfinite(b))),
              f"oracle real_auc={b}")
        check(
            "S1.real_auc_byte_equal", auc_columns_equal(a, b),
            f"impl={a} oracle={b}",
        )


def s2_mixed_scored_unscored() -> None:
    """Pre-scored row preserved; valid+empty filled; invalid stays NaN."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet = make_test_data(tmp, n=100, seed=2)
        df = pd.DataFrame({
            "expression": [
                "lab_X_last + lab_Y_last",   # already scored
                "lab_X_last * lab_Y_last",   # valid empty -> fill
                "lab_X_last - lab_Y_last",   # valid empty -> fill
                "lab_X_last / lab_Y_last",   # is_valid=False -> stay NaN
            ],
            "is_valid": [True, True, True, False],
            "real_auc": [0.65, np.nan, np.nan, np.nan],
        })
        csv_impl = make_csv(tmp, "impl.csv", df)
        csv_oracle = make_csv(tmp, "oracle.csv", df)

        r_impl = run(
            IMPL,
            csv=csv_impl, parquet=parquet, target_col="icd_TEST",
        )
        r_oracle = run(
            ORACLE,
            csv=csv_oracle, parquet=parquet, target_col="icd_TEST",
        )

        check("S2.impl_exit_zero", r_impl.returncode == 0,
              f"rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}")
        check("S2.oracle_exit_zero", r_oracle.returncode == 0,
              f"rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}")

        a = read_real_auc(csv_impl)
        b = read_real_auc(csv_oracle)

        # Row 0: preserved at 0.65 in both.
        check(
            "S2.impl_row0_preserved", abs(a[0] - 0.65) < 1e-12,
            f"impl row0={a[0]}",
        )
        check(
            "S2.oracle_row0_preserved", abs(b[0] - 0.65) < 1e-12,
            f"oracle row0={b[0]}",
        )

        # Rows 1, 2: filled (finite) in both.
        check("S2.impl_row1_filled", bool(np.isfinite(a[1])),
              f"impl row1={a[1]}")
        check("S2.impl_row2_filled", bool(np.isfinite(a[2])),
              f"impl row2={a[2]}")
        check("S2.oracle_row1_filled", bool(np.isfinite(b[1])),
              f"oracle row1={b[1]}")
        check("S2.oracle_row2_filled", bool(np.isfinite(b[2])),
              f"oracle row2={b[2]}")

        # Row 3: still NaN in both.
        check("S2.impl_row3_nan", bool(np.isnan(a[3])),
              f"impl row3={a[3]}")
        check("S2.oracle_row3_nan", bool(np.isnan(b[3])),
              f"oracle row3={b[3]}")

        # Cross-impl agreement on the filled rows.
        check(
            "S2.rows12_byte_equal",
            abs(a[1] - b[1]) < 1e-9 and abs(a[2] - b[2]) < 1e-9,
            f"impl={a[1:3]} oracle={b[1:3]}",
        )

        # Whole-column equality.
        check(
            "S2.real_auc_byte_equal", auc_columns_equal(a, b),
            f"impl={a} oracle={b}",
        )


def s3_boolean_coercion_string() -> None:
    """``is_valid`` arrives as strings — both implementations must coerce."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet = make_test_data(tmp, n=100, seed=3)
        df = pd.DataFrame({
            "expression": [
                "lab_X_last + lab_Y_last",
                "lab_X_last * lab_Y_last",
                "lab_X_last - lab_Y_last",
                "lab_X_last / lab_Y_last",
            ],
            "is_valid": ["True", "True", "True", "False"],
            "real_auc": [0.65, np.nan, np.nan, np.nan],
        })
        csv_impl = make_csv(tmp, "impl.csv", df)
        csv_oracle = make_csv(tmp, "oracle.csv", df)

        r_impl = run(
            IMPL,
            csv=csv_impl, parquet=parquet, target_col="icd_TEST",
        )
        r_oracle = run(
            ORACLE,
            csv=csv_oracle, parquet=parquet, target_col="icd_TEST",
        )

        check("S3.impl_exit_zero", r_impl.returncode == 0,
              f"rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}")
        check("S3.oracle_exit_zero", r_oracle.returncode == 0,
              f"rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}")

        a = read_real_auc(csv_impl)
        b = read_real_auc(csv_oracle)

        check("S3.impl_row3_nan_string_false", bool(np.isnan(a[3])),
              f"impl row3={a[3]} (string 'False' should be invalid)")
        check("S3.oracle_row3_nan_string_false", bool(np.isnan(b[3])),
              f"oracle row3={b[3]} (string 'False' should be invalid)")
        check("S3.impl_row1_filled_string_true", bool(np.isfinite(a[1])),
              f"impl row1={a[1]} (string 'True' should be valid)")
        check("S3.oracle_row1_filled_string_true", bool(np.isfinite(b[1])),
              f"oracle row1={b[1]} (string 'True' should be valid)")

        check(
            "S3.real_auc_byte_equal", auc_columns_equal(a, b),
            f"impl={a} oracle={b}",
        )


def s4_invalid_expressions_stay_nan() -> None:
    """Gibberish expressions: both should leave real_auc as NaN, exit 0."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet = make_test_data(tmp, n=100, seed=4)
        df = pd.DataFrame({
            "expression": ["@@@", "lab_NOTACOL + 1", "((("],
            "is_valid": [True, True, True],
            "real_auc": [np.nan, np.nan, np.nan],
        })
        csv_impl = make_csv(tmp, "impl.csv", df)
        csv_oracle = make_csv(tmp, "oracle.csv", df)

        r_impl = run(
            IMPL,
            csv=csv_impl, parquet=parquet, target_col="icd_TEST",
        )
        r_oracle = run(
            ORACLE,
            csv=csv_oracle, parquet=parquet, target_col="icd_TEST",
        )

        check("S4.impl_exit_zero", r_impl.returncode == 0,
              f"rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}")
        check("S4.oracle_exit_zero", r_oracle.returncode == 0,
              f"rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}")

        a = read_real_auc(csv_impl)
        b = read_real_auc(csv_oracle)

        check(
            "S4.impl_all_nan", bool(np.all(np.isnan(a))),
            f"impl real_auc={a}",
        )
        check(
            "S4.oracle_all_nan", bool(np.all(np.isnan(b))),
            f"oracle real_auc={b}",
        )
        check(
            "S4.real_auc_byte_equal", auc_columns_equal(a, b),
            f"impl={a} oracle={b}",
        )


def s5_bootstrap_ci_columns() -> None:
    """``--n-bootstrap=50`` adds CI columns; mean and CI must agree."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet = make_test_data(tmp, n=200, seed=5)
        df = pd.DataFrame({
            "expression": [
                "lab_X_last + lab_Y_last",
                "lab_X_last * lab_Y_last",
                "lab_X_last - lab_Y_last",
            ],
            "is_valid": [True, True, True],
            "real_auc": [np.nan, np.nan, np.nan],
        })
        csv_impl = make_csv(tmp, "impl.csv", df)
        csv_oracle = make_csv(tmp, "oracle.csv", df)

        r_impl = run(
            IMPL,
            csv=csv_impl, parquet=parquet, target_col="icd_TEST",
            n_bootstrap=50,
        )
        r_oracle = run(
            ORACLE,
            csv=csv_oracle, parquet=parquet, target_col="icd_TEST",
            n_bootstrap=50,
        )

        check("S5.impl_exit_zero", r_impl.returncode == 0,
              f"rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}")
        check("S5.oracle_exit_zero", r_oracle.returncode == 0,
              f"rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}")

        df_impl = pd.read_csv(csv_impl)
        df_oracle = pd.read_csv(csv_oracle)

        for c in ("real_auc_ci_low", "real_auc_ci_high"):
            check(
                f"S5.impl_has_{c}", c in df_impl.columns,
                f"impl columns: {list(df_impl.columns)}",
            )
            check(
                f"S5.oracle_has_{c}", c in df_oracle.columns,
                f"oracle columns: {list(df_oracle.columns)}",
            )

        a = df_impl["real_auc"].to_numpy(dtype=np.float64)
        b = df_oracle["real_auc"].to_numpy(dtype=np.float64)
        check(
            "S5.real_auc_byte_equal", auc_columns_equal(a, b),
            f"impl={a} oracle={b}",
        )

        # CI should also match because both default seed=42 in bootstrap_auc_ci.
        if (
            "real_auc_ci_low" in df_impl.columns
            and "real_auc_ci_low" in df_oracle.columns
        ):
            ci_lo_a = df_impl["real_auc_ci_low"].to_numpy(dtype=np.float64)
            ci_lo_b = df_oracle["real_auc_ci_low"].to_numpy(dtype=np.float64)
            check(
                "S5.ci_low_byte_equal", auc_columns_equal(ci_lo_a, ci_lo_b),
                f"impl_ci_low={ci_lo_a} oracle_ci_low={ci_lo_b}",
            )
        if (
            "real_auc_ci_high" in df_impl.columns
            and "real_auc_ci_high" in df_oracle.columns
        ):
            ci_hi_a = df_impl["real_auc_ci_high"].to_numpy(dtype=np.float64)
            ci_hi_b = df_oracle["real_auc_ci_high"].to_numpy(dtype=np.float64)
            check(
                "S5.ci_high_byte_equal", auc_columns_equal(ci_hi_a, ci_hi_b),
                f"impl_ci_high={ci_hi_a} oracle_ci_high={ci_hi_b}",
            )


def s6_exit_codes_match() -> None:
    """Missing CSV -> 2, missing parquet -> 3, missing target column -> 4."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet = make_test_data(tmp, n=100, seed=6)

        # Build a real, valid CSV for the missing-parquet and missing-target tests.
        good_csv = make_csv(tmp, "good.csv", pd.DataFrame({
            "expression": ["lab_X_last + lab_Y_last"],
            "is_valid": [True],
            "real_auc": [np.nan],
        }))

        # Missing CSV.
        ghost = os.path.join(tmp, "does_not_exist.csv")
        r_impl = run(
            IMPL, csv=ghost, parquet=parquet, target_col="icd_TEST",
        )
        r_oracle = run(
            ORACLE, csv=ghost, parquet=parquet, target_col="icd_TEST",
        )
        check(
            "S6.missing_csv_impl_exit_2", r_impl.returncode == 2,
            f"impl rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}",
        )
        check(
            "S6.missing_csv_oracle_exit_2", r_oracle.returncode == 2,
            f"oracle rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}",
        )

        # Missing parquet.
        ghost_pq = os.path.join(tmp, "does_not_exist.parquet")
        r_impl = run(
            IMPL, csv=good_csv, parquet=ghost_pq, target_col="icd_TEST",
        )
        r_oracle = run(
            ORACLE, csv=good_csv, parquet=ghost_pq, target_col="icd_TEST",
        )
        check(
            "S6.missing_parquet_impl_exit_3", r_impl.returncode == 3,
            f"impl rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}",
        )
        check(
            "S6.missing_parquet_oracle_exit_3", r_oracle.returncode == 3,
            f"oracle rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}",
        )

        # Missing target column.
        r_impl = run(
            IMPL, csv=good_csv, parquet=parquet, target_col="icd_NOPE",
        )
        r_oracle = run(
            ORACLE, csv=good_csv, parquet=parquet, target_col="icd_NOPE",
        )
        check(
            "S6.missing_target_impl_exit_4", r_impl.returncode == 4,
            f"impl rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}",
        )
        check(
            "S6.missing_target_oracle_exit_4", r_oracle.returncode == 4,
            f"oracle rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}",
        )


def s7_silent_stdout_invariant() -> None:
    """No AUC-looking float may leak to stdout — the silent invariant."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet = make_test_data(tmp, n=200, seed=7)
        df = pd.DataFrame({
            "expression": [
                "lab_X_last + lab_Y_last",
                "lab_X_last * lab_Y_last",
                "lab_X_last - lab_Y_last",
                "lab_X_last / lab_Y_last",
            ],
            "is_valid": [True, True, True, True],
            "real_auc": [np.nan] * 4,
        })
        csv_impl = make_csv(tmp, "impl.csv", df)
        csv_oracle = make_csv(tmp, "oracle.csv", df)

        r_impl = run(
            IMPL,
            csv=csv_impl, parquet=parquet, target_col="icd_TEST",
        )
        r_oracle = run(
            ORACLE,
            csv=csv_oracle, parquet=parquet, target_col="icd_TEST",
        )

        impl_lines = (r_impl.stdout or "").splitlines()
        oracle_lines = (r_oracle.stdout or "").splitlines()
        impl_leaks = [ln for ln in impl_lines if _AUC_RE.search(ln)]
        oracle_leaks = [ln for ln in oracle_lines if _AUC_RE.search(ln)]

        check(
            "S7.impl_stdout_no_auc_leak", len(impl_leaks) == 0,
            f"impl leaks: {impl_leaks!r}",
        )
        check(
            "S7.oracle_stdout_no_auc_leak", len(oracle_leaks) == 0,
            f"oracle leaks: {oracle_leaks!r}",
        )

        # Bonus: confirm the summary line shape is present (so we know stdout
        # isn't fully silent — only AUCs are forbidden).
        check(
            "S7.impl_stdout_has_summary",
            any("wrote" in ln and "updates" in ln for ln in impl_lines),
            f"impl stdout: {r_impl.stdout!r}",
        )
        check(
            "S7.oracle_stdout_has_summary",
            any("wrote" in ln and "updates" in ln for ln in oracle_lines),
            f"oracle stdout: {r_oracle.stdout!r}",
        )


def s8_output_vs_in_place() -> None:
    """``--output`` must not touch the input CSV; result equal to in-place."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet = make_test_data(tmp, n=100, seed=8)
        df = pd.DataFrame({
            "expression": [
                "lab_X_last + lab_Y_last",
                "lab_X_last * lab_Y_last",
                "lab_X_last - lab_Y_last",
            ],
            "is_valid": [True, True, True],
            "real_auc": [np.nan, np.nan, np.nan],
        })
        csv_impl = make_csv(tmp, "impl_in.csv", df)
        csv_oracle = make_csv(tmp, "oracle_in.csv", df)
        out_impl = os.path.join(tmp, "impl_out.csv")
        out_oracle = os.path.join(tmp, "oracle_out.csv")

        # Snapshot input bytes for the don't-touch check.
        input_bytes_impl = Path(csv_impl).read_bytes()
        input_bytes_oracle = Path(csv_oracle).read_bytes()

        r_impl = run(
            IMPL,
            csv=csv_impl, parquet=parquet, target_col="icd_TEST",
            output=out_impl,
        )
        r_oracle = run(
            ORACLE,
            csv=csv_oracle, parquet=parquet, target_col="icd_TEST",
            output=out_oracle,
        )

        check("S8.impl_exit_zero", r_impl.returncode == 0,
              f"rc={r_impl.returncode} stderr={r_impl.stderr[-200:]!r}")
        check("S8.oracle_exit_zero", r_oracle.returncode == 0,
              f"rc={r_oracle.returncode} stderr={r_oracle.stderr[-200:]!r}")

        # Inputs untouched.
        check(
            "S8.impl_input_untouched",
            Path(csv_impl).read_bytes() == input_bytes_impl,
            "impl mutated its input CSV despite --output",
        )
        check(
            "S8.oracle_input_untouched",
            Path(csv_oracle).read_bytes() == input_bytes_oracle,
            "oracle mutated its input CSV despite --output",
        )

        # Output paths exist and contain finite AUCs.
        check("S8.impl_output_exists", os.path.exists(out_impl),
              f"missing: {out_impl}")
        check("S8.oracle_output_exists", os.path.exists(out_oracle),
              f"missing: {out_oracle}")

        if os.path.exists(out_impl) and os.path.exists(out_oracle):
            a = read_real_auc(out_impl)
            b = read_real_auc(out_oracle)
            check(
                "S8.real_auc_byte_equal", auc_columns_equal(a, b),
                f"impl={a} oracle={b}",
            )


# ---------------------------------------------------------------------------
# Run all scenarios.
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("S1", s1_empty_real_auc_fill_all),
    ("S2", s2_mixed_scored_unscored),
    ("S3", s3_boolean_coercion_string),
    ("S4", s4_invalid_expressions_stay_nan),
    ("S5", s5_bootstrap_ci_columns),
    ("S6", s6_exit_codes_match),
    ("S7", s7_silent_stdout_invariant),
    ("S8", s8_output_vs_in_place),
]

for label, fn in SCENARIOS:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        FAIL.append((f"{label}.exception", repr(exc)))
        print(f"[{label}.exception] FAIL: scenario raised "
              f"{type(exc).__name__}: {exc!r}")


total = len(PASS) + len(FAIL)
print(f"\n{len(PASS)}/{total} PASS")
if FAIL:
    print(f"{len(FAIL)} FAIL — silent_real_auc implementations DIFFER")
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
else:
    print("silent_real_auc implementations agree")

sys.exit(1 if FAIL else 0)
