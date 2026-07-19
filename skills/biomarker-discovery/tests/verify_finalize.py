#!/usr/bin/env python3
"""Diff IMPLEMENTER vs ORACLE finalize implementations.

Both:

  * IMPL   — ``.claude/skills/biomarker-discovery/scripts/finalize.py``
  * ORACLE — ``.claude/skills/biomarker-discovery/scripts/_finalize_oracle.py``

are CLI wrappers that take an ``iterative_details.csv``, a cohort parquet, and
a baseline directory of third-party LLM-tool eval CSVs, and emit a
``final_report.json`` (machine-readable summary) plus ``final_report.md``
(human-readable summary). The PASS/FAIL decision compares the **top expression
by GAT score**, re-evaluated against the parquet, to the best baseline AUC
across all baseline tools.

This script drives both through identical synthetic inputs and asserts:

* matching exit codes per scenario,
* byte-equal load-bearing fields in ``final_report.json`` (with ~1e-9 tolerance
  on the float AUC fields),
* the PASS/FAIL line on stdout matches (modulo whitespace),
* presence of ``final_report.md`` containing the top expression as a literal
  substring,
* the JSON key set is exactly the documented contract,
* graceful handling of edge cases (empty baselines, all-invalid CSV, missing
  target, NaN top_real_auc that needs fresh recomputation).

Run after both implementations exist::

    python .claude/skills/biomarker-discovery/tests/verify_finalize.py

Exit 0 = all PASS; exit 1 = any FAIL; exit 2 = blocked (one or both files
missing — verifier cannot run yet).

Style follows ``Code/verify_orchestrator_helpers.py``,
``gat_agent_tool/tests/verify_registry_equivalence.py``, and
``.claude/skills/biomarker-discovery/tests/verify_silent_real_auc.py``.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
SCRIPTS = os.path.join(SKILL_ROOT, "scripts")
IMPL = os.path.join(SCRIPTS, "finalize.py")
ORACLE = os.path.join(SCRIPTS, "_finalize_oracle.py")


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
# Documented contract — the exact key set the JSON must contain. The first
# block is the original spec; the second block is the reproducibility extension
# (paths + seed + variant + n_invalid + bootstrap mean) that both impl and
# oracle now emit so a single JSON file is enough to reproduce the run.
EXPECTED_JSON_KEYS = {
    "disease_id", "disease_name", "cohort", "timestamp",
    "top_expression", "top_gat_score", "top_real_auc",
    "ci_low", "ci_high", "n_bootstrap",
    "n_total", "n_valid",
    "baselines", "baseline_best", "delta", "passed",
    "iterations", "pop_per_iter",
    # Reproducibility extension:
    "n_invalid", "random_seed", "target_col",
    "cohort_parquet", "gat_checkpoint", "checkpoint_variant",
    "ci_mean_bootstrap",
}


def make_parquet(tmp: str, n: int = 200, seed: int = 0) -> str:
    """Build a small parquet with two lab features and one binary target."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "lab_X_last": rng.standard_normal(n),
        "lab_Y_last": rng.standard_normal(n) + 0.3,
        "icd_TEST": (rng.random(n) > 0.7).astype(int),
    })
    p = os.path.join(tmp, "test.parquet")
    df.to_parquet(p)
    return p


def make_csv(tmp: str, name: str, df: pd.DataFrame) -> str:
    p = os.path.join(tmp, name)
    df.to_csv(p, index=False)
    return p


def make_baselines(
    bdir: str,
    *,
    gpt_max: float = 0.55,
    gemini_max: float = 0.50,
    scispace_max: float = 0.60,
) -> None:
    """Write three eval-detail CSVs into ``bdir`` with the given max AUCs."""
    os.makedirs(bdir, exist_ok=True)
    pd.DataFrame({
        "expression_id": [0, 1],
        "expression": ["e0", "e1"],
        "univariate_auc": [0.50, gpt_max],
    }).to_csv(os.path.join(bdir, "GPT_eval_details.csv"), index=False)
    pd.DataFrame({
        "expression_id": [0],
        "expression": ["e0"],
        "univariate_auc": [gemini_max],
    }).to_csv(os.path.join(bdir, "Gemini_eval_details.csv"), index=False)
    pd.DataFrame({
        "expression_id": [0, 1],
        "expression": ["e0", "e1"],
        "univariate_auc": [scispace_max, 0.40],
    }).to_csv(os.path.join(bdir, "SciSpace_eval_details.csv"), index=False)


def make_inputs(
    tmp: str,
    *,
    parquet_seed: int = 0,
    gpt_max: float = 0.55,
    gemini_max: float = 0.50,
    scispace_max: float = 0.60,
    csv_df: pd.DataFrame | None = None,
    skip_baselines: bool = False,
) -> dict:
    """Build a synthetic input bundle — CSV, parquet, baseline dir, out dir."""
    pq = make_parquet(tmp, n=200, seed=parquet_seed)

    if csv_df is None:
        csv_df = pd.DataFrame({
            "iter": [1, 1, 2, 2, 3],
            "rank_in_iter": [0, 1, 0, 1, 0],
            "expression": [
                "lab_X_last + lab_Y_last",
                "lab_X_last",
                "lab_Y_last - lab_X_last",
                "lab_X_last * lab_Y_last",
                "foo",
            ],
            "gat_score": [0.85, 0.65, 0.78, 0.72, np.nan],
            "real_auc": [0.62, 0.55, 0.59, 0.58, np.nan],
            "is_valid": [True, True, True, True, False],
            "duplicate_of": ["", "", "", "", ""],
        })
    csv_path = make_csv(tmp, "iterative_details.csv", csv_df)

    bdir = os.path.join(tmp, "baselines")
    if skip_baselines:
        os.makedirs(bdir, exist_ok=True)
    else:
        make_baselines(
            bdir,
            gpt_max=gpt_max, gemini_max=gemini_max, scispace_max=scispace_max,
        )

    out = os.path.join(tmp, "out")
    os.makedirs(out, exist_ok=True)

    return {
        "csv": csv_path,
        "parquet": pq,
        "baseline_dir": bdir,
        "out_dir": out,
    }


def run_finalize(script: str, paths: dict, **extra) -> subprocess.CompletedProcess:
    """Invoke a finalize script on a prepared input bundle."""
    args = [
        "--csv", paths["csv"],
        "--parquet", paths["parquet"],
        "--target-col", "icd_TEST",
        "--baseline-dir", paths["baseline_dir"],
        "--out-dir", paths["out_dir"],
        "--disease-id", "714",
        "--disease-name", "RheumatoidArthritis",
        "--cohort", "mimic",
        "--n-bootstrap", "100",
        "--random-seed", "42",
    ]
    for k, v in extra.items():
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            args.append(flag)
        elif v is False or v is None:
            continue
        else:
            args += [flag, str(v)]
    return subprocess.run(
        [sys.executable, script] + args,
        capture_output=True, text=True, timeout=120,
    )


def load_report(out_dir: str) -> dict | None:
    p = os.path.join(out_dir, "final_report.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def normalize_pass_fail_line(stdout: str) -> str | None:
    """Extract the PASS/FAIL summary line from stdout, normalize whitespace."""
    if not stdout:
        return None
    for line in stdout.splitlines():
        line_stripped = line.strip()
        # Match a line that starts with PASS or FAIL (case-sensitive) — these
        # are the documented summary line tokens.
        if re.match(r"^(PASS|FAIL)\b", line_stripped):
            return re.sub(r"\s+", " ", line_stripped)
    return None


def floats_close(a, b, tol: float = 1e-9) -> bool:
    """True if both NaN, both None, or |a - b| < tol after coercion to float."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(af) and math.isnan(bf):
        return True
    if math.isnan(af) or math.isnan(bf):
        return False
    return abs(af - bf) < tol


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def s1_pass_case() -> None:
    """PASS case — top expression beats baselines (best baseline = 0.60)."""
    with tempfile.TemporaryDirectory() as impl_tmp, \
         tempfile.TemporaryDirectory() as oracle_tmp:
        # Identical seed => both implementations see identical synthetic data.
        pi = make_inputs(impl_tmp, parquet_seed=1)
        po = make_inputs(oracle_tmp, parquet_seed=1)

        ri = run_finalize(IMPL, pi)
        ro = run_finalize(ORACLE, po)

        check("S1.impl_rc0", ri.returncode == 0,
              f"rc={ri.returncode} stderr={ri.stderr[-200:]!r}")
        check("S1.oracle_rc0", ro.returncode == 0,
              f"rc={ro.returncode} stderr={ro.stderr[-200:]!r}")

        check("S1.exit_codes_match", ri.returncode == ro.returncode,
              f"impl={ri.returncode} oracle={ro.returncode}")

        ji = load_report(pi["out_dir"])
        jo = load_report(po["out_dir"])
        check("S1.impl_json_exists", ji is not None,
              f"missing final_report.json in {pi['out_dir']}")
        check("S1.oracle_json_exists", jo is not None,
              f"missing final_report.json in {po['out_dir']}")
        if ji is None or jo is None:
            return

        # Byte-equal on the discrete fields.
        for k in ("top_expression", "top_gat_score", "baseline_best", "passed"):
            check(
                f"S1.json_{k}_match",
                ji.get(k) == jo.get(k),
                f"impl={ji.get(k)!r} oracle={jo.get(k)!r}",
            )

        # Floating-point: tolerate <1e-9 on the AUC fields.
        for k in ("top_real_auc", "delta", "ci_low", "ci_high"):
            check(
                f"S1.json_{k}_close",
                floats_close(ji.get(k), jo.get(k), tol=1e-9),
                f"impl={ji.get(k)} oracle={jo.get(k)}",
            )

        # PASS/FAIL line on stdout matches.
        impl_line = normalize_pass_fail_line(ri.stdout)
        oracle_line = normalize_pass_fail_line(ro.stdout)
        check("S1.impl_has_pass_fail_line", impl_line is not None,
              f"stdout={ri.stdout!r}")
        check("S1.oracle_has_pass_fail_line", oracle_line is not None,
              f"stdout={ro.stdout!r}")
        if impl_line is not None and oracle_line is not None:
            check(
                "S1.pass_fail_line_match",
                impl_line == oracle_line,
                f"impl={impl_line!r} oracle={oracle_line!r}",
            )

        # final_report.md exists and contains the top expression literally.
        md_impl = os.path.join(pi["out_dir"], "final_report.md")
        md_oracle = os.path.join(po["out_dir"], "final_report.md")
        check("S1.impl_md_exists", os.path.exists(md_impl),
              f"missing: {md_impl}")
        check("S1.oracle_md_exists", os.path.exists(md_oracle),
              f"missing: {md_oracle}")
        if os.path.exists(md_impl) and ji is not None:
            md_text = Path(md_impl).read_text()
            check(
                "S1.impl_md_has_top_expr",
                ji.get("top_expression", "") in md_text,
                f"top_expression {ji.get('top_expression')!r} not in impl md",
            )
        if os.path.exists(md_oracle) and jo is not None:
            md_text = Path(md_oracle).read_text()
            check(
                "S1.oracle_md_has_top_expr",
                jo.get("top_expression", "") in md_text,
                f"top_expression {jo.get('top_expression')!r} not in oracle md",
            )


def s2_fail_case() -> None:
    """FAIL case — best baseline = 0.99 (impossible to beat)."""
    with tempfile.TemporaryDirectory() as impl_tmp, \
         tempfile.TemporaryDirectory() as oracle_tmp:
        pi = make_inputs(impl_tmp, parquet_seed=2, gpt_max=0.99)
        po = make_inputs(oracle_tmp, parquet_seed=2, gpt_max=0.99)

        ri = run_finalize(IMPL, pi)
        ro = run_finalize(ORACLE, po)

        check("S2.impl_rc0", ri.returncode == 0,
              f"rc={ri.returncode} stderr={ri.stderr[-200:]!r}")
        check("S2.oracle_rc0", ro.returncode == 0,
              f"rc={ro.returncode} stderr={ro.stderr[-200:]!r}")

        ji = load_report(pi["out_dir"])
        jo = load_report(po["out_dir"])
        if ji is None or jo is None:
            check("S2.both_jsons_present", False,
                  f"impl={ji is not None} oracle={jo is not None}")
            return

        # Both must report FAIL.
        check(
            "S2.impl_passed_false",
            ji.get("passed") is False,
            f"impl passed={ji.get('passed')!r} (expected False)",
        )
        check(
            "S2.oracle_passed_false",
            jo.get("passed") is False,
            f"oracle passed={jo.get('passed')!r} (expected False)",
        )
        check(
            "S2.passed_match",
            ji.get("passed") == jo.get("passed"),
            f"impl={ji.get('passed')!r} oracle={jo.get('passed')!r}",
        )
        check(
            "S2.baseline_best_match",
            ji.get("baseline_best") == jo.get("baseline_best"),
            f"impl={ji.get('baseline_best')!r} oracle={jo.get('baseline_best')!r}",
        )
        check(
            "S2.delta_close",
            floats_close(ji.get("delta"), jo.get("delta")),
            f"impl_delta={ji.get('delta')} oracle_delta={jo.get('delta')}",
        )

        # PASS/FAIL line should both be FAIL.
        impl_line = normalize_pass_fail_line(ri.stdout)
        oracle_line = normalize_pass_fail_line(ro.stdout)
        if impl_line is not None and oracle_line is not None:
            check(
                "S2.pass_fail_line_match",
                impl_line == oracle_line,
                f"impl={impl_line!r} oracle={oracle_line!r}",
            )


def s3_empty_baselines() -> None:
    """Empty baseline dir — both should treat baseline_best as NaN, PASS auto."""
    with tempfile.TemporaryDirectory() as impl_tmp, \
         tempfile.TemporaryDirectory() as oracle_tmp:
        pi = make_inputs(impl_tmp, parquet_seed=3, skip_baselines=True)
        po = make_inputs(oracle_tmp, parquet_seed=3, skip_baselines=True)

        ri = run_finalize(IMPL, pi)
        ro = run_finalize(ORACLE, po)

        check("S3.impl_rc0", ri.returncode == 0,
              f"rc={ri.returncode} stderr={ri.stderr[-200:]!r}")
        check("S3.oracle_rc0", ro.returncode == 0,
              f"rc={ro.returncode} stderr={ro.stderr[-200:]!r}")

        ji = load_report(pi["out_dir"])
        jo = load_report(po["out_dir"])
        if ji is None or jo is None:
            check("S3.both_jsons_present", False,
                  f"impl={ji is not None} oracle={jo is not None}")
            return

        # baseline_best = NaN/None when no baselines available.
        impl_bb = ji.get("baseline_best")
        oracle_bb = jo.get("baseline_best")

        def _is_nan_or_none(x) -> bool:
            if x is None:
                return True
            try:
                return math.isnan(float(x))
            except (TypeError, ValueError):
                return False

        check(
            "S3.impl_baseline_best_nan_or_none", _is_nan_or_none(impl_bb),
            f"impl baseline_best={impl_bb!r} (expected NaN/None)",
        )
        check(
            "S3.oracle_baseline_best_nan_or_none", _is_nan_or_none(oracle_bb),
            f"oracle baseline_best={oracle_bb!r} (expected NaN/None)",
        )

        # baselines dict should be empty in both.
        check(
            "S3.impl_baselines_empty",
            ji.get("baselines") == {},
            f"impl baselines={ji.get('baselines')!r}",
        )
        check(
            "S3.oracle_baselines_empty",
            jo.get("baselines") == {},
            f"oracle baselines={jo.get('baselines')!r}",
        )

        # Both should auto-PASS in absence of baselines.
        check(
            "S3.impl_passed_true", ji.get("passed") is True,
            f"impl passed={ji.get('passed')!r} (expected True with no baselines)",
        )
        check(
            "S3.oracle_passed_true", jo.get("passed") is True,
            f"oracle passed={jo.get('passed')!r} (expected True with no baselines)",
        )

        # JSON key sets agree.
        check(
            "S3.json_keys_match",
            set(ji.keys()) == set(jo.keys()),
            f"impl-keys={sorted(ji.keys())} oracle-keys={sorted(jo.keys())}",
        )


def s4_all_invalid_csv() -> None:
    """No is_valid=True rows — both should exit 5 with 'no valid' in stderr."""
    with tempfile.TemporaryDirectory() as impl_tmp, \
         tempfile.TemporaryDirectory() as oracle_tmp:
        df = pd.DataFrame({
            "iter": [1, 2],
            "rank_in_iter": [0, 0],
            "expression": ["foo", "bar"],
            "gat_score": [np.nan, np.nan],
            "real_auc": [np.nan, np.nan],
            "is_valid": [False, False],
            "duplicate_of": ["", ""],
        })
        pi = make_inputs(impl_tmp, parquet_seed=4, csv_df=df)
        po = make_inputs(oracle_tmp, parquet_seed=4, csv_df=df.copy())

        ri = run_finalize(IMPL, pi)
        ro = run_finalize(ORACLE, po)

        check("S4.impl_exit_5", ri.returncode == 5,
              f"impl rc={ri.returncode} stderr={ri.stderr[-200:]!r}")
        check("S4.oracle_exit_5", ro.returncode == 5,
              f"oracle rc={ro.returncode} stderr={ro.stderr[-200:]!r}")
        check("S4.exit_codes_match", ri.returncode == ro.returncode,
              f"impl={ri.returncode} oracle={ro.returncode}")

        # stderr should mention "no valid expressions" (or similar phrasing).
        impl_err = (ri.stderr or "").lower()
        oracle_err = (ro.stderr or "").lower()
        check(
            "S4.impl_stderr_mentions_no_valid",
            "no valid" in impl_err or "no_valid" in impl_err,
            f"impl stderr: {ri.stderr[-300:]!r}",
        )
        check(
            "S4.oracle_stderr_mentions_no_valid",
            "no valid" in oracle_err or "no_valid" in oracle_err,
            f"oracle stderr: {ro.stderr[-300:]!r}",
        )


def s5_missing_target_column() -> None:
    """Target column absent in parquet — both should exit 4."""
    with tempfile.TemporaryDirectory() as impl_tmp, \
         tempfile.TemporaryDirectory() as oracle_tmp:
        pi = make_inputs(impl_tmp, parquet_seed=5)
        po = make_inputs(oracle_tmp, parquet_seed=5)

        # Override target column to one that doesn't exist in the parquet.
        ri = subprocess.run(
            [sys.executable, IMPL,
             "--csv", pi["csv"], "--parquet", pi["parquet"],
             "--target-col", "icd_DOES_NOT_EXIST",
             "--baseline-dir", pi["baseline_dir"],
             "--out-dir", pi["out_dir"],
             "--disease-id", "714", "--disease-name", "RA",
             "--cohort", "mimic", "--n-bootstrap", "100",
             "--random-seed", "42"],
            capture_output=True, text=True, timeout=120,
        )
        ro = subprocess.run(
            [sys.executable, ORACLE,
             "--csv", po["csv"], "--parquet", po["parquet"],
             "--target-col", "icd_DOES_NOT_EXIST",
             "--baseline-dir", po["baseline_dir"],
             "--out-dir", po["out_dir"],
             "--disease-id", "714", "--disease-name", "RA",
             "--cohort", "mimic", "--n-bootstrap", "100",
             "--random-seed", "42"],
            capture_output=True, text=True, timeout=120,
        )

        check("S5.impl_exit_4", ri.returncode == 4,
              f"impl rc={ri.returncode} stderr={ri.stderr[-200:]!r}")
        check("S5.oracle_exit_4", ro.returncode == 4,
              f"oracle rc={ro.returncode} stderr={ro.stderr[-200:]!r}")
        check("S5.exit_codes_match", ri.returncode == ro.returncode,
              f"impl={ri.returncode} oracle={ro.returncode}")


def s6_nan_real_auc_at_top() -> None:
    """top_by_gat.real_auc=NaN — both must recompute fresh from parquet."""
    with tempfile.TemporaryDirectory() as impl_tmp, \
         tempfile.TemporaryDirectory() as oracle_tmp:
        # Row 0 has the highest gat_score AND a NaN real_auc.
        df = pd.DataFrame({
            "iter": [1, 1, 2, 2],
            "rank_in_iter": [0, 1, 0, 1],
            "expression": [
                "lab_X_last + lab_Y_last",   # top by gat, real_auc=NaN
                "lab_X_last",
                "lab_Y_last - lab_X_last",
                "lab_X_last * lab_Y_last",
            ],
            "gat_score": [0.99, 0.65, 0.78, 0.72],
            "real_auc": [np.nan, 0.55, 0.59, 0.58],
            "is_valid": [True, True, True, True],
            "duplicate_of": ["", "", "", ""],
        })
        pi = make_inputs(impl_tmp, parquet_seed=6, csv_df=df)
        po = make_inputs(oracle_tmp, parquet_seed=6, csv_df=df.copy())

        ri = run_finalize(IMPL, pi)
        ro = run_finalize(ORACLE, po)

        check("S6.impl_rc0", ri.returncode == 0,
              f"rc={ri.returncode} stderr={ri.stderr[-200:]!r}")
        check("S6.oracle_rc0", ro.returncode == 0,
              f"rc={ro.returncode} stderr={ro.stderr[-200:]!r}")

        ji = load_report(pi["out_dir"])
        jo = load_report(po["out_dir"])
        if ji is None or jo is None:
            check("S6.both_jsons_present", False,
                  f"impl={ji is not None} oracle={jo is not None}")
            return

        # Both must pick row 0 ("lab_X_last + lab_Y_last") as top by GAT.
        check(
            "S6.impl_picks_top_by_gat",
            ji.get("top_expression") == "lab_X_last + lab_Y_last",
            f"impl top={ji.get('top_expression')!r}",
        )
        check(
            "S6.oracle_picks_top_by_gat",
            jo.get("top_expression") == "lab_X_last + lab_Y_last",
            f"oracle top={jo.get('top_expression')!r}",
        )
        check(
            "S6.top_expression_match",
            ji.get("top_expression") == jo.get("top_expression"),
            f"impl={ji.get('top_expression')!r} oracle={jo.get('top_expression')!r}",
        )

        # top_real_auc must be a finite, recomputed value (NOT NaN — proves the
        # implementation re-evaluated against the parquet rather than reading the
        # NaN out of the CSV).
        impl_auc = ji.get("top_real_auc")
        oracle_auc = jo.get("top_real_auc")

        def _is_finite_number(x) -> bool:
            if x is None:
                return False
            try:
                return math.isfinite(float(x))
            except (TypeError, ValueError):
                return False

        check(
            "S6.impl_top_real_auc_finite", _is_finite_number(impl_auc),
            f"impl top_real_auc={impl_auc!r} (must be finite, not NaN)",
        )
        check(
            "S6.oracle_top_real_auc_finite", _is_finite_number(oracle_auc),
            f"oracle top_real_auc={oracle_auc!r} (must be finite, not NaN)",
        )
        check(
            "S6.top_real_auc_close",
            floats_close(impl_auc, oracle_auc, tol=1e-9),
            f"impl={impl_auc} oracle={oracle_auc}",
        )


def s7_json_keys_present() -> None:
    """JSON key set must equal the documented contract; timestamp non-empty."""
    with tempfile.TemporaryDirectory() as impl_tmp, \
         tempfile.TemporaryDirectory() as oracle_tmp:
        pi = make_inputs(impl_tmp, parquet_seed=7)
        po = make_inputs(oracle_tmp, parquet_seed=7)

        # ``iterations`` / ``pop_per_iter`` are inferred from the CSV's
        # ``iter`` column — neither implementation takes them as CLI flags.
        ri = run_finalize(IMPL, pi)
        ro = run_finalize(ORACLE, po)

        check("S7.impl_rc0", ri.returncode == 0,
              f"rc={ri.returncode} stderr={ri.stderr[-200:]!r}")
        check("S7.oracle_rc0", ro.returncode == 0,
              f"rc={ro.returncode} stderr={ro.stderr[-200:]!r}")

        ji = load_report(pi["out_dir"])
        jo = load_report(po["out_dir"])
        if ji is None or jo is None:
            check("S7.both_jsons_present", False,
                  f"impl={ji is not None} oracle={jo is not None}")
            return

        impl_keys = set(ji.keys())
        oracle_keys = set(jo.keys())

        # Each implementation independently must match the contract.
        check(
            "S7.impl_keyset_exact",
            impl_keys == EXPECTED_JSON_KEYS,
            f"missing={EXPECTED_JSON_KEYS - impl_keys!r} "
            f"extra={impl_keys - EXPECTED_JSON_KEYS!r}",
        )
        check(
            "S7.oracle_keyset_exact",
            oracle_keys == EXPECTED_JSON_KEYS,
            f"missing={EXPECTED_JSON_KEYS - oracle_keys!r} "
            f"extra={oracle_keys - EXPECTED_JSON_KEYS!r}",
        )
        # Cross-impl keys agree.
        check(
            "S7.json_keysets_agree",
            impl_keys == oracle_keys,
            f"impl-only={impl_keys - oracle_keys!r} "
            f"oracle-only={oracle_keys - impl_keys!r}",
        )

        # Timestamp must be a non-empty string in both (allowed to differ since
        # it's wall-clock).
        ts_impl = ji.get("timestamp")
        ts_oracle = jo.get("timestamp")
        check(
            "S7.impl_timestamp_nonempty_string",
            isinstance(ts_impl, str) and len(ts_impl) > 0,
            f"impl timestamp={ts_impl!r}",
        )
        check(
            "S7.oracle_timestamp_nonempty_string",
            isinstance(ts_oracle, str) and len(ts_oracle) > 0,
            f"oracle timestamp={ts_oracle!r}",
        )


def s8_md_contains_top_expression() -> None:
    """final_report.md must exist in both and contain the top_expression."""
    with tempfile.TemporaryDirectory() as impl_tmp, \
         tempfile.TemporaryDirectory() as oracle_tmp:
        pi = make_inputs(impl_tmp, parquet_seed=8)
        po = make_inputs(oracle_tmp, parquet_seed=8)

        ri = run_finalize(IMPL, pi)
        ro = run_finalize(ORACLE, po)

        check("S8.impl_rc0", ri.returncode == 0,
              f"rc={ri.returncode} stderr={ri.stderr[-200:]!r}")
        check("S8.oracle_rc0", ro.returncode == 0,
              f"rc={ro.returncode} stderr={ro.stderr[-200:]!r}")

        md_impl = os.path.join(pi["out_dir"], "final_report.md")
        md_oracle = os.path.join(po["out_dir"], "final_report.md")
        check("S8.impl_md_exists", os.path.exists(md_impl),
              f"missing: {md_impl}")
        check("S8.oracle_md_exists", os.path.exists(md_oracle),
              f"missing: {md_oracle}")

        ji = load_report(pi["out_dir"])
        jo = load_report(po["out_dir"])
        if ji is not None and os.path.exists(md_impl):
            md_text = Path(md_impl).read_text()
            top = ji.get("top_expression", "")
            check(
                "S8.impl_md_contains_top_expr",
                isinstance(top, str) and top != "" and top in md_text,
                f"impl top={top!r} not in md (md head={md_text[:200]!r})",
            )
        if jo is not None and os.path.exists(md_oracle):
            md_text = Path(md_oracle).read_text()
            top = jo.get("top_expression", "")
            check(
                "S8.oracle_md_contains_top_expr",
                isinstance(top, str) and top != "" and top in md_text,
                f"oracle top={top!r} not in md (md head={md_text[:200]!r})",
            )


# ---------------------------------------------------------------------------
# Run all scenarios.
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("S1", s1_pass_case),
    ("S2", s2_fail_case),
    ("S3", s3_empty_baselines),
    ("S4", s4_all_invalid_csv),
    ("S5", s5_missing_target_column),
    ("S6", s6_nan_real_auc_at_top),
    ("S7", s7_json_keys_present),
    ("S8", s8_md_contains_top_expression),
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
    print(f"{len(FAIL)} FAIL — finalize implementations DIFFER")
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
else:
    print("finalize implementations agree")

sys.exit(1 if FAIL else 0)
