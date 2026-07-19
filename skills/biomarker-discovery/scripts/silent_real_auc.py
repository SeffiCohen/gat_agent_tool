#!/usr/bin/env python3
"""Silently fill the ``real_auc`` column of an ``iterative_details.csv``.

The biomarker-discovery skill runs Claude through a propose-score-refine loop.
After each iteration it appends scored expressions to ``iterative_details.csv``
with a blank ``real_auc`` column. This script computes the **REAL univariate
AUC** of every newly-proposed (valid) expression on a cohort parquet and writes
it back into the CSV.

INVARIANT — the silent-AUC contract (see ``Code/iterative_llm_gat.py:386``):
the values written to ``real_auc`` are for the CSV (audit / final-report
consumption) only. They MUST NOT be surfaced to Claude in any subsequent
iteration prompt; that is the experimental control. Accordingly this script's
``stdout`` prints only counts and timing — never per-expression AUCs. Progress
and warnings go to ``stderr`` via ``logging``.

Run::

    python silent_real_auc.py \
        --csv    Results/714/iterative_details.csv \
        --parquet Data/MIMIC/preprocessed/714_RheumatoidArthritis_mimic.parquet \
        --target-col icd_714 \
        [--n-bootstrap 0] [--in-place | --output OTHER.csv]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Bootstrap path to the shared scoring helpers in ``Code/_external_scoring.py``.
def _find_repo_root(start: Path) -> Path:
    """Locate the repository root by searching upward for ``Code/_external_scoring.py``.

    Override with the ``GAT_AGENT_TOOL_REPO`` environment variable. This replaces an
    earlier fixed ``parents[4]`` hop that assumed one specific checkout layout.
    """
    import os
    env = os.environ.get("GAT_AGENT_TOOL_REPO")
    if env:
        return Path(env).resolve()
    start = start.resolve()
    for up in [start, *start.parents]:
        if (up / "Code" / "_external_scoring.py").exists():
            return up
    raise SystemExit(
        "Could not locate Code/_external_scoring.py above this script; "
        "set GAT_AGENT_TOOL_REPO to the repository root."
    )


_REPO_ROOT = _find_repo_root(Path(__file__))
sys.path.insert(0, str(_REPO_ROOT / "Code"))
from _external_scoring import (  # noqa: E402  (after sys.path patch)
    bootstrap_auc_ci,
    eval_expression,
    univariate_auc,
)

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("silent_real_auc")


def _coerce_is_valid(series: pd.Series) -> pd.Series:
    """Robust True/False coercion: handles bool, ``"True"``/``"False"`` strings,
    ``1``/``0``, and NaN (treated as False)."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", required=True, type=Path,
                   help="path to iterative_details.csv (will be updated)")
    p.add_argument("--parquet", required=True, type=Path,
                   help="path to cohort parquet with lab_* + target column")
    p.add_argument("--target-col", required=True,
                   help="binary outcome column name (e.g., icd_714)")
    p.add_argument("--n-bootstrap", type=int, default=0,
                   help="if > 0, also write real_auc_ci_low / real_auc_ci_high")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--in-place", action="store_true", default=True,
                     help="overwrite the input CSV (default)")
    out.add_argument("--output", type=Path, default=None,
                     help="write to a different file instead of in-place")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()

    if not args.csv.exists():
        logger.error("CSV not found: %s", args.csv)
        return 2
    if not args.parquet.exists():
        logger.error("parquet not found: %s", args.parquet)
        return 3

    df_csv = pd.read_csv(args.csv)
    logger.info("loaded CSV: %s (%d rows)", args.csv, len(df_csv))

    # Ensure required columns exist; create the output cols if missing.
    for col in ("expression", "is_valid"):
        if col not in df_csv.columns:
            logger.error("CSV missing required column %r; have %s",
                         col, list(df_csv.columns))
            return 4
    if "real_auc" not in df_csv.columns:
        df_csv["real_auc"] = np.nan
    if args.n_bootstrap > 0:
        for c in ("real_auc_ci_low", "real_auc_ci_high"):
            if c not in df_csv.columns:
                df_csv[c] = np.nan

    df_data = pd.read_parquet(args.parquet)
    logger.info("loaded parquet: %s (%d rows, %d cols)",
                args.parquet, len(df_data), len(df_data.columns))

    if args.target_col not in df_data.columns:
        logger.error("target column %r not in parquet; available cols: %s",
                     args.target_col, list(df_data.columns))
        return 4

    target = df_data[args.target_col].to_numpy(dtype=np.float64)

    is_valid = _coerce_is_valid(df_csv["is_valid"])
    needs_score = is_valid & df_csv["real_auc"].isna()
    skipped = int((is_valid & df_csv["real_auc"].notna()).sum())
    todo_idx = list(df_csv.index[needs_score])
    logger.info("rows to score: %d (skipping %d already-scored, %d invalid)",
                len(todo_idx), skipped, int((~is_valid).sum()))

    t0 = time.time()
    updates = 0
    for idx in todo_idx:
        expr = str(df_csv.at[idx, "expression"])
        try:
            values = eval_expression(expr, df_data)
        except Exception as exc:  # asteval should never raise, but be defensive
            logger.warning("eval raised on row %d (%r): %s", idx, expr, exc)
            values = None
        if values is None:
            # Leave real_auc as NaN; do NOT flip is_valid (that is the GAT's verdict).
            continue
        try:
            # `real_auc` is always the POINT AUC on the full data — direct,
            # comparable with baseline CSVs (which report point AUC). Bootstrap
            # mean would be a different statistic (biased upward by the
            # direction-flip in `univariate_auc`).
            auc = univariate_auc(values, target)
            df_csv.at[idx, "real_auc"] = auc
            if args.n_bootstrap > 0:
                _, lo, hi = bootstrap_auc_ci(values, target, n_boot=args.n_bootstrap)
                df_csv.at[idx, "real_auc_ci_low"] = lo
                df_csv.at[idx, "real_auc_ci_high"] = hi
            if np.isnan(auc):
                continue
        except Exception as exc:
            logger.warning("AUC raised on row %d (%r): %s", idx, expr, exc)
            continue
        updates += 1

    out_path = args.output if args.output is not None else args.csv
    df_csv.to_csv(out_path, index=False)
    elapsed = time.time() - t0

    # Stdout line — counts and timing only. NEVER per-expression AUCs.
    print(f"wrote {updates} updates in {elapsed:.2f} seconds "
          f"(skipped {skipped} already-scored, evaluated {len(todo_idx)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
