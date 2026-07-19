"""Oracle implementation of `silent_real_auc.py` (intentionally divergent style).

Backfills `real_auc` for `is_valid` rows whose `real_auc` is NaN. Silent: never
prints per-expression AUCs (preserves the experimental control where the LLM
must not see ground-truth AUCs in iterative loops).

Style notes — this oracle deliberately uses different idioms from the likely
implementer so a verifier can diff outputs:

  - `df.itertuples(index=True)` rather than `df.iterrows()`.
  - Target column extracted once at the top, never re-fetched per row.
  - `argparse` with explicit `parser.error()` on `--in-place`/`--output` mutex
    rather than `add_mutually_exclusive_group()`.
  - `_to_bool()` helper using explicit `isinstance` dispatch rather than a
    one-liner string-cast.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

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
from _external_scoring import bootstrap_auc_ci, eval_expression, univariate_auc  # noqa: E402


_TRUE_TOKENS = {"true", "1", "t", "yes", "y"}


def _to_bool(v: Any) -> bool:
    """Coerce CSV-roundtripped is_valid to a real bool with explicit dispatch."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, float):
        if math.isnan(v):
            return False
        return bool(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in _TRUE_TOKENS
    if v is None:
        return False
    # last-ditch: trust truthiness
    return bool(v)


def _is_nan_scalar(x: Any) -> bool:
    """True iff x is NaN-like (float NaN, pandas NA, None)."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Silent backfill of real_auc for iterative LLM-GAT CSVs (oracle).",
    )
    p.add_argument("--csv", required=True, type=Path, help="Input CSV path.")
    p.add_argument("--parquet", required=True, type=Path, help="Cohort parquet path.")
    p.add_argument("--target-col", required=True, type=str, help="Binary target column.")
    p.add_argument("--n-bootstrap", type=int, default=0, help="Bootstrap iters; 0 = skip CIs.")
    p.add_argument("--in-place", action="store_true", help="Write back over --csv.")
    p.add_argument("--output", type=Path, default=None, help="Alternative output path.")
    return p


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.in_place and args.output is not None:
        parser.error("--in-place and --output are mutually exclusive")

    if not args.csv.exists():
        print(f"error: missing CSV {args.csv}", file=sys.stderr)
        return 2
    if not args.parquet.exists():
        print(f"error: missing parquet {args.parquet}", file=sys.stderr)
        return 3

    out_path: Path = args.csv if args.in_place else (args.output or args.csv)

    df_csv = pd.read_csv(args.csv)
    df_data = pd.read_parquet(args.parquet)

    if args.target_col not in df_data.columns:
        print(f"error: target column {args.target_col!r} not in parquet", file=sys.stderr)
        return 4

    target_arr: np.ndarray = df_data[args.target_col].to_numpy(dtype=np.float64)

    # Ensure real_auc is float (so NaN comparisons work after CSV roundtrip).
    if "real_auc" not in df_csv.columns:
        df_csv["real_auc"] = np.nan
    df_csv["real_auc"] = pd.to_numeric(df_csv["real_auc"], errors="coerce")

    want_ci = args.n_bootstrap > 0
    if want_ci:
        for col in ("real_auc_ci_low", "real_auc_ci_high"):
            if col not in df_csv.columns:
                df_csv[col] = np.nan
            df_csv[col] = pd.to_numeric(df_csv[col], errors="coerce")

    n_skipped = 0
    n_evaluated = 0
    n_updated = 0

    t_start = time.perf_counter()
    # itertuples is faster + lower-overhead than iterrows; deliberate divergence.
    for row in df_csv.itertuples(index=True):
        if not _to_bool(getattr(row, "is_valid")):
            continue
        if not _is_nan_scalar(getattr(row, "real_auc")):
            n_skipped += 1
            continue
        n_evaluated += 1
        expr = getattr(row, "expression")
        if not isinstance(expr, str) or not expr.strip():
            continue
        values = eval_expression(expr, df_data)
        if values is None:
            continue
        auc = univariate_auc(values, target_arr)
        if math.isnan(auc):
            continue
        df_csv.at[row.Index, "real_auc"] = auc
        n_updated += 1
        if want_ci:
            mean_b, lo, hi = bootstrap_auc_ci(values, target_arr, n_boot=args.n_bootstrap)
            df_csv.at[row.Index, "real_auc_ci_low"] = lo
            df_csv.at[row.Index, "real_auc_ci_high"] = hi

    # Tidy up string-ish columns so they roundtrip cleanly (no "nan" literals).
    for col in ("duplicate_of", "expression"):
        if col in df_csv.columns and df_csv[col].dtype == object:
            df_csv[col] = df_csv[col].fillna("")

    df_csv.to_csv(out_path, index=False)
    elapsed = time.perf_counter() - t_start
    print(
        f"wrote {n_updated} updates in {elapsed:.2f} seconds "
        f"(skipped {n_skipped} already-scored, evaluated {n_evaluated})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
