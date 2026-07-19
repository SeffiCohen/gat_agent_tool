"""Oracle implementation of `finalize.py` (intentionally divergent style).

Produces ``final_report.md`` and ``final_report.json`` for the iterative
LLM-GAT loop, plus the one-line PASS/FAIL stdout verdict comparing the top
GAT-scored expression against external-method (GPT/Gemini/SciSpace) baselines.

Style notes — deliberately divergent idioms vs the likely implementer so a
verifier can diff outputs while exercising different code paths:

  - Top-by-GAT pick uses ``df.loc[df["gat_score"].idxmax()]`` (idxmax-based,
    not ``sort_values(...).iloc[0]``).
  - Baseline CSV enumeration uses ``pathlib.Path.glob`` (not ``glob.glob``).
  - Template placeholders are assembled in named groups
    (``pl_overview``, ``pl_baselines``, ``pl_repro``, ...) then merged.
  - NaN guards use both ``math.isnan`` and ``pd.isna`` (belt-and-suspenders).
  - JSON output is built as an ``OrderedDict`` to lock field order, serialised
    with a ``default=`` callable that maps NaN-likes to ``None``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

_HERE = Path(__file__).resolve().parent
_TEMPLATE_PATH = _HERE.parent / "assets" / "template_final_report.md"


def _bootstrap_mean(values, target, n_boot: int, seed: int) -> float:
    """Mean AUC across n_boot percentile-bootstrap resamples — separate from
    the point AUC reported in `top_real_auc`. Returns NaN on degenerate input."""
    mean, _, _ = bootstrap_auc_ci(values, target, n_boot=n_boot, seed=seed)
    return mean


# ---------------------------------------------------------------------------
# NaN-aware helpers
# ---------------------------------------------------------------------------
def _is_nan(x: Any) -> bool:
    """True iff x is NaN-like (float NaN, pd.NA, None). Belt-and-suspenders."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """Coerce CSV-roundtripped is_valid (bool / "True" / 1 / etc.) to bool."""
    return s.astype(str).str.strip().str.lower().isin(("true", "1"))


def _json_default(o: Any) -> Any:
    """JSON serialiser fallback: NaN-likes -> None; numpy scalars unwrapped."""
    if _is_nan(o):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"unserialisable {type(o).__name__}")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render final_report.{md,json} for the iterative LLM-GAT loop (oracle).",
    )
    p.add_argument("--csv", required=True, type=Path, help="iterative_details.csv")
    p.add_argument("--parquet", required=True, type=Path, help="cohort parquet")
    p.add_argument("--target-col", required=True, type=str, help="binary target col")
    p.add_argument("--baseline-dir", required=True, type=Path,
                   help="dir of <Tool>_eval_details.csv baseline files")
    p.add_argument("--out-dir", required=True, type=Path, help="output dir")
    p.add_argument("--disease-id", required=True, type=str)
    p.add_argument("--disease-name", required=True, type=str)
    p.add_argument("--cohort", required=True, type=str)
    p.add_argument("--gat-checkpoint", type=Path, default=None)
    p.add_argument("--checkpoint-variant", type=str, default="best_by_loss")
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    return p


# ---------------------------------------------------------------------------
# Baseline enumeration (pathlib.Path.glob — divergent from glob.glob)
# ---------------------------------------------------------------------------
def _enumerate_baselines(baseline_dir: Path) -> List[Path]:
    """Return *_eval_details.csv files, excluding training_rank and gat_ranked_*."""
    found: List[Path] = []
    for p in sorted(baseline_dir.glob("*_eval_details.csv")):
        name = p.name
        if "training_rank" in name:
            continue
        if name.startswith("gat_ranked_"):
            continue
        found.append(p)
    return found


def _tool_name_from_filename(p: Path) -> str:
    """``GPTDeepResearch_eval_details.csv`` -> ``GPTDeepResearch``."""
    stem = p.stem
    suffix = "_eval_details"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _baseline_summary(baseline_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Read each baseline CSV; return per-tool {best_auc, n_expr} + overall best."""
    summary: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    best_overall = float("-inf")
    for csv_path in _enumerate_baselines(baseline_dir):
        tool = _tool_name_from_filename(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            summary[tool] = {"best_auc": float("nan"), "n_expr": 0}
            continue
        if "univariate_auc" not in df.columns or len(df) == 0:
            summary[tool] = {"best_auc": float("nan"), "n_expr": int(len(df))}
            continue
        col = pd.to_numeric(df["univariate_auc"], errors="coerce")
        finite = col[~col.isna()]
        if len(finite) == 0:
            summary[tool] = {"best_auc": float("nan"), "n_expr": int(len(df))}
            continue
        best = float(finite.max())
        summary[tool] = {"best_auc": best, "n_expr": int(len(df))}
        if best > best_overall:
            best_overall = best
    if math.isinf(best_overall):
        best_overall = float("nan")
    return summary, best_overall


# ---------------------------------------------------------------------------
# Iteration trajectory + top-10 tables
# ---------------------------------------------------------------------------
def _iter_trajectory_rows(df_valid: pd.DataFrame) -> List[Tuple[int, int, float, float]]:
    """Per-iter: (iter, n_new_scored, best_gat_this_iter, best_gat_cumulative)."""
    if "iter" not in df_valid.columns or len(df_valid) == 0:
        return []
    rows: List[Tuple[int, int, float, float]] = []
    cum_best = float("-inf")
    for it_val, sub in df_valid.groupby("iter", sort=True):
        n = int(len(sub))
        gat_col = pd.to_numeric(sub["gat_score"], errors="coerce")
        gat_finite = gat_col.dropna()
        best_iter = float(gat_finite.max()) if len(gat_finite) else float("nan")
        if not math.isnan(best_iter) and best_iter > cum_best:
            cum_best = best_iter
        cum_show = cum_best if not math.isinf(cum_best) else float("nan")
        try:
            it_int = int(it_val)
        except (TypeError, ValueError):
            it_int = -1
        rows.append((it_int, n, best_iter, cum_show))
    return rows


def _top10_rows(df_valid: pd.DataFrame) -> List[Tuple[int, str, float, float, int]]:
    """Top-10 rows by gat_score: (rank, expression, gat, real_auc, iter)."""
    if "gat_score" not in df_valid.columns or len(df_valid) == 0:
        return []
    df_sorted = df_valid.assign(
        _gat_num=pd.to_numeric(df_valid["gat_score"], errors="coerce")
    ).sort_values("_gat_num", ascending=False, na_position="last").head(10)
    out: List[Tuple[int, str, float, float, int]] = []
    for rank, (_, row) in enumerate(df_sorted.iterrows(), start=1):
        expr = str(row.get("expression", "")) if not _is_nan(row.get("expression")) else ""
        gat_v = row.get("_gat_num")
        gat_f = float(gat_v) if not _is_nan(gat_v) else float("nan")
        real_v = row.get("real_auc")
        real_f = float(real_v) if not _is_nan(real_v) else float("nan")
        it_v = row.get("iter")
        try:
            it_i = int(it_v) if not _is_nan(it_v) else -1
        except (TypeError, ValueError):
            it_i = -1
        out.append((rank, expr, gat_f, real_f, it_i))
    return out


def _fmt_top10_table(rows: List[Tuple[int, str, float, float, int]]) -> str:
    if not rows:
        return "| - | _no valid expressions_ | - | - | - |"
    lines = []
    for rank, expr, gat, real_auc, it in rows:
        gat_s = f"{gat:.4f}" if not math.isnan(gat) else "n/a"
        auc_s = f"{real_auc:.4f}" if not math.isnan(real_auc) else "n/a"
        it_s = str(it) if it >= 0 else "-"
        lines.append(f"| {rank} | `{expr}` | {gat_s} | {auc_s} | {it_s} |")
    return "\n".join(lines)


def _fmt_trajectory_table(rows: List[Tuple[int, int, float, float]]) -> str:
    if not rows:
        return "| - | - | - | - |"
    lines = []
    for it, n, best_iter, best_cum in rows:
        bi = f"{best_iter:.4f}" if not math.isnan(best_iter) else "n/a"
        bc = f"{best_cum:.4f}" if not math.isnan(best_cum) else "n/a"
        lines.append(f"| {it} | {n} | {bi} | {bc} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.csv.exists():
        print(f"error: missing CSV {args.csv}", file=sys.stderr)
        return 2
    if not args.parquet.exists():
        print(f"error: missing parquet {args.parquet}", file=sys.stderr)
        return 3

    df_csv = pd.read_csv(args.csv)
    df_data = pd.read_parquet(args.parquet)

    if args.target_col not in df_data.columns:
        print(f"error: target column {args.target_col!r} not in parquet",
              file=sys.stderr)
        return 4

    if not args.baseline_dir.exists() or not args.baseline_dir.is_dir():
        print(f"error: baseline dir not found / empty: {args.baseline_dir}",
              file=sys.stderr)
        return 4

    # Filter to is_valid == True.
    if "is_valid" not in df_csv.columns:
        df_csv["is_valid"] = False
    valid_mask = _coerce_bool_series(df_csv["is_valid"])
    df_valid = df_csv[valid_mask].copy()

    n_total = int(len(df_csv))
    n_valid = int(len(df_valid))
    n_invalid = n_total - n_valid

    if n_valid == 0:
        print("error: no valid expressions in CSV", file=sys.stderr)
        return 5

    # Top by GAT — idxmax-based pick (divergent from sort_values).
    if "gat_score" not in df_valid.columns:
        print("error: gat_score column missing", file=sys.stderr)
        return 5
    gat_num = pd.to_numeric(df_valid["gat_score"], errors="coerce")
    if gat_num.dropna().empty:
        print("error: no valid expressions with finite gat_score", file=sys.stderr)
        return 5
    top_idx = gat_num.idxmax()
    top_row = df_valid.loc[top_idx]
    top_expression = str(top_row["expression"])
    top_gat_score = float(gat_num.loc[top_idx])

    # Real AUC + bootstrap CI on parquet.
    target_arr = df_data[args.target_col].to_numpy(dtype=np.float64)
    values = eval_expression(top_expression, df_data)
    if values is None:
        point_auc = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        point_auc = univariate_auc(values, target_arr)
        if args.n_bootstrap and args.n_bootstrap > 0:
            _, ci_low, ci_high = bootstrap_auc_ci(
                values, target_arr,
                n_boot=args.n_bootstrap, seed=args.random_seed,
            )
        else:
            ci_low = float("nan")
            ci_high = float("nan")

    # Baselines.
    baseline_summary, baseline_best = _baseline_summary(args.baseline_dir)
    delta = (point_auc - baseline_best) if (
        not math.isnan(point_auc) and not math.isnan(baseline_best)
    ) else float("nan")

    # PASS / FAIL — strict greater-than. If baseline_best is NaN (no
    # baselines on disk), auto-PASS provided we have a valid point_auc.
    if not math.isnan(point_auc) and math.isnan(baseline_best):
        passed = True
    else:
        passed = (
            not math.isnan(point_auc)
            and not math.isnan(baseline_best)
            and point_auc > baseline_best
        )

    # Iteration / population stats from CSV.
    iterations: Optional[int] = None
    if "iter" in df_csv.columns:
        it_num = pd.to_numeric(df_csv["iter"], errors="coerce").dropna()
        if len(it_num):
            iterations = int(it_num.max()) + 1
    pop_per_iter: Optional[int] = None
    if "iter" in df_csv.columns and len(df_csv):
        try:
            counts = df_csv.groupby("iter").size()
            if len(counts):
                pop_per_iter = int(counts.iloc[0])
        except Exception:
            pop_per_iter = None

    # Render report.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    template_text = _TEMPLATE_PATH.read_text(encoding="utf-8")

    # Tool-by-tool placeholders. Manuscript uses three known tools.
    def _baseline_get(prefix: str) -> Tuple[float, int]:
        for tool, info in baseline_summary.items():
            if tool.lower().startswith(prefix.lower()):
                return float(info.get("best_auc", float("nan"))), int(info.get("n_expr", 0))
        return float("nan"), 0

    auc_gpt, n_gpt = _baseline_get("GPT")
    auc_gem, n_gem = _baseline_get("Gemini")
    auc_sci, n_sci = _baseline_get("SciSpace")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pass_marker = "PASS" if passed else "FAIL"

    # Placeholder groups (deliberate clarity-by-grouping divergence).
    pl_overview: Dict[str, Any] = {
        "disease_name": args.disease_name,
        "disease_id": args.disease_id,
        "timestamp": timestamp,
        "cohort": args.cohort,
        "iterations": iterations if iterations is not None else "n/a",
        "pop_per_iter": pop_per_iter if pop_per_iter is not None else "n/a",
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "pass_fail_marker": pass_marker,
        "top_expression": top_expression,
    }
    pl_metrics: Dict[str, Any] = {
        "top_gat_score": top_gat_score if not math.isnan(top_gat_score) else float("nan"),
        "top_real_auc": point_auc if not math.isnan(point_auc) else float("nan"),
        "n_boot": args.n_bootstrap,
        "ci_low": ci_low if not math.isnan(ci_low) else float("nan"),
        "ci_high": ci_high if not math.isnan(ci_high) else float("nan"),
    }
    pl_baselines: Dict[str, Any] = {
        "auc_gpt": auc_gpt, "n_gpt": n_gpt,
        "auc_gem": auc_gem, "n_gem": n_gem,
        "auc_sci": auc_sci, "n_sci": n_sci,
        "baseline_best": baseline_best if not math.isnan(baseline_best) else float("nan"),
        "delta": delta if not math.isnan(delta) else float("nan"),
    }
    pl_tables: Dict[str, Any] = {
        "top10_table": _fmt_top10_table(_top10_rows(df_valid)),
        "trajectory_table": _fmt_trajectory_table(_iter_trajectory_rows(df_valid)),
    }
    pl_repro: Dict[str, Any] = {
        "gat_checkpoint": str(args.gat_checkpoint) if args.gat_checkpoint else "n/a",
        "cohort_parquet": str(args.parquet),
        "target_col": args.target_col,
        "checkpoint_variant": args.checkpoint_variant,
        "n_features_used": int(sum(1 for c in df_data.columns if c.startswith("lab_"))),
        "random_seed": args.random_seed,
    }
    pl: Dict[str, Any] = {**pl_overview, **pl_metrics, **pl_baselines,
                          **pl_tables, **pl_repro}

    # Render the markdown — replace any NaNs with "n/a" so the .4f format works
    # only on real floats. We do this by walking known float keys.
    for k in ("top_gat_score", "top_real_auc", "ci_low", "ci_high",
              "auc_gpt", "auc_gem", "auc_sci", "baseline_best", "delta"):
        if isinstance(pl[k], float) and math.isnan(pl[k]):
            # Substitute a sentinel that survives .4f formatting via a fallback.
            pl[k] = float("nan")
    # Simple format with a custom Formatter would be cleaner; emulate by
    # try/except per-key substitution. The template uses ``:.4f`` and ``:+.4f``,
    # so NaN values would render as "nan"; tolerate that — both impls produce
    # identical strings since neither tries to mask NaN here.
    md = template_text.format(**pl)

    md_path = args.out_dir / "final_report.md"
    md_path.write_text(md, encoding="utf-8")

    # JSON output — OrderedDict locks insertion order.
    od: "OrderedDict[str, Any]" = OrderedDict()
    od["disease_id"] = args.disease_id
    od["disease_name"] = args.disease_name
    od["cohort"] = args.cohort
    od["timestamp"] = timestamp
    od["top_expression"] = top_expression
    od["top_gat_score"] = top_gat_score
    od["top_real_auc"] = point_auc
    od["ci_low"] = ci_low
    od["ci_high"] = ci_high
    od["n_bootstrap"] = int(args.n_bootstrap)
    od["n_total"] = n_total
    od["n_valid"] = n_valid
    od["baselines"] = {
        tool: {"best_auc": info["best_auc"], "n_expr": info["n_expr"]}
        for tool, info in baseline_summary.items()
    }
    od["baseline_best"] = baseline_best
    od["delta"] = delta
    od["passed"] = bool(passed)
    od["iterations"] = iterations
    od["pop_per_iter"] = pop_per_iter
    # Reproducibility keys (extension of original contract — keep impl & oracle aligned).
    od["n_invalid"] = n_total - n_valid
    od["random_seed"] = int(args.random_seed)
    od["target_col"] = args.target_col
    od["cohort_parquet"] = str(args.parquet)
    od["gat_checkpoint"] = args.gat_checkpoint or "(not provided)"
    od["checkpoint_variant"] = args.checkpoint_variant
    od["ci_mean_bootstrap"] = (
        float("nan") if math.isnan(point_auc) or args.n_bootstrap <= 0
        else _bootstrap_mean(values, target_arr, args.n_bootstrap, args.random_seed)
    )

    json_path = args.out_dir / "final_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(od, f, indent=2, default=_json_default)

    # Stdout one-liner — PASS / FAIL with cohort, AUC, baseline_best, delta.
    # Operator must reflect the verdict: > on PASS, <= on FAIL.
    auc_disp = f"{point_auc:.4f}" if not math.isnan(point_auc) else "nan"
    bb_disp = f"{baseline_best:.4f}" if not math.isnan(baseline_best) else "nan"
    delta_disp = f"{delta:+.4f}" if not math.isnan(delta) else "nan"
    verdict = "PASS" if passed else "FAIL"
    op = ">" if passed else "<="
    print(
        f"{verdict} top_by_gat {args.cohort.upper()} AUC={auc_disp} "
        f"{op} baseline_best={bb_disp} (delta={delta_disp})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
