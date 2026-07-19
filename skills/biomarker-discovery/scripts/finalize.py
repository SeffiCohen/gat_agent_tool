#!/usr/bin/env python3
"""Phase 3 of the biomarker-discovery skill: finalize a propose-score-refine run.

Reads ``iterative_details.csv`` (already populated by ``silent_real_auc.py``),
picks the top expression by GAT score, computes a bootstrap CI on the cohort,
compares against the published external-method baselines for that ICD, and
writes ``final_report.md`` + ``final_report.json`` to ``--out-dir``.

The single stdout line ("PASS ..." or "FAIL ...") is the acceptance signal the
skill uses to decide success: top-by-GAT real univariate AUC strictly greater
than ``baseline_best`` => PASS.

Run::

    python finalize.py \\
        --csv      Results/714/iterative_llm_gat/medgemma-4b-it/iterative_details.csv \\
        --parquet  Data/MIMIC/preprocessed/714_RheumatoidArthritis_mimic.parquet \\
        --target-col icd_714 \\
        --baseline-dir Results/714/evaluation_results_external_v2/mimic \\
        --out-dir  Results/714/iterative_llm_gat/medgemma-4b-it \\
        --disease-id 714 --disease-name RheumatoidArthritis --cohort mimic

Exit codes:
    0  success
    2  CSV missing
    3  parquet missing
    4  schema problem (missing target col, no lab_* cols, missing CSV cols)
    5  no valid expressions in CSV
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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

_TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "template_final_report.md"

# Canonical CSV columns (per silent_real_auc.py contract).
_REQUIRED_CSV_COLS = (
    "iter", "rank_in_iter", "expression", "gat_score",
    "real_auc", "is_valid", "duplicate_of",
)

# Cross-cohort discovery — kept here (not in references/diseases.md) because the
# skill's auto-discovery has to match the on-disk preprocessor layout (set in
# Code/preprocess_*_external.py). Source-of-truth for the path patterns is
# CLAUDE.md "External validation" section.
_COHORT_DIRS = {
    "mimic": "MIMIC",
    "ehrshot": "EHRShot",
    "nhanes": "NHANES",
}
# Per Code/preprocess_nhanes_external.py:NHANES_CASE_DEFINITIONS — only these 5
# diseases have NHANES labels. ICD 242 and 2452 share the "any-thyroid" label.
_NHANES_SUPPORTED_DISEASES = {"242", "250", "696", "714", "2452"}


def _eprint(*args: Any, **kwargs: Any) -> None:
    """stderr print helper (warnings, progress)."""
    print(*args, file=sys.stderr, **kwargs)


def _coerce_is_valid(series: pd.Series) -> pd.Series:
    """Robust True/False coercion: bool, ``"True"``/``"False"``, ``1``/``0``, NaN->False."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1"))


def _nan_to_none(o: Any) -> Any:
    """Coerce NaN to None for json-serialisable output (json.dumps barfs on NaN)."""
    if isinstance(o, float) and math.isnan(o):
        return None
    if pd.isna(o) if not isinstance(o, (list, dict, tuple)) else False:
        return None
    return o


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", required=True, type=Path,
                   help="path to iterative_details.csv (with real_auc populated)")
    p.add_argument("--parquet", required=True, type=Path,
                   help="cohort parquet with lab_* + binary target column")
    p.add_argument("--target-col", required=True,
                   help="binary outcome column in parquet (e.g., icd_714)")
    p.add_argument("--baseline-dir", required=True, type=Path,
                   help="dir of <Tool>_eval_details.csv baselines for this ICD/cohort")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="output directory for final_report.md + final_report.json")
    p.add_argument("--disease-id", required=True,
                   help="ICD code (e.g., 714)")
    p.add_argument("--disease-name", required=True,
                   help="human disease label (e.g., RheumatoidArthritis)")
    p.add_argument("--cohort", required=True,
                   help="cohort name (e.g., mimic / ehrshot / nhanes)")
    p.add_argument("--gat-checkpoint", default="",
                   help="path to GAT checkpoint used (for reproducibility block)")
    p.add_argument("--checkpoint-variant", default="best_by_loss",
                   help="checkpoint variant string (default: best_by_loss)")
    p.add_argument("--n-bootstrap", type=int, default=500,
                   help="number of bootstrap resamples for CI (default: 500)")
    p.add_argument("--random-seed", type=int, default=42,
                   help="bootstrap seed (default: 42)")
    p.add_argument("--auto-cohorts", action=argparse.BooleanOptionalAction, default=True,
                   help="auto-discover and evaluate the picks on every available "
                        "external cohort (mimic + ehrshot + nhanes-where-supported); "
                        "default: on. Pass --no-auto-cohorts to evaluate only --cohort.")
    return p.parse_args(argv)


def _load_baselines(baseline_dir: Path) -> dict[str, dict[str, float | int]]:
    """Glob ``<Tool>_eval_details.csv`` baseline files; return per-tool max AUC + count.

    Skips files containing ``training_rank`` or starting with ``gat_ranked_`` (those
    are reranks, not the original tool outputs)."""
    baselines: dict[str, dict[str, float | int]] = {}
    for path in sorted(baseline_dir.glob("*_eval_details.csv")):
        name = path.name
        if "training_rank" in name or name.startswith("gat_ranked_"):
            continue
        tool = name[: -len("_eval_details.csv")]
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            _eprint(f"warn: could not read baseline {path}: {exc}")
            continue
        if "univariate_auc" not in df.columns:
            _eprint(f"warn: baseline {path} missing 'univariate_auc' col, skipping")
            continue
        max_auc = df["univariate_auc"].max(skipna=True)
        n_expr = int(df["univariate_auc"].notna().sum())
        baselines[tool] = {"max_auc": float(max_auc), "n_expressions": n_expr}
    return baselines


def _baseline_lookup(baselines: dict[str, dict[str, float | int]],
                     prefixes: tuple[str, ...]) -> tuple[float, int]:
    """Find first baseline whose tool name starts with any of ``prefixes`` (case-insensitive).

    Used to fill the GPT/Gemini/SciSpace rows in the markdown table — file naming
    is inconsistent across disease folders (``GPT.csv`` vs ``GPTDeepResearch.csv``,
    ``SciSpaceAgent.csv`` vs ``SciSpace_BM_Agent.csv``)."""
    lower_prefixes = tuple(p.lower() for p in prefixes)
    for tool, b in baselines.items():
        if tool.lower().startswith(lower_prefixes):
            return float(b["max_auc"]), int(b["n_expressions"])
    return float("nan"), 0


def _fmt_auc(x: float) -> str:
    """4-decimal AUC, or ``N/A`` for NaN."""
    return "N/A" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.4f}"


def _nan_metrics() -> dict[str, float]:
    """All-NaN metrics dict — placeholder for unavailable cohorts / failed evals."""
    return {"point_auc": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "mean_boot": float("nan")}


def _fmt_signed(x: Any) -> str:
    """``+0.0123`` or ``-0.0123`` or ``+N/A``."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "+N/A"
    return f"{x:+.4f}"


def _ci_str(metrics: dict[str, float]) -> str:
    """Return ``[lo, hi]`` formatted, or ``[N/A, N/A]``."""
    return f"[{_fmt_auc(metrics['ci_low'])}, {_fmt_auc(metrics['ci_high'])}]"


def _resolve_extra_cohort_specs(
    disease_id: str, disease_name: str, primary_cohort: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Auto-discover cohort specs other than ``primary_cohort``.

    For each of the 3 canonical external cohorts (mimic, ehrshot, nhanes), build
    the expected parquet path and probe it on disk. Returns one spec dict per
    cohort found, with fields: ``cohort``, ``parquet`` (Path), ``target_col``
    (always ``icd_<disease_id>``), ``baseline_dir`` (Path; first existing
    candidate, else first candidate).

    Skips:
    - the primary cohort (already handled by --cohort/--parquet flags)
    - NHANES for diseases not in :data:`_NHANES_SUPPORTED_DISEASES`
    - cohorts whose parquet doesn't exist on disk
    """
    extras: list[dict[str, Any]] = []
    for cohort, dir_name in _COHORT_DIRS.items():
        if cohort == primary_cohort:
            continue
        if cohort == "nhanes" and disease_id not in _NHANES_SUPPORTED_DISEASES:
            _eprint(f"info: skipping nhanes (disease {disease_id} not in supported set)")
            continue
        parquet = (repo_root / "Data" / dir_name / "preprocessed"
                   / f"{disease_id}_{disease_name}_{cohort}.parquet")
        if not parquet.exists():
            _eprint(f"info: skipping {cohort} (parquet not found: {parquet})")
            continue
        # Try the three known baseline dir locations in order. The 2nd is the
        # canonical one per SKILL.md; the 1st is where 555/* etc. live in this
        # repo (full eval); the 3rd is the legacy inline-eval folder.
        candidates = [
            repo_root / "Results" / disease_id / "evaluation_results_external_v2_full" / cohort,
            repo_root / "Results" / disease_id / "evaluation_results_external_v2" / cohort,
            repo_root / "Results" / disease_id / "evaluation_results_unified" / f"external_validation_{cohort}",
        ]
        baseline_dir = next(
            (c for c in candidates if c.exists() and any(c.glob("*_eval_details.csv"))),
            candidates[0],  # falls back to v2_full path even if missing
        )
        extras.append({
            "cohort": cohort,
            "parquet": parquet,
            "target_col": f"icd_{disease_id}",
            "baseline_dir": baseline_dir,
        })
    return extras


def _evaluate_cohort(
    spec: dict[str, Any],
    top_expr: str, first_seed_expr: str, iter_winner_expr: str,
    n_boot: int, seed: int,
) -> dict[str, Any]:
    """Run the three picks (top-by-GAT, first-seed, iter-winner) on one cohort.

    Returns a dict with: cohort name, availability + cohort sizes, the three
    pick metrics (each a NaN-safe dict), baselines + baseline_best, and the two
    deltas (top vs baseline_best, iter_winner vs first_seed)."""
    out: dict[str, Any] = {
        "cohort": spec["cohort"],
        "parquet": str(spec["parquet"]),
        "baseline_dir": str(spec["baseline_dir"]),
        "target_col": spec["target_col"],
        "is_primary": spec.get("is_primary", False),
        "available": False,
        "n_total": 0, "n_pos": 0, "n_neg": 0,
        "top": _nan_metrics(),
        "first_seed": _nan_metrics(),
        "iter_winner": _nan_metrics(),
        "baselines": {},
        "baseline_best": float("nan"),
        "delta_top_vs_baseline": float("nan"),
        "delta_iter_vs_seed": float("nan"),
        "warnings": [],
    }
    parquet_path = Path(spec["parquet"])
    if not parquet_path.exists():
        out["warnings"].append(f"parquet not found: {parquet_path}")
        return out
    try:
        df_cohort = pd.read_parquet(parquet_path)
    except Exception as exc:
        out["warnings"].append(f"failed to read parquet: {exc}")
        return out
    if spec["target_col"] not in df_cohort.columns:
        out["warnings"].append(f"target column missing: {spec['target_col']}")
        return out
    target_arr = df_cohort[spec["target_col"]].to_numpy(dtype=np.float64)
    out["available"] = True
    out["n_total"] = len(df_cohort)
    out["n_pos"] = int(target_arr.sum())
    out["n_neg"] = int((target_arr == 0).sum())

    if top_expr:
        out["top"] = _eval_with_ci(top_expr, df_cohort, target_arr,
                                   n_boot=n_boot, seed=seed)
    if first_seed_expr:
        out["first_seed"] = _eval_with_ci(first_seed_expr, df_cohort, target_arr,
                                          n_boot=n_boot, seed=seed)
    if iter_winner_expr:
        # Reuse top metrics if same expression (skip duplicate bootstrap).
        if iter_winner_expr == top_expr:
            out["iter_winner"] = dict(out["top"])
        else:
            out["iter_winner"] = _eval_with_ci(iter_winner_expr, df_cohort,
                                               target_arr, n_boot=n_boot, seed=seed)

    baseline_dir = Path(spec["baseline_dir"])
    if baseline_dir.exists():
        out["baselines"] = _load_baselines(baseline_dir)
        if out["baselines"]:
            aucs = [b["max_auc"] for b in out["baselines"].values()
                    if not (b["max_auc"] is None or
                            (isinstance(b["max_auc"], float) and math.isnan(b["max_auc"])))]
            if aucs:
                out["baseline_best"] = float(max(aucs))
                if not math.isnan(out["top"]["point_auc"]):
                    out["delta_top_vs_baseline"] = (
                        out["top"]["point_auc"] - out["baseline_best"]
                    )

    if (not math.isnan(out["first_seed"]["point_auc"])
            and not math.isnan(out["iter_winner"]["point_auc"])):
        out["delta_iter_vs_seed"] = (
            out["iter_winner"]["point_auc"] - out["first_seed"]["point_auc"]
        )
    return out


def _build_cross_cohort_top_table(cohort_results: list[dict[str, Any]]) -> str:
    """Markdown rows for the per-cohort top-by-GAT table."""
    if not cohort_results:
        return "| _no cohorts evaluated_ | — | — | — | — |"
    rows = []
    for r in cohort_results:
        marker = " (primary)" if r["is_primary"] else ""
        if not r["available"]:
            note = "; ".join(r["warnings"]) or "not available"
            rows.append(f"| {r['cohort']}{marker} | _{note}_ | — | — | — |")
            continue
        n_str = f"N={r['n_total']:,} (pos={r['n_pos']})"
        rows.append(
            f"| {r['cohort']}{marker} ({n_str}) | {_fmt_auc(r['top']['point_auc'])} "
            f"| {_ci_str(r['top'])} | {_fmt_auc(r['baseline_best'])} "
            f"| {_fmt_signed(r['delta_top_vs_baseline'])} |"
        )
    return "\n".join(rows)


def _build_before_after_per_cohort_table(cohort_results: list[dict[str, Any]]) -> str:
    """Markdown rows for the per-cohort before-vs-after-GAT table."""
    if not cohort_results:
        return "| _no cohorts evaluated_ | — | — | — |"
    rows = []
    for r in cohort_results:
        marker = " (primary)" if r["is_primary"] else ""
        if not r["available"]:
            note = "; ".join(r["warnings"]) or "not available"
            rows.append(f"| {r['cohort']}{marker} | _{note}_ | — | — |")
            continue
        seed_str = f"{_fmt_auc(r['first_seed']['point_auc'])} {_ci_str(r['first_seed'])}"
        iter_str = f"{_fmt_auc(r['iter_winner']['point_auc'])} {_ci_str(r['iter_winner'])}"
        rows.append(
            f"| {r['cohort']}{marker} | {seed_str} | {iter_str} "
            f"| {_fmt_signed(r['delta_iter_vs_seed'])} |"
        )
    return "\n".join(rows)


def _eval_with_ci(
    expr: str, df_data: pd.DataFrame, target: np.ndarray,
    n_boot: int, seed: int,
) -> dict[str, float]:
    """Evaluate ``expr`` on ``df_data``, return point AUC + bootstrap CI.

    All four numeric outputs (point_auc, ci_low, ci_high, mean_boot) are NaN
    if the expression fails to evaluate — caller can detect via isnan."""
    out = {"point_auc": float("nan"), "ci_low": float("nan"),
           "ci_high": float("nan"), "mean_boot": float("nan")}
    try:
        values = eval_expression(expr, df_data)
    except Exception as exc:
        _eprint(f"warn: eval failed for {expr!r}: {exc}")
        return out
    if values is None:
        return out
    try:
        out["point_auc"] = float(univariate_auc(values, target))
        mean_boot, lo, hi = bootstrap_auc_ci(values, target, n_boot=n_boot, seed=seed)
        out["mean_boot"] = float(mean_boot)
        out["ci_low"] = float(lo)
        out["ci_high"] = float(hi)
    except Exception as exc:
        _eprint(f"warn: bootstrap failed for {expr!r}: {exc}")
    return out


def _pick_first_expression_and_iter_winner(df_valid: pd.DataFrame) -> tuple[Any, Any]:
    """Return (first_expression_row, iter_winner_row) used by the before-vs-after-GAT block.

    - first_expression = the row with ``rank_in_iter == 0`` in the **lowest**
      iter that has any such row (typically ``iter == 1`` under the new flow:
      the LLM's #1 self-ranked pick from the initial 200-batch). This is the
      "first expression made" — the LLM's first guess BEFORE any GAT-driven
      refinement. For back-compat with older runs that scored literature seeds
      at ``iter == 0``, ``iter == 0``'s rank-0 row wins if present.
    - iter_winner = row with max ``gat_score`` among iters STRICTLY AFTER the
      first batch (``iter > first_iter``). This isolates the value the
      GAT-feedback loop adds beyond the LLM's first guess. Falls back to
      ``iter >= 1`` if first_iter is 0 (legacy seed-scoring layout). None if
      there are no rows past the first batch (single-batch runs).
    """
    rank0 = df_valid[df_valid["rank_in_iter"] == 0]
    if len(rank0) == 0:
        return None, None
    first_iter = int(rank0["iter"].min())
    first_iter_zero = rank0[rank0["iter"] == first_iter]
    first_expression = first_iter_zero.iloc[0]

    iter_after_first = df_valid[df_valid["iter"] > first_iter]
    iter_winner = (
        iter_after_first.sort_values("gat_score", ascending=False).iloc[0]
        if len(iter_after_first) else None
    )
    return first_expression, iter_winner


def _build_top10_table(df_valid: pd.DataFrame, cohort: str) -> str:
    """Markdown rows (no header — header is in the template) for top-10 by GAT score."""
    top10 = df_valid.sort_values("gat_score", ascending=False).head(10)
    rows = []
    for rank, (_, r) in enumerate(top10.iterrows(), start=1):
        expr = str(r["expression"]).replace("|", "\\|")
        gat = r["gat_score"]
        ra = r["real_auc"]
        it = r["iter"]
        rows.append(
            f"| {rank} | `{expr}` | {_fmt_auc(float(gat)) if pd.notna(gat) else 'N/A'} "
            f"| {_fmt_auc(float(ra)) if pd.notna(ra) else 'N/A'} | {it} |"
        )
    return "\n".join(rows) if rows else "| — | _no valid expressions_ | — | — | — |"


def _build_trajectory_table(df_valid: pd.DataFrame) -> tuple[str, int]:
    """Markdown rows for the per-iter trajectory; also return the iter count.

    For each iter: n_new (count of valid expressions proposed),
    best_gat_this_iter (max gat_score among that iter's rows),
    best_gat_cumulative (running max across iters)."""
    if df_valid.empty:
        return "| — | 0 | — | — |", 0
    grouped = df_valid.groupby("iter", sort=True)
    rows = []
    cum_best = float("-inf")
    for it, sub in grouped:
        n_new = len(sub)
        best_this = float(sub["gat_score"].max(skipna=True))
        if pd.notna(best_this) and best_this > cum_best:
            cum_best = best_this
        cum_str = _fmt_auc(cum_best) if cum_best != float("-inf") else "N/A"
        rows.append(f"| {it} | {n_new} | {_fmt_auc(best_this)} | {cum_str} |")
    return "\n".join(rows), len(grouped)


def main() -> int:
    args = _parse_args()

    # ----- Step 1: Load + validate inputs ----------------------------------
    if not args.csv.exists():
        _eprint(f"error: CSV not found: {args.csv}")
        return 2
    if not args.parquet.exists():
        _eprint(f"error: parquet not found: {args.parquet}")
        return 3
    if not args.baseline_dir.exists():
        _eprint(f"warn: baseline-dir does not exist: {args.baseline_dir}")
        # Don't bail — we'll just have an empty baseline dict and auto-PASS.

    df = pd.read_csv(args.csv)
    missing = [c for c in _REQUIRED_CSV_COLS if c not in df.columns]
    if missing:
        _eprint(f"error: CSV missing required columns: {missing}; have {list(df.columns)}")
        return 4

    df_data = pd.read_parquet(args.parquet)
    if args.target_col not in df_data.columns:
        _eprint(f"error: target column {args.target_col!r} not in parquet")
        return 4
    if not any(c.startswith("lab_") for c in df_data.columns):
        _eprint(f"error: parquet has no lab_* columns")
        return 4

    n_total = len(df)
    df["_is_valid_bool"] = _coerce_is_valid(df["is_valid"])
    df_valid = df[df["_is_valid_bool"]].copy()
    n_valid = len(df_valid)
    n_invalid = n_total - n_valid

    if df_valid.empty:
        _eprint("error: no valid expressions in CSV")
        return 5

    # ----- Step 2: Identify top picks --------------------------------------
    top_by_gat = df_valid.sort_values("gat_score", ascending=False).iloc[0]
    # NaN-safe argmax over real_auc (idxmax ignores NaN by default).
    real_auc_series = df_valid["real_auc"]
    if real_auc_series.notna().any():
        top_by_real_auc = df_valid.loc[real_auc_series.idxmax()]
    else:
        top_by_real_auc = top_by_gat
        _eprint("warn: no non-NaN real_auc values in CSV (silent_real_auc not run?)")

    # ----- Step 3: Identify expressions, prepare cohort spec list -----------
    top_expr = str(top_by_gat["expression"])
    top_gat_score = float(top_by_gat["gat_score"])

    # Identify the first expression (rank_in_iter==0 in the lowest iter) and
    # iter winner (best-by-GAT among rows AFTER that first iter). These are
    # CSV-level identifiers — same for every cohort.
    first_seed_row, iter_winner_row = _pick_first_expression_and_iter_winner(df_valid)
    if first_seed_row is not None:
        first_seed_expr = str(first_seed_row["expression"])
        first_seed_gat = float(first_seed_row["gat_score"])
    else:
        first_seed_expr, first_seed_gat = "", float("nan")
        _eprint("warn: no rank_in_iter==0 rows in CSV; "
                "before-vs-after-GAT block will be N/A")
    if iter_winner_row is not None:
        iter_winner_expr = str(iter_winner_row["expression"])
        iter_winner_gat = float(iter_winner_row["gat_score"])
    else:
        iter_winner_expr, iter_winner_gat = "", float("nan")
        _eprint("warn: no rows past the first iter in CSV; "
                "iteration winner block will be N/A")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Primary-cohort spec (always evaluated).
    primary_spec = {
        "cohort": args.cohort,
        "parquet": args.parquet,
        "target_col": args.target_col,
        "baseline_dir": args.baseline_dir,
        "is_primary": True,
    }
    cohort_specs = [primary_spec]
    if args.auto_cohorts:
        cohort_specs.extend(_resolve_extra_cohort_specs(
            args.disease_id, args.disease_name, args.cohort, _REPO_ROOT))
    _eprint(f"info: evaluating {len(cohort_specs)} cohort(s): "
            f"{', '.join(s['cohort'] for s in cohort_specs)}")

    # ----- Step 4: Evaluate primary cohort first (drives degraded path) -----
    primary_result = _evaluate_cohort(
        primary_spec, top_expr, first_seed_expr, iter_winner_expr,
        n_boot=args.n_bootstrap, seed=args.random_seed,
    )
    if not primary_result["available"] or math.isnan(primary_result["top"]["point_auc"]):
        # Degraded path — top expression failed to evaluate on the primary
        # cohort. Render minimal report and exit 0 so outer driver completes.
        _eprint(f"warn: top-by-GAT expression failed to evaluate on primary "
                f"cohort ({args.cohort}): {top_expr!r}")
        degraded_md = (
            f"# Biomarker discovery — {args.disease_name} (ICD {args.disease_id})\n\n"
            f"## DEGRADED REPORT — top expression failed evaluation on primary cohort\n\n"
            f"```\n{top_expr}\n```\n\n"
            f"GAT score: {top_gat_score:.4f}\n\n"
            f"Could not score against {args.cohort} cohort "
            f"({args.parquet.name}). The expression may reference unknown "
            f"features or have a parse error. Inspect `iterative_details.csv` "
            f"for alternatives.\n"
        )
        (args.out_dir / "final_report.md").write_text(degraded_md)
        (args.out_dir / "final_report.json").write_text(json.dumps({
            "disease_id": args.disease_id, "disease_name": args.disease_name,
            "cohort": args.cohort, "top_expression": top_expr,
            "top_gat_score": top_gat_score, "degraded": True,
            "reason": "top expression failed evaluation on primary cohort",
        }, indent=2))
        print(f"FAIL top_by_gat {args.cohort.upper()} AUC=N/A "
              f"<= baseline_best=N/A (degraded: top expression failed eval)")
        return 0

    # Sanity-check vs the silent-real-auc value (FP-precision agreement).
    csv_real_auc = top_by_gat["real_auc"]
    if pd.notna(csv_real_auc) and pd.notna(primary_result["top"]["point_auc"]):
        diff = abs(float(csv_real_auc) - primary_result["top"]["point_auc"])
        if diff > 1e-9:
            _eprint(f"warn: top-by-GAT real_auc in CSV ({csv_real_auc}) differs from "
                    f"recomputed point_auc ({primary_result['top']['point_auc']}) "
                    f"by {diff:.3e}")

    # ----- Step 5: Evaluate extra cohorts (if --auto-cohorts) ---------------
    cohort_results = [primary_result]
    for spec in cohort_specs[1:]:
        res = _evaluate_cohort(
            spec, top_expr, first_seed_expr, iter_winner_expr,
            n_boot=args.n_bootstrap, seed=args.random_seed,
        )
        if res["available"]:
            _eprint(f"info: {res['cohort']}: top AUC={_fmt_auc(res['top']['point_auc'])} "
                    f"(N={res['n_total']:,}, pos={res['n_pos']})")
        else:
            _eprint(f"info: {res['cohort']}: skipped — "
                    f"{'; '.join(res['warnings']) or 'unavailable'}")
        cohort_results.append(res)

    # Pull primary metrics for the existing single-cohort templating contract.
    point_auc = primary_result["top"]["point_auc"]
    ci_low = primary_result["top"]["ci_low"]
    ci_high = primary_result["top"]["ci_high"]
    mean_boot = primary_result["top"]["mean_boot"]
    baselines = primary_result["baselines"]
    baseline_best = primary_result["baseline_best"]
    delta = primary_result["delta_top_vs_baseline"]
    first_seed_metrics = primary_result["first_seed"]
    iter_winner_metrics = primary_result["iter_winner"]
    iter_vs_seed_delta = primary_result["delta_iter_vs_seed"]

    # ----- Step 6: PASS/FAIL gate (primary-cohort only) ---------------------
    if math.isnan(baseline_best):
        passed = True  # auto-PASS if no baselines available
        baseline_note = " (no baselines found)"
    else:
        passed = bool(point_auc > baseline_best)
        baseline_note = ""

    # ----- Step 7: Trajectory ------------------------------------------------
    trajectory_table, iterations = _build_trajectory_table(df_valid)
    pop_per_iter = (
        int(round(n_valid / iterations)) if iterations > 0 else 0
    )

    # ----- Step 8: Render markdown -----------------------------------------
    auc_gpt, n_gpt = _baseline_lookup(baselines, ("GPT", "GPTDeepResearch"))
    auc_gem, n_gem = _baseline_lookup(baselines, ("Gemini", "Gemini3Pro"))
    auc_sci, n_sci = _baseline_lookup(baselines, ("SciSpace",))

    # Number of distinct lab_* features in the parquet (for reproducibility block).
    n_features_used = sum(1 for c in df_data.columns if c.startswith("lab_"))

    # The template uses :.4f formatting on AUC fields — must be float, not NaN.
    # For the table cells we substitute "N/A" strings if missing by patching
    # the template formatter call below. Easiest: use safe placeholders.
    def _safe_4f(x: float) -> str:
        return _fmt_auc(x)

    pass_fail_marker = "PASS" if passed else "FAIL"

    template = _TEMPLATE.read_text()

    # Build placeholder dict. Template fields with `:.4f` need real floats; for
    # potentially-NaN values we pre-format and inject as strings via {name}
    # by replacing the format-spec'd placeholders before .format().
    rendered = template
    # Pre-substitute the AUC-with-format-spec placeholders that may be NaN.
    for key, value in {
        "top_gat_score": top_gat_score,
        "top_real_auc": point_auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "auc_gpt": auc_gpt,
        "auc_gem": auc_gem,
        "auc_sci": auc_sci,
        "baseline_best": baseline_best,
        "first_seed_gat": first_seed_gat,
        "first_seed_real_auc": first_seed_metrics["point_auc"],
        "first_seed_ci_low": first_seed_metrics["ci_low"],
        "first_seed_ci_high": first_seed_metrics["ci_high"],
        "iter_winner_gat": iter_winner_gat,
        "iter_winner_real_auc": iter_winner_metrics["point_auc"],
        "iter_winner_ci_low": iter_winner_metrics["ci_low"],
        "iter_winner_ci_high": iter_winner_metrics["ci_high"],
    }.items():
        rendered = rendered.replace("{" + key + ":.4f}", _safe_4f(value))
    # Delta uses sign-prefixed format `:+.4f`.
    delta_str = "+N/A" if math.isnan(delta) else f"{delta:+.4f}"
    rendered = rendered.replace("{delta:+.4f}", delta_str)
    iter_vs_seed_delta_str = (
        "+N/A" if math.isnan(iter_vs_seed_delta) else f"{iter_vs_seed_delta:+.4f}"
    )
    rendered = rendered.replace("{iter_vs_seed_delta:+.4f}", iter_vs_seed_delta_str)

    # Build cross-cohort tables (markdown rows for the unified report block).
    cross_cohort_top_table = _build_cross_cohort_top_table(cohort_results)
    cross_cohort_before_after_table = _build_before_after_per_cohort_table(cohort_results)
    n_cohorts_evaluated = sum(1 for r in cohort_results if r["available"])
    cohorts_evaluated_csv = ", ".join(
        r["cohort"] + (" (primary)" if r["is_primary"] else "")
        for r in cohort_results if r["available"]
    )

    # Now plain str.format for all remaining {name} placeholders.
    timestamp = datetime.now().isoformat(timespec="seconds")
    placeholders = {
        "disease_id": args.disease_id,
        "disease_name": args.disease_name,
        "cohort": args.cohort,
        "timestamp": timestamp,
        "iterations": iterations,
        "pop_per_iter": pop_per_iter,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "pass_fail_marker": f"**{pass_fail_marker}**",
        "top_expression": top_expr,
        "n_boot": args.n_bootstrap,
        "n_gpt": n_gpt,
        "n_gem": n_gem,
        "n_sci": n_sci,
        "top10_table": _build_top10_table(df_valid, args.cohort),
        "trajectory_table": trajectory_table,
        "gat_checkpoint": args.gat_checkpoint or "(not provided)",
        "cohort_parquet": str(args.parquet),
        "target_col": args.target_col,
        "checkpoint_variant": args.checkpoint_variant,
        "n_features_used": n_features_used,
        "random_seed": args.random_seed,
        "first_seed_expression": first_seed_expr or "(no iter==0 rows)",
        "iter_winner_expression": iter_winner_expr or "(no iter>=1 rows — iteration loop did not produce a winner)",
        "cross_cohort_top_table": cross_cohort_top_table,
        "cross_cohort_before_after_table": cross_cohort_before_after_table,
        "n_cohorts_evaluated": n_cohorts_evaluated,
        "cohorts_evaluated_csv": cohorts_evaluated_csv or "(none)",
    }
    try:
        report_md = rendered.format(**placeholders)
    except KeyError as exc:
        _eprint(f"warn: template missing key {exc}; rendering with safe-substitute")
        # Fall back to repeated .replace() so we don't lose the report on a
        # template/placeholder mismatch.
        report_md = rendered
        for k, v in placeholders.items():
            report_md = report_md.replace("{" + k + "}", str(v))

    (args.out_dir / "final_report.md").write_text(report_md)

    # ----- Step 9: JSON sidecar --------------------------------------------
    json_payload = {
        "disease_id": args.disease_id,
        "disease_name": args.disease_name,
        "cohort": args.cohort,
        "timestamp": timestamp,
        "top_expression": top_expr,
        "top_gat_score": _nan_to_none(top_gat_score),
        "top_real_auc": _nan_to_none(point_auc),
        "ci_low": _nan_to_none(ci_low),
        "ci_high": _nan_to_none(ci_high),
        "ci_mean_bootstrap": _nan_to_none(mean_boot),
        "n_bootstrap": args.n_bootstrap,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "baselines": {
            tool: {
                "max_auc": _nan_to_none(b["max_auc"]),
                "n_expressions": int(b["n_expressions"]),
            } for tool, b in baselines.items()
        },
        "baseline_best": _nan_to_none(baseline_best),
        "delta": _nan_to_none(delta),
        "passed": passed,
        "cohorts": [
            {
                "cohort": r["cohort"],
                "is_primary": r["is_primary"],
                "available": r["available"],
                "parquet": r["parquet"],
                "baseline_dir": r["baseline_dir"],
                "target_col": r["target_col"],
                "n_total": r["n_total"],
                "n_pos": r["n_pos"],
                "n_neg": r["n_neg"],
                "top": {k: _nan_to_none(v) for k, v in r["top"].items()},
                "first_seed": {k: _nan_to_none(v) for k, v in r["first_seed"].items()},
                "iter_winner": {k: _nan_to_none(v) for k, v in r["iter_winner"].items()},
                "baselines": {
                    tool: {
                        "max_auc": _nan_to_none(b["max_auc"]),
                        "n_expressions": int(b["n_expressions"]),
                    } for tool, b in r["baselines"].items()
                },
                "baseline_best": _nan_to_none(r["baseline_best"]),
                "delta_top_vs_baseline": _nan_to_none(r["delta_top_vs_baseline"]),
                "delta_iter_vs_seed": _nan_to_none(r["delta_iter_vs_seed"]),
                "warnings": r["warnings"],
            } for r in cohort_results
        ],
        "before_vs_after_gat": {
            "first_biomarker": {
                "expression": first_seed_expr or None,
                "gat_score": _nan_to_none(first_seed_gat),
                "real_auc": _nan_to_none(first_seed_metrics["point_auc"]),
                "ci_low": _nan_to_none(first_seed_metrics["ci_low"]),
                "ci_high": _nan_to_none(first_seed_metrics["ci_high"]),
                "ci_mean_bootstrap": _nan_to_none(first_seed_metrics["mean_boot"]),
                "selection_rule": (
                    "rank_in_iter==0 in the lowest iter present "
                    "(LLM's first self-ranked pick from the initial 200-batch)"
                ),
            },
            "iteration_winner": {
                "expression": iter_winner_expr or None,
                "gat_score": _nan_to_none(iter_winner_gat),
                "real_auc": _nan_to_none(iter_winner_metrics["point_auc"]),
                "ci_low": _nan_to_none(iter_winner_metrics["ci_low"]),
                "ci_high": _nan_to_none(iter_winner_metrics["ci_high"]),
                "ci_mean_bootstrap": _nan_to_none(iter_winner_metrics["mean_boot"]),
                "selection_rule": (
                    "max gat_score among rows with iter > first_iter "
                    "(best from GAT-driven feedback iterations)"
                ),
            },
            "delta_real_auc": _nan_to_none(iter_vs_seed_delta),
        },
        "iterations": iterations,
        "pop_per_iter": pop_per_iter,
        "gat_checkpoint": args.gat_checkpoint,
        "checkpoint_variant": args.checkpoint_variant,
        "target_col": args.target_col,
        "cohort_parquet": str(args.parquet),
        "random_seed": args.random_seed,
    }
    (args.out_dir / "final_report.json").write_text(
        json.dumps(json_payload, indent=2, default=str)
    )

    # ----- Step 10: Acceptance signals on stdout ---------------------------
    # Two lines:
    #   1. Existing PASS/FAIL gate vs published external-method baselines
    #      (kept for back-compat with any tooling that grep's the line).
    #   2. NEW: BASELINE_DELTA — iter_winner's real AUC vs the initial-expression
    #      baseline (best iter==0 row by gat_score). This is the primary signal
    #      of the iteration loop's value: did K rounds of GAT-driven proposal-
    #      score-refine actually improve real AUC over the literature seed?
    cohort_upper = args.cohort.upper()
    auc_str = _fmt_auc(point_auc)
    base_str = _fmt_auc(baseline_best)
    op = ">" if passed else "<="
    print(
        f"{pass_fail_marker} top_by_gat {cohort_upper} AUC={auc_str} "
        f"{op} baseline_best={base_str} (delta={delta_str}){baseline_note}"
    )
    # First-expression (LLM's #1 self-ranked pick from initial 200) signal.
    first_expr_auc_str = _fmt_auc(first_seed_metrics["point_auc"])
    iter_auc_str = _fmt_auc(iter_winner_metrics["point_auc"])
    iter_vs_seed_signed = (
        "+N/A" if math.isnan(iter_vs_seed_delta) else f"{iter_vs_seed_delta:+.4f}"
    )
    if math.isnan(iter_vs_seed_delta):
        baseline_verdict = "N/A"
    elif iter_vs_seed_delta > 0:
        baseline_verdict = "IMPROVED"
    else:
        baseline_verdict = "REGRESSED"
    print(
        f"BASELINE_DELTA {cohort_upper} iter_winner={iter_auc_str} "
        f"vs first_expression={first_expr_auc_str} ({iter_vs_seed_signed}) → {baseline_verdict}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
