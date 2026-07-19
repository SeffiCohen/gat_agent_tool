#!/usr/bin/env python3
"""verify_first_expression_baseline.py — exercise finalize.py's picker.

Synthesizes a 3-iteration ``iterative_details.csv`` and a stub parquet with
the right schema, runs ``scripts/finalize.py``, asserts the resulting
``final_report.json`` selects:
  - ``first_biomarker.expression`` == the row at iter==1, rank_in_iter==0
    (NOT the highest-gat_score row of iter==1 — that's the regression we're
    guarding against; the new picker uses rank_in_iter, not gat_score).
  - ``iteration_winner.expression`` == the highest gat_score row among iter>1.
  - The BASELINE_DELTA stdout line uses ``first_expression=...`` not
    ``baseline=...`` (label rename verifier).

Synthesized CSV layout:
  - iter=1: 200 rows. rank_in_iter 0..199. gat_scores ascending from 0.55..0.65,
    so rank_in_iter==0 has gat_score=0.55 — DELIBERATELY not the GAT-best.
  - iter=2: 200 rows. gat_scores in [0.60, 0.70]. Best ~ 0.70.
  - iter=3: 200 rows. gat_scores in [0.62, 0.72]. Best ~ 0.72 (the iter winner).

Exit 0 on all-pass, 1 on first failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_SKILL_DIR = Path(__file__).resolve().parent.parent
_FINALIZE = _SKILL_DIR / "scripts" / "finalize.py"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _build_csv(out_path: Path) -> None:
    """Write a 600-row iterative_details.csv to ``out_path``.

    Crucially, iter==1's rank_in_iter==0 row has gat_score=0.55 — the LOWEST
    score in iter==1 — so any picker that selects 'best by GAT' will pick
    a different row, exposing the bug.
    """
    rows = []
    rng = np.random.default_rng(42)

    # iter == 1: 200 rows with ascending gat_scores (rank 0 = score 0.55,
    # rank 199 = score 0.6485).
    for k in range(200):
        gat = 0.55 + k * 0.0005  # 0.55 .. 0.6495
        rows.append({
            "iter": 1,
            "rank_in_iter": k,
            "expression": f"lab_PLT_last + lab_HB_last  # iter1_rank{k}",
            "gat_score": gat,
            "real_auc": gat - 0.02,  # silent_real_auc would have set this
            "is_valid": True,
            "duplicate_of": "",
        })

    # iter == 2: 200 rows in [0.60, 0.70].
    for k in range(200):
        gat = 0.60 + rng.uniform(0, 0.10)
        rows.append({
            "iter": 2,
            "rank_in_iter": k,
            "expression": f"lab_NEUTpct_last / lab_LYMpct_last  # iter2_rank{k}",
            "gat_score": gat,
            "real_auc": gat - 0.015,
            "is_valid": True,
            "duplicate_of": "",
        })

    # iter == 3: 200 rows in [0.62, 0.72]. The highest score in iter 3 is
    # the global max — this becomes the iter_winner.
    for k in range(200):
        if k == 0:
            gat = 0.72  # deterministic peak — the iter winner
        else:
            gat = 0.62 + rng.uniform(0, 0.09)
        rows.append({
            "iter": 3,
            "rank_in_iter": k,
            "expression": f"lab_RDW_last * lab_PLT_last  # iter3_rank{k}",
            "gat_score": gat,
            "real_auc": gat - 0.01,
            "is_valid": True,
            "duplicate_of": "",
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)


def _build_parquet(out_path: Path, target_col: str) -> None:
    """Write a minimal parquet with all 14 CBC features + target col.

    finalize.py requires:
      - target_col present
      - at least one column starting with `lab_`
      - parquet readable
    The actual feature values matter for `eval_expression` — but our test
    expressions all reference the same feature set, so as long as values are
    finite the bootstrap won't blow up.
    """
    n = 100
    rng = np.random.default_rng(0)
    data = {target_col: rng.integers(0, 2, size=n).astype(np.int64)}
    feats = ["BASO_pct", "EOS_pct", "HB", "HCT", "LYMpct", "MCH", "MCHC",
             "MCV", "MONOpct", "NEUTpct", "PLT", "RBC", "RDW", "WBC"]
    for f in feats:
        data[f"lab_{f}_last"] = rng.uniform(1.0, 100.0, size=n)
    pd.DataFrame(data).to_parquet(out_path)


def main() -> int:
    if not _FINALIZE.exists():
        _fail(f"finalize.py missing at {_FINALIZE}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "iterative_details.csv"
        parquet_path = tmp / "stub.parquet"
        out_dir = tmp / "out"
        out_dir.mkdir()

        target_col = "icd_555"
        _build_csv(csv_path)
        _build_parquet(parquet_path, target_col)
        baseline_dir = tmp / "baselines_empty"  # empty -> auto-PASS

        # Sanity: the iter=1 rank=0 row should have gat_score=0.55 (lowest in iter=1).
        df = pd.read_csv(csv_path)
        i1r0 = df[(df["iter"] == 1) & (df["rank_in_iter"] == 0)].iloc[0]
        if abs(i1r0["gat_score"] - 0.55) > 1e-6:
            _fail(f"synthesized fixture broken: iter1_rank0 gat={i1r0['gat_score']}")
        i1_max = df[df["iter"] == 1]["gat_score"].max()
        if abs(i1_max - 0.55) < 0.05:
            _fail(f"fixture broken: iter1 max gat ({i1_max}) is too close to "
                  f"rank0 gat (0.55) — picker bug won't be detectable")

        # Run finalize.py.
        proc = subprocess.run(
            [
                sys.executable, str(_FINALIZE),
                "--csv", str(csv_path),
                "--parquet", str(parquet_path),
                "--target-col", target_col,
                "--baseline-dir", str(baseline_dir),
                "--out-dir", str(out_dir),
                "--disease-id", "555",
                "--disease-name", "CrohnDisease",
                "--cohort", "mimic",
                "--n-bootstrap", "20",  # fast
                "--no-auto-cohorts",
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            _fail(f"finalize.py exited {proc.returncode}\n"
                  f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

        json_path = out_dir / "final_report.json"
        if not json_path.exists():
            _fail(f"final_report.json missing under {out_dir}")
        payload = json.loads(json_path.read_text())

        # Assertion 1: first_expression has rank_in_iter==0 in iter==1.
        first_expr = payload["before_vs_after_gat"]["first_biomarker"]["expression"]
        if "iter1_rank0" not in (first_expr or ""):
            _fail(
                f"first_biomarker should be the iter==1, rank_in_iter==0 row "
                f"(synthesized as 'iter1_rank0'). "
                f"Got: {first_expr!r}. "
                f"This indicates the picker is using gat_score instead of "
                f"rank_in_iter — REGRESSION."
            )
        # Sanity: gat_score should be 0.55 (NOT the iter==1 max).
        first_gat = payload["before_vs_after_gat"]["first_biomarker"]["gat_score"]
        if abs(first_gat - 0.55) > 1e-3:
            _fail(f"first_biomarker.gat_score should be 0.55 (synthesized "
                  f"rank-0 score), got {first_gat}")

        # Assertion 2: iter_winner is the highest-gat row in iter>1.
        iter_winner_expr = payload["before_vs_after_gat"]["iteration_winner"]["expression"]
        if "iter3_rank0" not in (iter_winner_expr or ""):
            _fail(f"iteration_winner should be iter3_rank0 (peak gat=0.72). "
                  f"Got: {iter_winner_expr!r}")
        iter_winner_gat = payload["before_vs_after_gat"]["iteration_winner"]["gat_score"]
        if abs(iter_winner_gat - 0.72) > 1e-3:
            _fail(f"iteration_winner.gat_score should be 0.72, got {iter_winner_gat}")

        # Assertion 3: stdout BASELINE_DELTA line uses 'first_expression=' label.
        if "BASELINE_DELTA" not in proc.stdout:
            _fail(f"missing BASELINE_DELTA stdout line.\nstdout:\n{proc.stdout}")
        baseline_line = next(
            (line for line in proc.stdout.splitlines()
             if line.startswith("BASELINE_DELTA")),
            None,
        )
        if baseline_line is None:
            _fail("BASELINE_DELTA line not found in stdout")
        if "vs first_expression=" not in baseline_line:
            _fail(
                f"BASELINE_DELTA line missing 'vs first_expression=' label "
                f"(label rename regression). Got: {baseline_line!r}"
            )
        if "vs baseline=" in baseline_line:
            _fail(f"BASELINE_DELTA line still uses old 'vs baseline=' label "
                  f"— rename regression. Got: {baseline_line!r}")

        # Assertion 4: selection_rule reflects new semantics.
        rule = payload["before_vs_after_gat"]["first_biomarker"]["selection_rule"]
        if "rank_in_iter==0" not in rule:
            _fail(f"first_biomarker.selection_rule should mention "
                  f"'rank_in_iter==0', got: {rule!r}")

    print("PASS verify_first_expression_baseline: finalize.py picks "
          "rank_in_iter==0 (not best-by-GAT) as the baseline; iter winner "
          "is best-by-GAT among iter>1; BASELINE_DELTA stdout uses new label")
    return 0


if __name__ == "__main__":
    sys.exit(main())
