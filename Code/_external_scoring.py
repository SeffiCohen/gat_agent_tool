"""Shared expression-evaluation helpers for external-cohort validation.

Factored out of ``score_ehrshot_external.py`` so multiple per-cohort scoring
scripts (EHRShot, NHANES, ...) share one implementation of:

  - :func:`repair_implicit_multiplication` — patch LaTeX-ish LLM output into
    Python-evaluable form (``lastlab_`` → ``last*lab_``, etc.).
  - :func:`eval_expression` — asteval-based safe evaluation of ``lab_``-only
    arithmetic expressions over a DataFrame.
  - :func:`univariate_auc` — direction-agnostic univariate AUC in [0.5, 1.0].
  - :func:`bootstrap_auc_ci` — percentile bootstrap confidence interval for
    :func:`univariate_auc`.

No behavior change versus the original inline EHRShot definitions; this module
exists purely to avoid copy-paste duplication when adding new external cohorts.
"""
from __future__ import annotations

import logging
import re
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def repair_implicit_multiplication(expr: str) -> str:
    """Insert missing ``*`` operators that LLM producers elided as LaTeX-style
    implicit multiplication.

    Some LLM producers (notably GPT Deep Research on T1D / RA) emitted expressions
    like ``lab_X_lastlab_Y_last`` (two identifiers run together) or
    ``(lab_A_last)(lab_B_last)`` (adjacent parenthesized groups), which are
    mathematically unambiguous in LaTeX but invalid Python. We repair them:
      - ``lastlab_``  →  ``last*lab_``
      - ``)(``        →  ``)*(``
      - ``)X``        →  ``)*X``    (X = letter / underscore)
      - ``X(``        →  ``X*(``    (X = alphanumeric or underscore)
    Never changes a well-formed expression.
    """
    prev = None
    cur = expr
    while prev != cur:
        prev = cur
        cur = cur.replace("lastlab_", "last*lab_")
    cur = re.sub(r"\)\s*\(", ")*(", cur)
    cur = re.sub(r"\)(?=[A-Za-z_])", ")*", cur)
    cur = re.sub(r"([A-Za-z0-9_])(?=\()", r"\1*", cur)
    return cur


def eval_expression(expr: str, df: pd.DataFrame) -> Optional[np.ndarray]:
    """Evaluate arithmetic expression over ``df``'s ``lab_*`` columns using asteval.

    Returns a 1-D numpy array (len == len(df)) or None on failure. Attempts
    repair via :func:`repair_implicit_multiplication` on parse failure before
    giving up.
    """
    try:
        from asteval import Interpreter
    except ImportError:
        logging.error("asteval not available - install with `pip install asteval`")
        return None

    lab_cols = [c for c in df.columns if c.startswith("lab_")]
    symtable: Dict[str, np.ndarray] = {c: df[c].to_numpy(dtype=np.float64) for c in lab_cols}
    symtable.update({
        "np": np, "log": np.log, "exp": np.exp, "sqrt": np.sqrt,
        "maximum": np.maximum, "minimum": np.minimum, "abs": np.abs,
        "power": np.power, "where": np.where,
    })

    def _try(expr_variant: str) -> Optional[np.ndarray]:
        aeval = Interpreter(symtable=dict(symtable), err_writer=None, use_numpy=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = aeval.eval(expr_variant)
        if aeval.error or val is None:
            return None
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(len(df), float(arr), dtype=np.float64)
        if arr.shape != (len(df),):
            return None
        arr[np.isinf(arr)] = np.nan
        return arr

    out = _try(expr)
    if out is not None:
        return out
    repaired = repair_implicit_multiplication(expr)
    if repaired != expr:
        return _try(repaired)
    return None


def univariate_auc(values: np.ndarray, target: np.ndarray) -> float:
    """Direction-agnostic univariate AUC = max(AUC, 1-AUC), in [0.5, 1.0].

    Returns ``NaN`` when AUC is undefined (fewer than 10 valid observations, or
    ``target`` has only one class). Callers should propagate NaN as "undefined"
    rather than substitute 0.5 — the downstream summary aggregations use
    ``.max()`` / ``.mean()`` which drop NaN.
    """
    mask = ~(np.isnan(values) | np.isnan(target))
    if mask.sum() < 10:
        return float("nan")
    t, v = target[mask], values[mask]
    if len(np.unique(t)) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            auc = roc_auc_score(t, v)
        except ValueError:
            return float("nan")
    if not np.isfinite(auc):
        return float("nan")
    return float(max(auc, 1.0 - auc))


def bootstrap_auc_ci(
    values: np.ndarray,
    target: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean AUC, 2.5-percentile, 97.5-percentile) from bootstrap resamples.

    Returns ``(NaN, NaN, NaN)`` when the bootstrap is undefined (too few samples,
    single-class target, or every resample collapses to a single class).
    """
    mask = ~(np.isnan(values) | np.isnan(target))
    if mask.sum() < 10:
        return (float("nan"), float("nan"), float("nan"))
    v, t = values[mask], target[mask]
    if len(np.unique(t)) < 2:
        return (float("nan"), float("nan"), float("nan"))
    n = len(v)
    rng = np.random.default_rng(seed)
    boots = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                auc = roc_auc_score(t[idx], v[idx])
                if np.isfinite(auc):
                    boots.append(max(auc, 1.0 - auc))
            except ValueError:
                continue
    if not boots:
        return (float("nan"), float("nan"), float("nan"))
    boots_arr = np.array(boots)
    return (
        float(np.mean(boots_arr)),
        float(np.percentile(boots_arr, 2.5)),
        float(np.percentile(boots_arr, 97.5)),
    )
