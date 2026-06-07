"""gat_agent_tool.drivers.local_hf — iterative propose-score-refine loop with
a local HuggingFace chat model as the proposer.

The LLM never runs the GAT directly; it gets numerical feedback (predicted AUC)
inside its next prompt. This is deliberate: it works on any completion-style
model, even those without native tool-calling support (Gemma, most Llamas).
If you want a true tool-calling loop, use the OpenAI driver or the MCP server.

Typical invocation:
    gat-agent-tool loop-hf \\
        --checkpoint /path/to/gat_model.pt \\
        --llm_model_name google/medgemma-4b-it \\
        --output_csv out.csv \\
        --iterations 10 --pop_per_iter 10
    # optional — enables silent real-AUC logging:
        --data_path train.parquet --target_column 714

The output CSV has one row per proposal, sorted by ``gat_score`` descending:
    iter, rank_in_iter, expression, gat_score, real_auc, is_valid, duplicate_of
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from ..core import GatScorerTool  # noqa: F401

from ..validation import parse_candidates


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
_DEFAULT_TOPIC = (
    "biomarker expressions for clinical prediction from routine lab features"
)


def build_prompt(
    topic: str,
    allowed_features: List[str],
    operators: List[str],
    max_depth: int,
    history: List[Dict[str, Any]],
    pop_per_iter: int,
    max_history_shown: int,
) -> str:
    """Assemble the plaintext prompt fed to the LLM each iteration.

    Half the history slots hold the best-ever-scored proposals (global memory);
    the other half hold the most recent iteration's batch (recency). This keeps
    the prompt bounded across long runs.
    """
    feat_line = ", ".join(allowed_features[:60])
    feature_block = (
        "Use ONLY these features, case-sensitive:\n  " + feat_line
    )
    if len(allowed_features) > 60:
        feature_block += (
            "\n  " + ", ".join(allowed_features[60:])
        )

    history_block = ""
    scored = [h for h in history if h.get("gat_score") is not None]
    if scored:
        half = max(1, max_history_shown // 2)
        top = sorted(scored, key=lambda h: h["gat_score"], reverse=True)[:half]
        last_iter = max(h["iter"] for h in history)
        recent = [h for h in history if h["iter"] == last_iter][:half]
        shown: Dict[str, Dict[str, Any]] = {}
        for h in top + recent:
            shown.setdefault(h["expression"], h)
        lines = []
        for h in sorted(shown.values(), key=lambda h: (h.get("gat_score") or 0), reverse=True):
            score = h.get("gat_score")
            s = "  NA " if score is None else f"{score:.3f}"
            lines.append(f"  {s}   {h['expression']}")
        history_block = (
            "\nPreviously scored (higher predicted AUC is better, max 1.000):\n"
            + "\n".join(lines)
            + "\n"
        )

    ops_line = " ".join(operators)
    return (
        f"You are designing {topic}.\n"
        f"A graph-attention-network (GAT) will score each expression and "
        f"return a predicted AUC in [0.5, 1.0]. Your goal is to propose "
        f"expressions that score as high as possible.\n\n"
        f"{feature_block}\n\n"
        f"Allowed arithmetic operators: {ops_line}\n"
        f"Rules:\n"
        f"  - NO numeric constants of any kind (feature arithmetic only).\n"
        f"  - Combine 2 to 5 features; keep tree depth <= {max_depth}.\n"
        f"  - Parenthesize so precedence is unambiguous.\n"
        f"  - Spell feature names EXACTLY as written above.\n"
        f"{history_block}\n"
        f"Propose {pop_per_iter} NEW expressions that should score higher than "
        f"any above. Do not repeat any expression verbatim.\n"
        f"Output ONE expression per line. No numbering, no bullets, no prose."
    )


# ---------------------------------------------------------------------------
# HF chat model wrapper
# ---------------------------------------------------------------------------
def load_hf_chat(model_name: str):
    """Load an HF AutoModelForCausalLM + tokenizer. Raises a clean SystemExit
    with a pip-install hint if ``transformers`` isn't available."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "The `transformers` package is required for the local-HF driver. "
            "Install with: pip install 'gat-agent-tool[hf]'"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )
    model.eval()
    return tokenizer, model


def hf_generate(
    tokenizer,
    model,
    prompt: str,
    *,
    num_candidates: int,
    temperature: float,
    max_new_tokens: int = 512,
    top_p: float = 0.95,
) -> List[str]:
    """Return `num_candidates` independently sampled completions for `prompt`.
    Uses the tokenizer's chat template if one is defined, else treats the
    prompt as raw completion input."""
    import torch

    use_chat = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    )
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = prompt

    enc = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            num_return_sequences=int(num_candidates),
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    texts: List[str] = []
    for seq in out_ids:
        generated = seq[prompt_len:]
        texts.append(tokenizer.decode(generated, skip_special_tokens=True))
    return texts


# ---------------------------------------------------------------------------
# Optional real-AUC computation (silent — never shown to the LLM)
# ---------------------------------------------------------------------------
def _compute_real_auc(
    expression: str,
    df,  # pandas DataFrame
    target_col: str,
    allowed_cols: List[str],
) -> Optional[float]:
    """Evaluate ``expression`` on ``df`` and return max(AUC, 1-AUC).

    Returns None if the expression can't be evaluated numerically or if
    asteval / sklearn are not installed.
    """
    try:
        import numpy as np
        import pandas as pd
        from asteval import Interpreter  # type: ignore
        from sklearn.metrics import roc_auc_score  # type: ignore
    except ImportError:
        return None

    try:
        symtable: Dict[str, Any] = {col: df[col].values for col in allowed_cols if col in df.columns}
        symtable.update({"np": np})
        aeval = Interpreter(symtable=symtable, err_writer=None, use_numpy=True)
        vals = aeval.eval(expression)
        if aeval.error or vals is None:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        if np.isscalar(arr) or arr.shape != (len(df),):
            return None
        mask = ~np.isnan(arr) & ~np.isinf(arr) & ~pd.Series(df[target_col]).isna().values
        if mask.sum() < 10:
            return None
        y = df[target_col].values[mask]
        x = arr[mask]
        if len(set(y.tolist())) < 2:
            return None
        auc = roc_auc_score(y, x)
        return float(max(auc, 1.0 - auc))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------
@dataclass
class IterationStats:
    iter: int
    n_proposed: int
    n_valid: int
    n_duplicates: int
    best_gat_in_iter: Optional[float]
    best_gat_ever: Optional[float]
    best_real_in_iter: Optional[float]
    best_real_ever: Optional[float]
    seconds: float


def run_iterative_hf_loop(
    *,
    scorer: "GatScorerTool",
    tokenizer,
    model,
    output_csv: str,
    iterations: int = 10,
    pop_per_iter: int = 10,
    max_history_shown: int = 20,
    plateau_patience: int = 3,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
    topic: str = _DEFAULT_TOPIC,
    data_df=None,  # optional pandas DataFrame for real-AUC logging
    target_column: Optional[str] = None,
    allowed_features_override: Optional[List[str]] = None,
) -> Tuple[Any, List[IterationStats]]:
    """Execute the iterative propose-score-refine loop.

    Parameters
    ----------
    scorer : GatScorerTool
        The preloaded GAT. The feature list comes from this.
    tokenizer, model
        Preloaded HuggingFace tokenizer / AutoModelForCausalLM.
    output_csv : str
        Destination for the per-proposal CSV (sorted by gat_score desc).
    iterations : int
        Max number of propose-score rounds.
    pop_per_iter : int
        Expressions proposed per round.
    max_history_shown : int
        Prior-round proposals fed back into the prompt (bounded).
    plateau_patience : int
        Stop early if the best GAT score doesn't improve for this many rounds.
        Set to 0 to disable.
    temperature, max_new_tokens : float, int
        LLM generation knobs.
    topic : str
        Plaintext string substituted into the prompt (default: biomarker).
    data_df, target_column : optional
        If provided, compute `real_auc` silently for each proposal (never put
        into the LLM prompt) and record in the CSV.
    allowed_features_override : optional
        Constrain the feature list shown to the LLM to a subset (default: all
        features the GAT was trained on).

    Returns
    -------
    (pandas.DataFrame, List[IterationStats])
        The proposal log as a DataFrame, sorted by gat_score desc, plus the
        per-iteration stats.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("pandas is required for the local-HF driver.") from e
    import numpy as np  # noqa: F401  (used in random-seed path)
    import torch

    torch.manual_seed(42)

    features = allowed_features_override or scorer.feature_names
    if not features:
        raise RuntimeError("GAT checkpoint exposes no features — cannot prompt the LLM.")
    operators = scorer.operators
    max_depth = scorer.max_depth

    # If real-AUC logging is enabled, restrict to features that exist in the data.
    real_auc_features: List[str] = []
    if data_df is not None and target_column is not None:
        if target_column not in data_df.columns:
            raise ValueError(
                f"target_column {target_column!r} not in data_df columns "
                f"(first 10: {list(data_df.columns)[:10]})"
            )
        real_auc_features = [f for f in features if f in data_df.columns]
        if not real_auc_features:
            logger.warning(
                "None of the GAT's features are present in data_df; "
                "real_auc will always be None."
            )

    history: List[Dict[str, Any]] = []
    seen_exprs: Dict[str, int] = {}
    per_iter_stats: List[IterationStats] = []
    best_gat_ever: Optional[float] = None
    best_real_ever: Optional[float] = None
    no_improve = 0

    for it in range(int(iterations)):
        t0 = time.time()
        prompt = build_prompt(
            topic, features, operators, max_depth,
            history, pop_per_iter, max_history_shown,
        )
        logger.info(
            "[iter %d] prompt=%d chars  history=%d  plateau=%d/%d",
            it, len(prompt), len(history), no_improve, plateau_patience,
        )

        raw_texts = hf_generate(
            tokenizer, model, prompt,
            num_candidates=pop_per_iter,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        candidates: List[str] = []
        for t in raw_texts:
            candidates.extend(parse_candidates(t, features, operators))

        in_iter_seen: set = set()
        novel: List[str] = []
        duplicates: Dict[str, int] = {}
        for expr in candidates:
            if expr in in_iter_seen:
                continue
            in_iter_seen.add(expr)
            if expr in seen_exprs:
                duplicates[expr] = seen_exprs[expr]
                continue
            novel.append(expr)

        gat_scores = scorer.score_batch(novel)

        best_gat_in_iter: Optional[float] = None
        best_real_in_iter: Optional[float] = None
        rank_idx = 0

        for expr, g in zip(novel, gat_scores):
            is_valid = g is not None
            real_auc: Optional[float] = None
            if is_valid and real_auc_features:
                real_auc = _compute_real_auc(expr, data_df, target_column, real_auc_features)
            history.append({
                "iter": it,
                "rank_in_iter": rank_idx,
                "expression": expr,
                "gat_score": g,
                "real_auc": real_auc,
                "is_valid": bool(is_valid),
                "duplicate_of": None,
            })
            seen_exprs[expr] = it
            rank_idx += 1
            if g is not None and (best_gat_in_iter is None or g > best_gat_in_iter):
                best_gat_in_iter = g
            if real_auc is not None and (best_real_in_iter is None or real_auc > best_real_in_iter):
                best_real_in_iter = real_auc

        for expr, first_iter in duplicates.items():
            prior = next((h for h in history if h["expression"] == expr), None)
            history.append({
                "iter": it,
                "rank_in_iter": rank_idx,
                "expression": expr,
                "gat_score": prior["gat_score"] if prior else None,
                "real_auc": prior["real_auc"] if prior else None,
                "is_valid": prior["is_valid"] if prior else False,
                "duplicate_of": int(first_iter),
            })
            rank_idx += 1

        prev_best = best_gat_ever
        if best_gat_in_iter is not None and (best_gat_ever is None or best_gat_in_iter > best_gat_ever):
            best_gat_ever = best_gat_in_iter
        if best_real_in_iter is not None and (best_real_ever is None or best_real_in_iter > best_real_ever):
            best_real_ever = best_real_in_iter
        if prev_best is not None and best_gat_ever is not None and best_gat_ever <= prev_best:
            no_improve += 1
        else:
            no_improve = 0

        per_iter_stats.append(IterationStats(
            iter=it,
            n_proposed=len(candidates),
            n_valid=sum(1 for g in gat_scores if g is not None),
            n_duplicates=len(duplicates),
            best_gat_in_iter=best_gat_in_iter,
            best_gat_ever=best_gat_ever,
            best_real_in_iter=best_real_in_iter,
            best_real_ever=best_real_ever,
            seconds=round(time.time() - t0, 2),
        ))
        logger.info(
            "[iter %d] proposed=%d novel=%d valid=%d dup=%d  gat_best_iter=%s gat_best_ever=%s  real_best_iter=%s real_best_ever=%s  (%.1fs)",
            it, len(candidates), len(novel),
            sum(1 for g in gat_scores if g is not None), len(duplicates),
            f"{best_gat_in_iter:.3f}" if best_gat_in_iter is not None else "NA",
            f"{best_gat_ever:.3f}" if best_gat_ever is not None else "NA",
            f"{best_real_in_iter:.3f}" if best_real_in_iter is not None else "NA",
            f"{best_real_ever:.3f}" if best_real_ever is not None else "NA",
            time.time() - t0,
        )

        if plateau_patience and no_improve >= plateau_patience:
            logger.info("[iter %d] plateau — stopping.", it)
            break

    out_df = pd.DataFrame(history)
    if not out_df.empty:
        out_df = out_df.sort_values(
            by=["gat_score"], ascending=False, na_position="last"
        ).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), output_csv)
    return out_df, per_iter_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="gat-agent-tool loop-hf",
        description="Iterative LLM (local HuggingFace) + GAT tool-use loop.",
    )
    p.add_argument("--checkpoint", required=True, help="Path to the GAT .pt checkpoint.")
    p.add_argument("--llm_model_name", default="google/medgemma-4b-it")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--topic", default=_DEFAULT_TOPIC)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--pop_per_iter", type=int, default=10)
    p.add_argument("--max_history_shown", type=int, default=20)
    p.add_argument("--plateau_patience", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device", default="auto")

    # Optional real-AUC logging
    p.add_argument("--data_path", default=None,
                   help="Optional parquet/csv with per-sample data to enable silent real-AUC logging.")
    p.add_argument("--target_column", default=None,
                   help="Column name of the binary target in --data_path. Required if --data_path is set.")

    p.add_argument("--log_level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Deferred import: torch_geometric is only needed once we actually run.
    from ..core import GatScorerTool
    scorer = GatScorerTool(args.checkpoint, device=args.device)
    tokenizer, model = load_hf_chat(args.llm_model_name)

    data_df = None
    if args.data_path:
        if not args.target_column:
            raise SystemExit("--target_column is required when --data_path is provided")
        try:
            import pandas as pd
        except ImportError as e:
            raise SystemExit("pandas is required for --data_path") from e
        if args.data_path.endswith(".parquet"):
            data_df = pd.read_parquet(args.data_path)
        else:
            data_df = pd.read_csv(args.data_path)

    try:
        out_df, stats = run_iterative_hf_loop(
            scorer=scorer,
            tokenizer=tokenizer,
            model=model,
            output_csv=args.output_csv,
            iterations=args.iterations,
            pop_per_iter=args.pop_per_iter,
            max_history_shown=args.max_history_shown,
            plateau_patience=args.plateau_patience,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            topic=args.topic,
            data_df=data_df,
            target_column=args.target_column,
        )
    finally:
        try:
            import gc
            import torch  # type: ignore
            if hasattr(model, "cpu"):
                model.cpu()
            del model, tokenizer, scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("cleanup warning: %s", e)

    top_row = out_df.iloc[0] if not out_df.empty else None
    summary = {
        "output_csv": args.output_csv,
        "n_rows": int(len(out_df)),
        "iterations_run": len(stats),
        "top1_gat_score": float(top_row["gat_score"]) if top_row is not None and top_row["gat_score"] == top_row["gat_score"] else None,
        "top1_real_auc":  float(top_row["real_auc"]) if top_row is not None and top_row["real_auc"] == top_row["real_auc"] else None,
        "top1_expression": top_row["expression"] if top_row is not None else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli()
