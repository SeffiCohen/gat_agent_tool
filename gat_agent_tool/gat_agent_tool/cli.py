"""gat_agent_tool.cli — unified command-line entry point.

Subcommands:
    info         — print model metadata (features, operators, depth, etc.).
    score        — score one expression on stdin or via --expression.
    score-batch  — score many expressions from a file (one per line).
    mcp          — launch the MCP server (see mcp_server.main).
    loop-hf      — run the iterative loop with a local HuggingFace chat model.
    loop-openai  — run the agentic function-calling loop against the OpenAI API.

All subcommands accept ``--checkpoint`` pointing at a trained GAT .pt file,
or the environment variable ``GAT_AGENT_CHECKPOINT``.

Example:
    gat-agent-tool info        --checkpoint ckpt.pt
    gat-agent-tool score       --checkpoint ckpt.pt --expression "(lab_X + lab_Y) / lab_Z"
    gat-agent-tool score-batch --checkpoint ckpt.pt --input candidates.txt
    gat-agent-tool mcp         --checkpoint ckpt.pt
    gat-agent-tool loop-hf     --checkpoint ckpt.pt --llm_model_name google/medgemma-4b-it --output_csv out.csv
    gat-agent-tool loop-openai --checkpoint ckpt.pt --model gpt-4o-mini --output_dir out/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional


def _ckpt_from_args_or_env(ns: argparse.Namespace) -> str:
    val = ns.checkpoint or os.environ.get("GAT_AGENT_CHECKPOINT")
    if not val:
        raise SystemExit("--checkpoint is required (or set $GAT_AGENT_CHECKPOINT).")
    return val


def _cmd_info(ns: argparse.Namespace) -> None:
    from .core import GatScorerTool
    scorer = GatScorerTool(_ckpt_from_args_or_env(ns), device=ns.device)
    info = scorer.info()
    out = {
        "checkpoint": info.checkpoint_path,
        "model_type": info.model_type,
        "num_features": info.num_features,
        "num_operators": info.num_operators,
        "max_depth": info.max_depth,
        "graph_format_version": info.graph_format_version,
        "has_auc_transform": info.has_auc_transform,
        "operators": scorer.operators,
        "features_sample": scorer.feature_names[:20],
        "features_total": len(scorer.feature_names),
        "device": str(scorer.device),
    }
    print(json.dumps(out, indent=2))


def _cmd_score(ns: argparse.Namespace) -> None:
    from .core import GatScorerTool
    scorer = GatScorerTool(_ckpt_from_args_or_env(ns), device=ns.device)
    if ns.expression:
        expressions = [ns.expression]
    else:
        expressions = [line.strip() for line in sys.stdin if line.strip()]
    scores = scorer.score_batch(expressions)
    for expr, s in zip(expressions, scores):
        label = "NA" if s is None else f"{s:.6f}"
        print(f"{label}\t{expr}")


def _cmd_score_batch(ns: argparse.Namespace) -> None:
    from .core import GatScorerTool
    scorer = GatScorerTool(_ckpt_from_args_or_env(ns), device=ns.device)
    with open(ns.input) as f:
        expressions = [line.strip() for line in f if line.strip()]
    scores = scorer.score_batch(expressions)
    if ns.output:
        with open(ns.output, "w") as f:
            f.write("score\texpression\n")
            for expr, s in zip(expressions, scores):
                label = "" if s is None else f"{s:.6f}"
                f.write(f"{label}\t{expr}\n")
        print(f"wrote {len(expressions)} rows to {ns.output}")
    else:
        for expr, s in zip(expressions, scores):
            label = "NA" if s is None else f"{s:.6f}"
            print(f"{label}\t{expr}")


def _cmd_mcp(_ns: argparse.Namespace) -> None:
    # Delegate entirely to the MCP server's own arg parser so the CLI is a
    # thin forward — we splice `sys.argv` to remove the `mcp` subcommand token.
    from . import mcp_server
    # Preserve any extra args passed through; the MCP server has its own argparse.
    # sys.argv currently looks like: ["gat-agent-tool", "mcp", ...rest...]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    mcp_server.main()


def _cmd_loop_hf(_ns: argparse.Namespace) -> None:
    from .drivers import local_hf
    # Similar passthrough: the driver has its own argparse for many knobs.
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    local_hf.cli()


def _cmd_loop_openai(_ns: argparse.Namespace) -> None:
    from .drivers import openai_api
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    openai_api.cli()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gat-agent-tool",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- info ----
    p_info = sub.add_parser("info", help="Print metadata about a GAT checkpoint.")
    p_info.add_argument("--checkpoint", default=None)
    p_info.add_argument("--device", default="auto")
    p_info.set_defaults(func=_cmd_info)

    # ---- score ----
    p_score = sub.add_parser("score", help="Score one expression (via --expression or stdin).")
    p_score.add_argument("--checkpoint", default=None)
    p_score.add_argument("--device", default="auto")
    p_score.add_argument("--expression", default=None, help="A single expression. If omitted, read one-per-line from stdin.")
    p_score.set_defaults(func=_cmd_score)

    # ---- score-batch ----
    p_batch = sub.add_parser("score-batch", help="Score many expressions from a file (one per line).")
    p_batch.add_argument("--checkpoint", default=None)
    p_batch.add_argument("--device", default="auto")
    p_batch.add_argument("--input", required=True, help="Path to a text file, one expression per line.")
    p_batch.add_argument("--output", default=None, help="Optional TSV output file (score<TAB>expression).")
    p_batch.set_defaults(func=_cmd_score_batch)

    # Subcommands whose args belong to the subcommand's own argparse.
    # We add no-op parent parsers just so the user sees them in top-level --help.
    p_mcp = sub.add_parser("mcp", help="Launch the MCP server. Extra args pass through; see `gat-agent-mcp --help`.")
    p_mcp.set_defaults(func=_cmd_mcp)

    p_hf = sub.add_parser("loop-hf", help="Iterative loop with a local HuggingFace chat model. Extra args pass through.")
    p_hf.set_defaults(func=_cmd_loop_hf)

    p_oa = sub.add_parser("loop-openai", help="Agentic loop with an OpenAI chat model + function-calling. Extra args pass through.")
    p_oa.set_defaults(func=_cmd_loop_openai)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    # For the pass-through subcommands, we only want to consume the subcommand
    # token here — everything after it is the sub-subcommand's concern. So:
    # split argv at the first non-flag after the subcommand and re-parse only up to it.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in {"mcp", "loop-hf", "loop-openai"}:
        # parse_known_args so unknown flags don't raise; we forward them.
        ns, _ = parser.parse_known_args([argv[0]])
    else:
        ns = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ns.func(ns)


if __name__ == "__main__":
    main()
