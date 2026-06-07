"""gat_agent_tool.mcp_server — expose a trained GAT as an MCP tool server.

MCP (Model Context Protocol) is the open standard Claude Code, Claude Desktop,
and modern ChatGPT (via connectors) all speak. Launching this server lets any of
those agents call the GAT as an external tool without writing consumer-specific
glue.

Tools exposed:
  * `score_expression(expression)`       — single expression → predicted AUC.
  * `score_expressions(expressions)`     — batch version.
  * `get_feature_list()`                 — lists legal features, operators, rules;
                                           the LLM should call this first.
  * `get_model_info()`                   — checkpoint metadata (debugging).

Launch:
    gat-agent-mcp --checkpoint /path/to/gat_model.pt

Or via the unified CLI:
    gat-agent-tool mcp --checkpoint /path/to/gat_model.pt

Example Claude Code config (.mcp.json or user settings):
    {
      "mcpServers": {
        "gat-scorer": {
          "command": "gat-agent-mcp",
          "args": ["--checkpoint", "/path/to/gat_model.pt"]
        }
      }
    }

IMPORTANT: All logging MUST go to stderr. MCP uses stdout for the JSON-RPC
protocol — a single print() to stdout will corrupt the connection.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    # Static type-only import — never runs at import time, so --help doesn't
    # need torch_geometric installed.
    from .core import GatScorerTool


logger = logging.getLogger(__name__)


# The GatScorerTool instance shared by all tool calls. Populated by main() from
# the --checkpoint arg before the MCP server enters its event loop.
_scorer: "Optional[GatScorerTool]" = None


def _require_scorer() -> "GatScorerTool":
    if _scorer is None:
        raise RuntimeError(
            "MCP server misconfigured: GatScorerTool not initialized. "
            "Launch via main() so --checkpoint is parsed first."
        )
    return _scorer


# ---------------------------------------------------------------------------
# FastMCP server. We import FastMCP lazily inside main() so that `--help` works
# even when the `mcp` package is not installed.
# ---------------------------------------------------------------------------
def _register_tools(mcp: Any) -> None:
    """Attach tool handlers to the FastMCP instance. Called from main() after
    we've confirmed `mcp` is importable and loaded the GAT checkpoint."""

    @mcp.tool()
    def score_expression(expression: str) -> Dict[str, Any]:
        """Score a single biomarker expression with the trained GAT.

        The expression must be feature-only arithmetic (no numeric constants),
        using feature names returned by `get_feature_list()` and operators
        ``+ - * /``. Depth should be <= the checkpoint's ``max_depth``.

        Returns a dict with:
            score: float in [0.5, 1.0], or null if the expression can't be parsed.
            error: null on success, or a short human-readable reason on failure.
            expression: echo of the input, for convenience.
        """
        s = _require_scorer()
        try:
            score = s.score(expression)
            return {
                "expression": expression,
                "score": score,
                "error": None if score is not None else "could not parse expression into a graph",
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("score_expression failed for %r", expression)
            return {"expression": expression, "score": None, "error": str(e)}

    @mcp.tool()
    def score_expressions(expressions: List[str]) -> List[Dict[str, Any]]:
        """Score many expressions in one call (batched through the GAT for speed).

        Use this when you want to evaluate an entire proposal round in a single
        tool call — much faster than calling `score_expression` N times.

        Returns a list of `{expression, score, error}` dicts, one per input,
        in the same order as the input list.
        """
        s = _require_scorer()
        try:
            scores = s.score_batch(list(expressions))
            return [
                {
                    "expression": e,
                    "score": sc,
                    "error": None if sc is not None else "could not parse expression into a graph",
                }
                for e, sc in zip(expressions, scores)
            ]
        except Exception as e:  # noqa: BLE001
            logger.exception("score_expressions failed")
            return [{"expression": e_str, "score": None, "error": str(e)} for e_str in expressions]

    @mcp.tool()
    def get_feature_list() -> Dict[str, Any]:
        """Return the legal features, operators, and rules for constructing
        expressions. Call this BEFORE proposing anything so every expression
        you build references only features the GAT has seen.

        Returns:
            features: list of feature-name strings (case-sensitive).
            operators: list of arithmetic operators (``+``, ``-``, ``*``, ``/``).
            max_depth: int; expressions deeper than this risk poor generalisation.
            model_type: which GAT variant is loaded (informational).
            rules: list of plain-language constraints to pass into the LLM system prompt.
            score_range: [0.5, 1.0] — the tool's output interval.
        """
        s = _require_scorer()
        info = s.info()
        return {
            "features": s.feature_names,
            "operators": s.operators,
            "max_depth": info.max_depth,
            "model_type": info.model_type,
            "rules": [
                "Every expression must reference at least one feature from `features`, spelled EXACTLY as listed (case-sensitive).",
                "Numeric constants are NOT allowed — the score will be null for any expression containing a literal number.",
                f"Keep tree depth <= {info.max_depth} — deeper expressions were not seen during GAT training.",
                "Only arithmetic operators `+ - * /` are supported (no functions, conditionals, or comparisons).",
                "Parenthesize liberally; precedence follows Python / standard maths.",
            ],
            "score_range": [0.5, 1.0],
            "higher_is_better": True,
        }

    @mcp.tool()
    def get_model_info() -> Dict[str, Any]:
        """Return metadata about the loaded GAT checkpoint (for debugging).

        Does not execute any scoring — use this to confirm the server is
        wrapping the checkpoint you expected.
        """
        s = _require_scorer()
        info = s.info()
        return {
            "model_type": info.model_type,
            "num_features": info.num_features,
            "num_operators": info.num_operators,
            "max_depth": info.max_depth,
            "graph_format_version": info.graph_format_version,
            "has_auc_transform": info.has_auc_transform,
            "checkpoint_path": info.checkpoint_path,
            "device": str(s.device),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gat-agent-mcp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("GAT_AGENT_CHECKPOINT"),
        help="Path to the .pt GAT checkpoint. Falls back to $GAT_AGENT_CHECKPOINT.",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', or 'cuda:N'.")
    parser.add_argument(
        "--server_name",
        default="gat-scorer",
        help="Name announced to MCP clients. Change this if you run multiple "
             "servers side-by-side (one per disease, say).",
    )
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("--checkpoint is required (or set $GAT_AGENT_CHECKPOINT).")

    # CRITICAL: log to stderr. MCP uses stdout for protocol traffic.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # Load the MCP SDK here (not at module top) so `--help` works without it.
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        sys.stderr.write(
            "[gat-agent-mcp] The `mcp` package is required. Install with:\n"
            "    pip install 'gat-agent-tool[mcp]'\n"
            "or directly:\n"
            "    pip install 'mcp>=1.0'\n"
        )
        raise SystemExit(1) from e

    # Same deferral for the core: importing it pulls in torch / torch_geometric.
    from .core import GatScorerTool

    global _scorer
    logger.info("Loading GAT checkpoint from %s (device=%s)…", args.checkpoint, args.device)
    _scorer = GatScorerTool(args.checkpoint, device=args.device)
    logger.info(
        "GAT ready: %s model, %d features, %d operators, max_depth=%d",
        _scorer.model_type, len(_scorer.feature_names),
        len(_scorer.operators), _scorer.max_depth,
    )

    mcp = FastMCP(args.server_name)
    _register_tools(mcp)

    logger.info("Starting MCP server '%s' on stdio…", args.server_name)
    mcp.run()  # blocks forever; default transport is stdio.


if __name__ == "__main__":
    main()
