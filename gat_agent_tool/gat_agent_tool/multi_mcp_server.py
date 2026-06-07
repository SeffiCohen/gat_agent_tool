"""gat_agent_tool.multi_mcp_server — multi-disease MCP variant of `gat-agent-mcp`.

Where :mod:`gat_agent_tool.mcp_server` exposes a single trained GAT checkpoint,
this module exposes a registry of per-disease checkpoints behind one MCP
endpoint. Every tool takes a ``disease_id`` first (except ``list_diseases``) so
an agent can score expressions against any of the BD_Paper phenotypes from a
single connection — no need to launch one server per ICD code.

Tools exposed:
  * `list_diseases()`                                     — enumerate available disease IDs.
  * `get_feature_list(disease_id)`                        — legal features, operators, rules.
  * `get_model_info(disease_id)`                          — checkpoint metadata (debugging).
  * `score_expression(disease_id, expression)`            — single expression → predicted AUC.
  * `score_expressions(disease_id, expressions)`          — batched form.

Launch:
    gat-agent-multi-mcp --gat-root /path/to/BD_Paper/GAT
    gat-agent-multi-mcp --registry-config /path/to/registry.json

Example Claude Code config (.mcp.json or user settings):
    {
      "mcpServers": {
        "gat-multi-scorer": {
          "command": "gat-agent-multi-mcp",
          "args": ["--gat-root", "/path/to/BD_Paper/GAT"]
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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    # Static type-only import — never runs at import time, so --help doesn't
    # need torch_geometric installed.
    from .registry import GatScorerRegistry


logger = logging.getLogger(__name__)


# The GatScorerRegistry instance shared by all tool calls. Populated by main()
# from --gat-root / --registry-config before the MCP server enters its event loop.
_registry: "Optional[GatScorerRegistry]" = None

# Human-readable display strings reported by `list_diseases()`. Set in main().
# Initialized to placeholders so the tool stays well-formed if some test path
# registers tools without going through the CLI entry point.
_gat_root_for_display: str = "<unset>"
_checkpoint_filename_for_display: str = "best_by_loss.pt"


def _require_registry() -> "GatScorerRegistry":
    if _registry is None:
        raise RuntimeError(
            "Multi-MCP server misconfigured: GatScorerRegistry not initialized. "
            "Launch via main() so --gat-root or --registry-config is parsed first."
        )
    return _registry


# ---------------------------------------------------------------------------
# FastMCP server. We import FastMCP lazily inside main() so that `--help` works
# even when the `mcp` package is not installed.
# ---------------------------------------------------------------------------
def _register_tools(mcp: Any) -> None:
    """Attach tool handlers to the FastMCP instance. Called from main() after
    we've confirmed `mcp` is importable and built the GatScorerRegistry."""

    @mcp.tool()
    def list_diseases() -> Dict[str, Any]:
        """Return the disease IDs this server can score against.

        Call this first in a multi-disease session so subsequent tool calls
        use a `disease_id` the registry actually knows. The ``gat_root`` and
        ``default_checkpoint_filename`` fields are informational — they tell
        the caller where checkpoints came from.
        """
        reg = _require_registry()
        # Prefer the registry's own `_gat_root` attribute when it exposes one
        # (the FakeRegistry in the test suite does; the production
        # GatScorerRegistry currently does not, so we fall back to the value
        # captured at CLI parse time).
        gat_root = getattr(reg, "_gat_root", None) or _gat_root_for_display
        return {
            "diseases": reg.list_diseases(),
            "gat_root": gat_root,
            "default_checkpoint_filename": _checkpoint_filename_for_display,
        }

    @mcp.tool()
    def get_feature_list(disease_id: str) -> Dict[str, Any]:
        """Return the legal features, operators, and rules for one disease.

        Call this BEFORE proposing expressions for ``disease_id`` so every
        feature you reference matches what that disease's GAT was trained on.
        If ``disease_id`` is unknown, returns an error envelope (not an
        exception) listing the available IDs.

        Returns on success:
            disease_id: echo of the input.
            features: list of feature-name strings (case-sensitive).
            operators: list of arithmetic operators (``+``, ``-``, ``*``, ``/``).
            max_depth: int; expressions deeper than this risk poor generalisation.
            model_type: which GAT variant is loaded (informational).
            rules: list of plain-language constraints to pass into the LLM system prompt.
            score_range: [0.5, 1.0] — the tool's output interval.
            higher_is_better: True (predicted AUC).
        """
        reg = _require_registry()
        if disease_id not in reg.list_diseases():
            return {
                "disease_id": disease_id,
                "error": f"unknown disease_id={disease_id!r}",
                "available": reg.list_diseases(),
            }
        try:
            info = reg.info(disease_id)
            features = reg.feature_names(disease_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("get_feature_list failed for %r", disease_id)
            return {
                "disease_id": disease_id,
                "error": str(e),
                "available": reg.list_diseases(),
            }
        max_depth = info["max_depth"]
        return {
            "disease_id": disease_id,
            "features": features,
            # Operators are constants in core.py (`["+", "-", "*", "/"]`).
            # Hardcode them here so this tool can answer without forcing a
            # checkpoint load via reg.get(disease_id).operators.
            "operators": ["+", "-", "*", "/"],
            "max_depth": max_depth,
            "model_type": info["model_type"],
            "rules": [
                "Every expression must reference at least one feature from `features`, spelled EXACTLY as listed (case-sensitive).",
                "Numeric constants are NOT allowed — the score will be null for any expression containing a literal number.",
                f"Keep tree depth <= {max_depth} — deeper expressions were not seen during GAT training.",
                "Only arithmetic operators `+ - * /` are supported (no functions, conditionals, or comparisons).",
                "Parenthesize liberally; precedence follows Python / standard maths.",
            ],
            "score_range": [0.5, 1.0],
            "higher_is_better": True,
        }

    @mcp.tool()
    def get_model_info(disease_id: str) -> Dict[str, Any]:
        """Return metadata about ``disease_id``'s loaded GAT checkpoint.

        Does not execute any scoring — use this to confirm the server is
        wrapping the checkpoint you expected. Unknown ``disease_id`` returns
        an error envelope (not an exception).
        """
        reg = _require_registry()
        if disease_id not in reg.list_diseases():
            return {
                "disease_id": disease_id,
                "error": f"unknown disease_id={disease_id!r}",
                "available": reg.list_diseases(),
            }
        try:
            info = reg.info(disease_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("get_model_info failed for %r", disease_id)
            return {
                "disease_id": disease_id,
                "error": str(e),
                "available": reg.list_diseases(),
            }
        return {
            "disease_id": disease_id,
            "model_type": info["model_type"],
            "num_features": info["num_features"],
            "num_operators": info["num_operators"],
            "max_depth": info["max_depth"],
            "graph_format_version": info["graph_format_version"],
            "has_auc_transform": info["has_auc_transform"],
            "checkpoint_path": info["checkpoint_path"],
            "device": info.get("device", "auto"),
        }

    @mcp.tool()
    def score_expression(disease_id: str, expression: str) -> Dict[str, Any]:
        """Score a single biomarker expression against ``disease_id``'s GAT.

        The expression must be feature-only arithmetic (no numeric constants),
        using feature names returned by `get_feature_list(disease_id)` and
        operators ``+ - * /``. Depth should be <= the checkpoint's
        ``max_depth``.

        Returns a dict with:
            disease_id: echo of the input.
            expression: echo of the input.
            score: float in [0.5, 1.0], or null if the expression can't be parsed.
            error: null on success, or a short human-readable reason on failure
                   (including unknown disease_id, in which case `available` is
                   also populated).
        """
        reg = _require_registry()
        if disease_id not in reg.list_diseases():
            return {
                "disease_id": disease_id,
                "expression": expression,
                "score": None,
                "error": f"unknown disease_id={disease_id!r}",
                "available": reg.list_diseases(),
            }
        try:
            score = reg.score(disease_id, expression)
            return {
                "disease_id": disease_id,
                "expression": expression,
                "score": score,
                "error": None if score is not None else "could not parse expression into a graph",
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("score_expression failed for %r/%r", disease_id, expression)
            return {
                "disease_id": disease_id,
                "expression": expression,
                "score": None,
                "error": str(e),
            }

    @mcp.tool()
    def score_expressions(disease_id: str, expressions: List[str]) -> List[Dict[str, Any]]:
        """Score many expressions in one call against ``disease_id``'s GAT.

        Use this when you want to evaluate an entire proposal round in a
        single tool call — much faster than calling `score_expression` N
        times. Output length and order match the input list. Unknown
        ``disease_id`` returns one error envelope per input rather than
        raising.
        """
        reg = _require_registry()
        if disease_id not in reg.list_diseases():
            return [
                {
                    "disease_id": disease_id,
                    "expression": e,
                    "score": None,
                    "error": f"unknown disease_id={disease_id!r}",
                    "available": reg.list_diseases(),
                }
                for e in expressions
            ]
        try:
            scores = reg.score_batch(disease_id, list(expressions))
            return [
                {
                    "disease_id": disease_id,
                    "expression": e,
                    "score": s,
                    "error": None if s is not None else "could not parse expression into a graph",
                }
                for e, s in zip(expressions, scores)
            ]
        except Exception as e:  # noqa: BLE001
            logger.exception("score_expressions failed for %r", disease_id)
            return [
                {
                    "disease_id": disease_id,
                    "expression": e_str,
                    "score": None,
                    "error": str(e),
                }
                for e_str in expressions
            ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gat-agent-multi-mcp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--gat-root",
        default=None,
        help="Directory containing per-disease subfolders (e.g. BD_Paper/GAT/). "
             "Auto-discovers <gat_root>/<icd>/trained_models/<checkpoint-filename>.",
    )
    src.add_argument(
        "--registry-config",
        default=None,
        help="Path to JSON {disease_id: checkpoint_path} mapping.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default="best_by_loss.pt",
        help="Filename to look for in each <icd>/trained_models/ folder.",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', or 'cuda:N'.")
    parser.add_argument(
        "--server_name",
        default="gat-multi-scorer",
        help="Name announced to MCP clients.",
    )
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Preload all checkpoints at startup (default: lazy).",
    )
    args = parser.parse_args()

    gat_root = args.gat_root or os.environ.get("GAT_AGENT_REGISTRY_ROOT")
    registry_config_path = args.registry_config

    if not gat_root and not registry_config_path:
        parser.error(
            "provide --gat-root, --registry-config, or set $GAT_AGENT_REGISTRY_ROOT"
        )

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
            "[gat-agent-multi-mcp] The `mcp` package is required. Install with:\n"
            "    pip install 'gat-agent-tool[mcp]'\n"
            "or directly:\n"
            "    pip install 'mcp>=1.0'\n"
        )
        raise SystemExit(1) from e

    # Same deferral for the registry: it pulls in torch / torch_geometric on
    # first scorer build.
    from .registry import GatScorerRegistry

    global _registry, _gat_root_for_display, _checkpoint_filename_for_display
    _checkpoint_filename_for_display = args.checkpoint_filename
    if registry_config_path:
        import json
        with open(registry_config_path) as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            parser.error(
                f"--registry-config must contain a JSON dict; got {type(cfg).__name__}"
            )
        _gat_root_for_display = f"<registry_config:{registry_config_path}>"
        _registry = GatScorerRegistry(
            registry_config={str(k): str(v) for k, v in cfg.items()},
            checkpoint_filename=args.checkpoint_filename,
            device=args.device,
            lazy=not args.eager,
        )
    else:
        _gat_root_for_display = gat_root
        _registry = GatScorerRegistry(
            gat_root=Path(gat_root),
            checkpoint_filename=args.checkpoint_filename,
            device=args.device,
            lazy=not args.eager,
        )
    logger.info(
        "Loaded registry with %d disease(s): %s",
        len(_registry.list_diseases()),
        ", ".join(_registry.list_diseases()),
    )

    mcp = FastMCP(args.server_name)
    _register_tools(mcp)

    logger.info("Starting MCP server '%s' on stdio…", args.server_name)
    mcp.run()  # blocks forever; default transport is stdio.


if __name__ == "__main__":
    main()
