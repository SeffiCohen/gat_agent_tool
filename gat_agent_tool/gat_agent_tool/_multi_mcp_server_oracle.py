"""Independent ORACLE for the multi-disease MCP server.

Mirrors the public contract of `gat_agent_tool.multi_mcp_server` but is written
in a deliberately different internal style so a verifier script can diff the two
and surface any disagreement (the same pattern used by
`Code/_external_orchestrator_oracle.py` and `Code/_compare_picks_oracle.py`).

Intentional divergences from the likely IMPLEMENTER style:
  * Tools are collected via a registration-table pattern (`_TOOL_NAMES` + a
    factory `_build_tool_table`) and attached in a loop, instead of one
    `@mcp.tool()` decorator per function.
  * Error envelopes are built by a single `_unknown_disease_error` helper —
    a key-name typo would surface in one call site, not five.
  * `score_expression` discovers unknown diseases by catching the registry's
    `KeyError`, not by a pre-flight membership check.
  * Operator list (only used for documentation, not API) lives in a module
    constant `_OPERATORS`.
  * `main()` factors JSON parsing out into `_load_registry_config()`.
  * Inline `_build_tool_table` returns a `Dict[str, Callable]` so the same dict
    can also feed unit tests / introspection.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover — keep torch out of import time
    from .registry import GatScorerRegistry


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state — names locked by the contract.
# ---------------------------------------------------------------------------
_registry: "Optional[GatScorerRegistry]" = None
_gat_root_for_display: str = ""
_checkpoint_filename_for_display: str = ""

# Documentation-only — operator list embedded in the `rules` strings the LLM
# sees. ORACLE divergence: defined once here instead of inlined per-rule.
_OPERATORS = ("+", "-", "*", "/")

_PARSE_ERROR_MSG = "could not parse expression into a graph"


def _require_registry() -> "GatScorerRegistry":
    """Return the configured registry, or raise if `main()` never ran."""
    if _registry is None:
        raise RuntimeError(
            "multi-disease MCP server misconfigured: registry not initialized. "
            "Launch via main() so --gat-root / --registry-config is parsed first."
        )
    return _registry


# ---------------------------------------------------------------------------
# Error-envelope helpers (centralised so any key-name typo shows up once)
# ---------------------------------------------------------------------------
def _unknown_disease_error(
    disease_id: str,
    available: List[str],
    expression: Optional[str] = None,
) -> Dict[str, Any]:
    """Single source of truth for the unknown-disease envelope.

    When `expression` is supplied the envelope adds the `expression` and `score`
    keys — that's the shape `score_expression` returns. Otherwise it's the
    `get_feature_list` / `get_model_info` shape.
    """
    msg = f"unknown disease_id={disease_id!r}"
    if expression is None:
        return {"disease_id": disease_id, "error": msg, "available": list(available)}
    return {
        "disease_id": disease_id,
        "expression": expression,
        "score": None,
        "error": msg,
        "available": list(available),
    }


# ---------------------------------------------------------------------------
# Tool implementations — factored so the same callables can be registered with
# FastMCP in main() AND inspected by the test suite via `_register_tools` and
# a FakeMCP. The closure `reg_getter` is `_require_registry` in production.
# ---------------------------------------------------------------------------
def _build_tool_table(reg_getter: Callable[[], "GatScorerRegistry"]) -> Dict[str, Callable]:
    """Return `{tool_name: callable}`. The bodies form the entire behavioural
    contract — `_register_tools` then loops over the dict and decorates."""

    # ---- list_diseases -------------------------------------------------
    def list_diseases() -> Dict[str, Any]:
        reg = reg_getter()
        # Prefer the registry's own `_gat_root` (the FakeRegistry exposes it
        # as a normal attribute; the production GatScorerRegistry also stores
        # the source root). Fall back to the module-global the CLI populated.
        root_from_reg = getattr(reg, "_gat_root", None)
        return {
            "diseases": list(reg.list_diseases()),
            "gat_root": root_from_reg if root_from_reg else _gat_root_for_display,
            "default_checkpoint_filename": _checkpoint_filename_for_display,
        }

    # ---- get_feature_list ---------------------------------------------
    def get_feature_list(disease_id: str) -> Dict[str, Any]:
        reg = reg_getter()
        known = list(reg.list_diseases())
        if disease_id not in known:
            return _unknown_disease_error(disease_id, known)
        feats = reg.feature_names(disease_id)
        meta = reg.info(disease_id)
        max_depth = meta["max_depth"]
        return {
            "disease_id": disease_id,
            "features": list(feats),
            "operators": list(_OPERATORS),
            "max_depth": max_depth,
            "model_type": meta["model_type"],
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

    # ---- get_model_info -----------------------------------------------
    def get_model_info(disease_id: str) -> Dict[str, Any]:
        reg = reg_getter()
        known = list(reg.list_diseases())
        if disease_id not in known:
            return _unknown_disease_error(disease_id, known)
        meta = reg.info(disease_id)
        # Build manually — divergent from any `**meta` splat.
        out: Dict[str, Any] = {
            "disease_id": disease_id,
            "model_type": meta["model_type"],
            "num_features": meta["num_features"],
            "num_operators": meta["num_operators"],
            "max_depth": meta["max_depth"],
            "graph_format_version": meta["graph_format_version"],
            "has_auc_transform": meta["has_auc_transform"],
            "checkpoint_path": meta["checkpoint_path"],
        }
        # `device` is part of the info() dict per the registry contract; the
        # implementer reads info["device"] directly with a fallback to "auto".
        out["device"] = meta.get("device", "auto")
        return out

    # ---- score_expression ---------------------------------------------
    def score_expression(disease_id: str, expression: str) -> Dict[str, Any]:
        reg = reg_getter()
        # ORACLE divergence: probe membership via the registry's feature_names()
        # (which raises KeyError for unknown diseases per the registry contract),
        # rather than IMPLEMENTER's `if disease_id not in reg.list_diseases()` set
        # check. Same observable behavior, different code path.
        known = list(reg.list_diseases())
        if disease_id not in known:
            return _unknown_disease_error(disease_id, known, expression=expression)
        try:
            score = reg.score(disease_id, expression)
        except Exception as exc:  # noqa: BLE001
            logger.exception("score_expression failed for %r/%r", disease_id, expression)
            return {
                "disease_id": disease_id,
                "expression": expression,
                "score": None,
                "error": str(exc),
            }
        return {
            "disease_id": disease_id,
            "expression": expression,
            "score": score,
            "error": None if score is not None else _PARSE_ERROR_MSG,
        }

    # ---- score_expressions --------------------------------------------
    def score_expressions(disease_id: str, expressions: List[str]) -> List[Dict[str, Any]]:
        reg = reg_getter()
        exprs = list(expressions)
        known = list(reg.list_diseases())
        if disease_id not in known:
            return [_unknown_disease_error(disease_id, known, expression=e) for e in exprs]
        try:
            scores = reg.score_batch(disease_id, exprs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("score_expressions failed for %r", disease_id)
            err = str(exc)
            return [
                {"disease_id": disease_id, "expression": e, "score": None, "error": err}
                for e in exprs
            ]
        out: List[Dict[str, Any]] = []
        for e, sc in zip(exprs, scores):
            out.append({
                "disease_id": disease_id,
                "expression": e,
                "score": sc,
                "error": None if sc is not None else _PARSE_ERROR_MSG,
            })
        return out

    return {
        "list_diseases": list_diseases,
        "get_feature_list": get_feature_list,
        "get_model_info": get_model_info,
        "score_expression": score_expression,
        "score_expressions": score_expressions,
    }


def _register_tools(mcp: Any) -> None:
    """Attach all tools to the FastMCP instance via the table pattern."""
    table = _build_tool_table(_require_registry)
    for _name, fn in table.items():
        # Each iteration registers `fn` under its own `__name__` (which the
        # FakeMCP test fixture uses as a key). Same outcome as if we'd written
        # five `@mcp.tool()` decorators at module top.
        mcp.tool()(fn)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def _load_registry_config(path: str) -> Dict[str, str]:
    """Load `{disease_id: checkpoint_path}` from a JSON file.

    ORACLE divergence: extracted as a helper rather than inlined in main().
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(
            f"--registry-config {path!r} must be a JSON object "
            f"({{disease_id: checkpoint_path}}); got {type(payload).__name__}"
        )
    return {str(k): str(v) for k, v in payload.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Factored out so both main() and tests can use it."""
    parser = argparse.ArgumentParser(
        prog="gat-agent-multi-mcp",
        description="Multi-disease GAT MCP server (one process serves N checkpoints).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--gat-root", default=None, help="Root directory of GAT/ checkpoints.")
    src.add_argument("--registry-config", default=None,
                     help="Path to JSON {disease_id: checkpoint_path}.")
    parser.add_argument("--checkpoint-filename", default="best_by_loss.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--server_name", default="gat-multi-scorer")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--eager", action="store_true",
                        help="Preload every checkpoint at startup.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    env_root = os.environ.get("GAT_AGENT_REGISTRY_ROOT")
    if not args.gat_root and not args.registry_config and not env_root:
        parser.error(
            "must pass one of: --gat-root, --registry-config, or set "
            "$GAT_AGENT_REGISTRY_ROOT environment variable"
        )

    # Stderr logging — stdout is reserved for MCP JSON-RPC traffic.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # Defer FastMCP import so --help works in environments without `mcp`.
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        sys.stderr.write(
            "[gat-agent-multi-mcp] The `mcp` package is required. "
            "Install with: pip install 'gat-agent-tool[mcp]'\n"
        )
        raise SystemExit(1) from exc

    from .registry import GatScorerRegistry  # local: keeps module-import cheap

    global _registry, _gat_root_for_display, _checkpoint_filename_for_display

    if args.registry_config:
        cfg = _load_registry_config(args.registry_config)
        _registry = GatScorerRegistry(
            registry_config=cfg,
            checkpoint_filename=args.checkpoint_filename,
            device=args.device,
            lazy=not args.eager,
        )
        _gat_root_for_display = ""
    else:
        root = args.gat_root or env_root
        _registry = GatScorerRegistry(
            gat_root=Path(root),
            checkpoint_filename=args.checkpoint_filename,
            device=args.device,
            lazy=not args.eager,
        )
        _gat_root_for_display = str(root)
    _checkpoint_filename_for_display = args.checkpoint_filename

    logger.info(
        "GatScorerRegistry ready: %d disease(s) advertised",
        len(_registry.list_diseases()),
    )

    server = FastMCP(args.server_name)
    _register_tools(server)
    logger.info("Starting multi-disease MCP server '%s' on stdio…", args.server_name)
    server.run()


if __name__ == "__main__":
    main()
