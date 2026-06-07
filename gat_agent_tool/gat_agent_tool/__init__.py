"""gat_agent_tool — wrap a trained GAT as an expression-scoring tool for LLM agents.

Three consumer modes are provided:

  1. Library / local-HF loop: `from gat_agent_tool import GatScorerTool` or run the
     built-in iterative loop via `gat-agent-tool loop-hf ...`.
  2. MCP server (Claude Code, Claude Desktop, modern ChatGPT connectors):
     `gat-agent-mcp --checkpoint <path>` or `gat-agent-tool mcp --checkpoint <path>`.
  3. OpenAI function-calling driver:
     `gat-agent-tool loop-openai --checkpoint <path> --model gpt-4o ...`.

The core `GatScorerTool` depends on `gat_model` and `expr_graph_utils` from the
main BD_Paper repo's `Code/` directory. We prepend that directory to sys.path at
package-import time so the tool can be used without extra PYTHONPATH setup — as
long as this package lives at `<repo>/gat_agent_tool/gat_agent_tool/` and the
main pipeline's sources live at `<repo>/Code/`.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path fix: make the main repo's Code/ directory importable.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# This file is at <repo>/gat_agent_tool/gat_agent_tool/__init__.py .
# Go up twice to reach <repo>/, then into Code/.
_CODE_DIR = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "Code"))
if os.path.isdir(_CODE_DIR) and _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# Users who have vendored/symlinked the core files elsewhere can override:
#   GAT_AGENT_TOOL_CODE_DIR=/custom/path  →  prepended instead.
_override = os.environ.get("GAT_AGENT_TOOL_CODE_DIR")
if _override and os.path.isdir(_override) and _override not in sys.path:
    sys.path.insert(0, _override)


__all__ = ["GatScorerTool", "__version__"]
__version__ = "0.1.0"


# Lazy re-export: importing `gat_agent_tool` should NOT eagerly pull in torch /
# torch_geometric, so that pure-Python consumers (e.g. the validation-helper
# tests, the MCP server's --help path) work in minimal environments. Only on
# attribute access (`gat_agent_tool.GatScorerTool` or
# `from gat_agent_tool import GatScorerTool`) do we import the heavy module.
def __getattr__(name: str):
    if name == "GatScorerTool":
        from .core import GatScorerTool
        return GatScorerTool
    if name == "ModelInfo":
        from .core import ModelInfo
        return ModelInfo
    raise AttributeError(f"module 'gat_agent_tool' has no attribute {name!r}")
