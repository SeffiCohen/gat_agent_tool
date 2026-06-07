# gat-agent-tool

Wrap a trained **Graph Attention Network (GAT)** as an *external scoring tool*
that an LLM agent can query iteratively — propose expressions, get predicted
AUCs back, refine, repeat.

One Python package, three ways to plug it in:

| mode         | who's proposing | how the LLM talks to the GAT               | good for                                      |
|--------------|-----------------|---------------------------------------------|-----------------------------------------------|
| `loop-hf`    | local HF model  | scored history injected into next prompt    | offline / no-API, any HF chat model           |
| `mcp`        | Claude Code / Desktop / ChatGPT connectors | native MCP tool calls   | interactive use, tool-use-native agents       |
| `loop-openai`| OpenAI API      | native function-calling tool calls          | scripted OpenAI runs, agent with budget caps  |

The GAT is loaded once and kept resident on GPU (or CPU). All three modes share
the same `GatScorerTool` core and the same validation rules.

---

## Install

```bash
# From inside the BD_Paper repo root:
pip install -e gat_agent_tool[all]

# Or, install only what you need:
pip install -e gat_agent_tool[mcp]      # just the MCP server
pip install -e gat_agent_tool[hf]       # local HuggingFace driver
pip install -e gat_agent_tool[openai]   # OpenAI function-calling driver
```

The package depends on `torch` and `torch-geometric` (the GAT backend), and
relies on `gat_model.py` + `expr_graph_utils.py` from the main repo's `Code/`
directory. These are added to `sys.path` automatically when the package is
imported, as long as the sub-project sits at `<repo>/gat_agent_tool/`. If you
extract the tool to another location, set
`GAT_AGENT_TOOL_CODE_DIR=/path/to/Code` before importing.

---

## Architecture

```
                ┌──────────────────────────────┐
                │   gat_agent_tool.core         │  ← GatScorerTool
                │   (loads checkpoint once;     │
                │    .score / .score_batch)     │
                └──────────────┬───────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
 ┌──────┴───────┐       ┌──────┴───────┐      ┌───────┴────────┐
 │ drivers/     │       │ mcp_server   │      │ drivers/       │
 │ local_hf.py  │       │ (FastMCP,    │      │ openai_api.py  │
 │              │       │  stdio)      │      │                │
 │ HF model     │       │              │      │ OpenAI chat    │
 │ proposes;    │       │ tools:       │      │ + function-    │
 │ tool returns │       │  score_*     │      │ calling; the   │
 │ scores in    │       │  get_feat... │      │ model calls    │
 │ next prompt  │       │              │      │ the GAT itself │
 └──────────────┘       └──────────────┘      └────────────────┘
```

Shared across all three: `gat_agent_tool.validation` — parses LLM output,
rejects numeric literals, deduplicates, enforces feature-name and operator
whitelists.

---

## Expression rules (passed to every LLM)

These constraints come from the main biomarker-discovery pipeline; every
driver enforces them the same way:

* **Features**: case-sensitive strings like `lab_RDW_mean`. The exact legal
  set is whatever the loaded GAT was trained on — query with
  `GatScorerTool.feature_names` or the `get_feature_list` tool.
* **Operators**: `+`, `-`, `*`, `/` only. No functions, comparisons, or
  conditionals.
* **No numeric constants.** Expressions are pure feature arithmetic — the
  GAT's tokenizer has no embedding for literals. Any expression containing
  `0.5`, `2`, etc. gets dropped before the GAT is called.
* **Depth limit**: keep tree depth ≤ the `max_depth` reported by
  `get_feature_list` (typically 3). Deeper expressions weren't seen during
  GAT training and will score unreliably.
* **Scores**: every GAT prediction is clamped to `[0.5, 1.0]` (direction-
  agnostic AUC); higher is better.

---

## Mode 1 — Local HuggingFace loop (`loop-hf`)

Great when you want a closed-world experiment and you already have the LLM
on disk. The LLM gets GAT scores embedded as text in the next prompt — no
function-calling required, so it works with Gemma / Llama / Mistral chat
models out of the box.

```bash
gat-agent-tool loop-hf \
    --checkpoint /path/to/gat_model.pt \
    --llm_model_name google/medgemma-4b-it \
    --output_csv out/medgemma_ra.csv \
    --iterations 10 \
    --pop_per_iter 10 \
    --topic "biomarker expressions for rheumatoid arthritis"

# Optional: also compute real (ground-truth) AUC silently per proposal:
    --data_path /path/to/train.parquet \
    --target_column 714
```

Output CSV columns (one row per proposal, sorted by `gat_score` desc):
`iter, rank_in_iter, expression, gat_score, real_auc, is_valid, duplicate_of`.

The `real_auc` column is never shown to the LLM — it's logged silently so
downstream analyses can test "did GAT-guided search find truly high-AUC
expressions?" without the LLM cheating by reading the labels.

---

## Mode 2 — MCP server (Claude Code / Claude Desktop / ChatGPT)

Launch the server once; any MCP client (Claude Code, Claude Desktop, ChatGPT's
MCP connectors, Cursor, Claude Dev, …) can call the GAT as a tool from
within a normal chat session.

```bash
# Run directly:
gat-agent-mcp --checkpoint /path/to/gat_model.pt

# Or via the unified CLI:
gat-agent-tool mcp --checkpoint /path/to/gat_model.pt
```

### Claude Code config

Drop this under `"mcpServers"` in `~/.claude/settings.json` or a project's
`.mcp.json`:

```jsonc
{
  "mcpServers": {
    "gat-scorer": {
      "command": "gat-agent-mcp",
      "args": ["--checkpoint", "/absolute/path/to/gat_model.pt"]
    }
  }
}
```

Then, in Claude Code, mention the server or ask it to use the tools:

> Design a biomarker expression for rheumatoid arthritis. Use the
> gat-scorer tools: start with `get_feature_list`, then iterate with
> `score_expressions` until the top score exceeds 0.75.

Claude Code will call into the MCP server autonomously and surface the
tool-call traces in the UI.

### Claude Desktop config

On macOS, edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "gat-scorer": {
      "command": "gat-agent-mcp",
      "args": ["--checkpoint", "/absolute/path/to/gat_model.pt"]
    }
  }
}
```

Restart the Desktop app. The server will appear under "Connected tools".

### ChatGPT

Modern ChatGPT supports MCP via the "Custom connectors" beta (ChatGPT Plus
and above). Add the same `gat-agent-mcp` command as a local connector and
chat as usual.

### Running multiple GATs side-by-side

Each MCP server gets a distinct name. To expose several disease-specific GATs
at once:

```jsonc
{
  "mcpServers": {
    "gat-ra":  { "command": "gat-agent-mcp", "args": ["--checkpoint", "/…/ra.pt",  "--server_name", "gat-ra"]  },
    "gat-t1d": { "command": "gat-agent-mcp", "args": ["--checkpoint", "/…/t1d.pt", "--server_name", "gat-t1d"] }
  }
}
```

Tool calls are namespaced by server so there's no collision — the LLM will
call `gat-ra.score_expression(...)` vs. `gat-t1d.score_expression(...)`.

### Tools exposed

| tool name              | purpose                                                      |
|------------------------|--------------------------------------------------------------|
| `get_feature_list()`   | Legal features, operators, depth limit, rules. **Call first.**|
| `score_expression(e)`  | Score a single expression → `{expression, score, error}`.     |
| `score_expressions(l)` | Batch-score a list — faster than calling `score_expression` N times. |
| `get_model_info()`     | Checkpoint metadata (debugging).                              |

---

## Mode 3 — OpenAI function-calling loop (`loop-openai`)

Agentic: the OpenAI model itself decides when to call the GAT, what to
propose next, when to stop. Bounded by `--max_rounds` and `--max_tool_calls`
(safety caps on API cost).

```bash
export OPENAI_API_KEY=sk-…

gat-agent-tool loop-openai \
    --checkpoint /path/to/gat_model.pt \
    --model gpt-4o-mini \
    --output_dir out/openai_run_1 \
    --max_rounds 30 \
    --max_tool_calls 150 \
    --topic "biomarker expressions for Type 1 diabetes"
```

The session writes to `--output_dir`:

* `messages.jsonl` — every chat message (system / user / assistant / tool).
* `tool_calls.csv` — flat table: round, tool_name, arguments, result, latency.
* `summary.json` — best expression found, total rounds / tool calls, final
  assistant message.

For a hand-rolled version you can copy and tweak, see
[`examples/openai_function_calling.py`](examples/openai_function_calling.py)
— it's the same logic without the CLI scaffolding.

---

## Python-library usage

```python
from gat_agent_tool import GatScorerTool

scorer = GatScorerTool("/path/to/gat_model.pt")

print(scorer.feature_names[:5])
# ['lab_RDW_mean', 'lab_RBC_max', ...]

score = scorer.score("(lab_RDW_mean + lab_RBC_max) / lab_WBC_mean")
print(score)  # e.g. 0.6731

batch = scorer.score_batch([
    "lab_NEUT_mean - lab_LYMPH_mean",
    "garbage * 2",                      # numeric literal → None
    "lab_PLT_max / lab_MCH_mean",
])
print(batch)  # [0.71, None, 0.65]
```

---

## Tips for the system prompt

Whether you're writing your own HF loop, MCP client, or OpenAI driver, these
system-prompt hints tend to make GAT-guided search work well:

1. **Call `get_feature_list` first.** Every expression must use features
   exactly as listed; spelling is case-sensitive and the LLM cannot guess.
2. **Batch.** `score_expressions([e1, e2, ...])` is much faster than N
   calls to `score_expression` — and on the OpenAI path, cheaper too.
3. **Remember scores.** The agent must track which expressions it's already
   scored. Duplicates are wasted budget.
4. **No constants.** If the LLM sneaks a `0.5` in, the GAT returns `null` —
   the tool doesn't crash, but the call is wasted. The driver's own
   validation layer filters these too.
5. **Target depth ≤ 3.** 2–4 features combined with 2–3 operators is the
   sweet spot for most trained GATs.

---

## Relationship to the main BD_Paper pipeline

This sub-project is a **reusable wrapper** around a pipeline component. The
main pipeline (under `Code/`) still produces GAT checkpoints, trains the
biomarker-discovery methods, and runs the one-shot "LLM emits top-K → GAT
reranks" flow that's reported in the manuscript. This tool is for the next
class of experiments — iterative, agentic uses of the same GAT from outside
the pipeline.

Shared files (imported, not vendored):

* `Code/gat_model.py` — the GAT model classes.
* `Code/expr_graph_utils.py` — expression ↔ graph conversion.

Everything else is self-contained inside `gat_agent_tool/`.

---

## Development

Pure-Python tests cover the validation helpers:

```bash
python gat_agent_tool/tests/test_validation.py
```

These run without any torch / transformers installation. The heavy modules
(core, drivers) need the real environment to exercise.

Syntax-only sanity:

```bash
python -m py_compile gat_agent_tool/gat_agent_tool/*.py gat_agent_tool/gat_agent_tool/drivers/*.py
```

---

## License

MIT.
