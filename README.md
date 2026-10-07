📄 Paper / Preprint: [Distilling private EHR evidence into a public agentic tool for CBC biomarker discovery](https://arxiv.org/abs/2610.04749)
# gat-agent-tool — a privacy-preserving Graph-Attention scorer for CBC biomarker discovery

`gat-agent-tool` releases a trained **Graph Attention Network (GAT)** that ranks candidate
complete-blood-count (CBC) biomarkers — short arithmetic expressions over routine lab
features — by their **predicted area-under-the-ROC-curve (AUC)**, inferred from the
*structure of the expression alone*. A language model or research agent proposes
candidate expressions; the GAT scores each one with a single forward pass over its
symbolic graph encoding and returns a predicted AUC in `[0.5, 1.0]`. Crucially, **once
the GAT is trained it never touches patient-level data at scoring time**: ranking is a
pure symbolic-to-scalar function. This decouples off-premises candidate *generation* from
on-premises cohort *evaluation*, so a useful discovery prior learned on a private
electronic-health-record (EHR) cohort can be distilled into a small, shareable artifact.
The weights in this repository were trained on the Clalit Health Services research
database; **only the trained weights and the scoring code are released — no patient data,
in any form, is included.** This repository is the companion software/model artifact to
the paper *"Distilling private EHR evidence into a public agentic tool for CBC biomarker
discovery."*

---

## What it does, in one breath

> An agent proposes `lab_NEUT_abs / lab_LYMPH_abs`; the GAT replies `0.71`. The agent
> refines, re-scores, and converges on high-AUC candidates — without the cohort ever
> leaving the institution that holds it.

The scorer accepts feature-only arithmetic expressions built from a fixed lab-feature
vocabulary and the four operators `+ - * /` (no numeric constants, no functions). The
legal feature set is whatever the loaded checkpoint was trained on and can be queried at
runtime. Higher predicted AUC is better; scores are direction-agnostic and clamped to
`[0.5, 1.0]`.

---

## Repository layout

```
gat-agent-tool/
├── Code/                          GAT model + graph-encoding code the package imports
│   ├── gat_model.py               EfficientGAT / AdvancedGAT / TreeAwareGAT class definitions
│   ├── expr_graph_utils.py        expression-string → PyG graph (string_to_data_obj, parser)
│   └── _external_scoring.py       expression evaluation + direction-agnostic AUC helpers
├── skills/                        optional Claude Code agent skills (see skills/README.md)
│   ├── biomarker-discovery/       literature-grounded propose–score–refine search loop
│   └── biomarker-explainer/       Shapley attribution + literature-concordance reporting
├── gat_agent_tool/                installable Python package (core scorer + registry + drivers)
│   ├── pyproject.toml             package metadata and console scripts
│   ├── gat_agent_tool/            package source (core.py, registry.py, MCP servers, drivers)
│   ├── examples/                  hand-rolled driver examples
│   └── tests/                     torch-free validation tests
├── GAT/<disease_id>/trained_models/best_by_loss.pt    13 released disease checkpoints
├── docs/
│   └── ICD_codes.csv              disease-name ↔ ICD-code reference table
├── LICENSE                        Apache License 2.0 (source code)
└── .mcp.json                      ready-to-use MCP wiring for Claude Code / Claude Desktop
```

The scorer package deliberately does **not** vendor the model architecture: it imports
`gat_model.py` and `expr_graph_utils.py` from the sibling `Code/` directory (both are
included here) so that the released weights are interpreted by exactly the same code that
produced them.

---

## Installation

Python 3.10–3.12.

```bash
# From the repository root:
pip install -e gat_agent_tool[all]      # all drivers: MCP + HuggingFace + OpenAI

# Or install only the parts you need:
pip install -e gat_agent_tool[mcp]      # MCP server only
pip install -e gat_agent_tool[hf]       # local HuggingFace propose-score loop
pip install -e gat_agent_tool[openai]   # OpenAI function-calling loop
```

The package imports `gat_model.py` and `expr_graph_utils.py` from the sibling `Code/`
directory. When the package sits next to `Code/` (as in this repository) that directory
is added to the import path automatically at package-import time. If you relocate the
package, point it at the model code explicitly:

```bash
export GAT_AGENT_TOOL_CODE_DIR=/absolute/path/to/Code
```

The scorer requires `torch` and `torch-geometric`; these are installed by every extra
above.

---

## Released disease checkpoints

Thirteen disease-specific GATs are released, one per autoimmune-forward phenotype, under
`GAT/<disease_id>/trained_models/best_by_loss.pt`. The `disease_id` is the ICD-9 stem used
on disk; the table below maps each to its clinical phenotype (see `docs/ICD_codes.csv` for
the full code lists).

| `disease_id` | Disease | ICD-9 reference |
|---|---|---|
| `242`  | Graves' disease | 242.0x |
| `250`  | Type I diabetes (incl. complications) | 250.x1 / 250.x3 |
| `277`  | Familial Mediterranean fever | 277.31 |
| `340`  | Multiple sclerosis | 340 |
| `555`  | Crohn's disease | 555.x |
| `556`  | Ulcerative colitis | 556.x |
| `696`  | Psoriasis | 696.0 / 696.1 |
| `2452` | Hashimoto thyroiditis | 245.2 |
| `5790` | Celiac disease | 579.0 |
| `7100` | Lupus erythematosus (DLE / SLE) | 710.0 (with 695.4) |
| `7101` | Scleroderma | 710.1 |
| `7102` | Sjögren's / Sicca syndrome | 710.2 |
| `714`  | Rheumatoid arthritis & inflammatory polyarthropathies | 714 / 714.x |

> **Note — Hashimoto folder name.** The checkpoint for `disease_id` **`2452`** physically
> lives in the on-disk folder **`GAT/2542/`** because of a historical digit-swap typo when
> that folder was first created. The registry carries a built-in override
> `{"2452": "2542"}`, so you should always request it by its correct ICD code `"2452"` —
> the registry resolves the path transparently. Pass `disk_folder_override={}` to disable
> the mapping, or supply your own `{advertised_id: on_disk_folder}` dictionary for other
> renames.

---

## Quickstart — Python

The multi-disease entry point is `GatScorerRegistry`: point it at the `GAT/` directory and
it auto-discovers every released checkpoint, loading each one lazily on first request and
caching it thereafter.

```python
from gat_agent_tool.registry import GatScorerRegistry

# Auto-discover all 13 disease checkpoints under GAT/<id>/trained_models/best_by_loss.pt
registry = GatScorerRegistry(gat_root="GAT")

# Which diseases are available?
print(registry.list_diseases())
# ['242', '250', '277', '340', '555', '556', '696', '2452',
#  '5790', '7100', '7101', '7102', '714']

# Inspect the legal feature vocabulary for one disease (case-sensitive).
print(registry.feature_names("714")[:5])
# ['lab_RDW_mean', 'lab_RBC_max', 'lab_WBC_mean', ...]

# Score a single candidate expression for rheumatoid arthritis (disease_id 714).
score = registry.score("714", "lab_NEUT_abs / lab_LYMPH_abs")
print(score)              # e.g. 0.71  (predicted AUC, in [0.5, 1.0])

# Batch-score several candidates at once (faster than scoring one by one).
batch = registry.score_batch("714", [
    "lab_NEUT_abs / lab_LYMPH_abs",
    "(lab_RDW_mean + lab_RBC_max) / lab_WBC_mean",
    "lab_PLT_max * 0.5",          # numeric literal → rejected → None
])
print(batch)              # [0.71, 0.66, None]

# Checkpoint metadata (model variant, feature/operator counts, depth limit).
print(registry.info("714"))
```

Returned values are the predicted AUC in `[0.5, 1.0]`, or `None` when an expression cannot
be parsed into a graph (unknown feature, unbalanced parentheses, or a numeric literal —
the vocabulary has no embedding for constants). Requesting an unknown `disease_id` raises
`KeyError` with the list of available IDs.

If you only ever need a single checkpoint and never switch diseases, you can skip the
registry and use the underlying scorer directly:

```python
from gat_agent_tool import GatScorerTool

scorer = GatScorerTool("GAT/714/trained_models/best_by_loss.pt")
scorer.score("lab_NEUT_abs / lab_LYMPH_abs")     # -> 0.71
scorer.score_batch(["lab_A + lab_B", "bad * 2"]) # -> [0.61, None]
scorer.feature_names                              # -> ['lab_RDW_mean', ...]
```

---

## Quickstart — MCP (Claude Code / Claude Desktop)

The repository ships a ready-to-use `.mcp.json` that exposes all 13 disease GATs through a
single Model Context Protocol (MCP) server named **`gat-multi-scorer`**, backed by the
`gat-agent-multi-mcp` console script:

```jsonc
{
  "mcpServers": {
    "gat-multi-scorer": {
      "command": "gat-agent-multi-mcp",
      "args": [
        "--gat-root", "GAT",
        "--checkpoint-filename", "best_by_loss.pt",
        "--server_name", "gat-multi-scorer",
        "--device", "auto",
        "--log_level", "INFO"
      ],
      "env": { "GAT_AGENT_TOOL_CODE_DIR": "Code" }
    }
  }
}
```

Launched from the repository root, the server auto-discovers every checkpoint under
`GAT/<id>/trained_models/best_by_loss.pt` and presents these tools to any MCP-native client
(Claude Code, Claude Desktop, or other MCP connectors):

| Tool | Purpose |
|---|---|
| `list_diseases()` | List the available `disease_id`s. |
| `get_feature_list(disease_id)` | Legal features, operators, and depth limit for a disease. **Call first.** |
| `score_expression(disease_id, expression)` | Score one candidate → predicted AUC. |
| `score_expressions(disease_id, expressions)` | Batch-score a list of candidates. |
| `get_model_info(disease_id)` | Checkpoint metadata (debugging). |

Every tool except `list_diseases` takes a `disease_id` as its first argument, so a single
server serves all 13 phenotypes. From a chat session you can simply ask the agent to
discover a biomarker and let it drive the tools, e.g. *"Use the gat-multi-scorer tools:
call get_feature_list for disease 714, then iterate with score_expressions until the top
predicted AUC exceeds 0.75."*

To register the server globally instead of per-project, copy the same `gat-multi-scorer`
block into the `"mcpServers"` map of `~/.claude/settings.json` (Claude Code) or
`~/Library/Application Support/Claude/claude_desktop_config.json` (Claude Desktop on
macOS), using an absolute `--gat-root` path.

---

## Agent skills

The scorer answers one question — *how promising is this expression?* Two optional
[Claude Code](https://claude.com/claude-code) skills in [`skills/`](skills/) wrap it into the
research loops described in the paper:

- **`biomarker-discovery`** — a literature-grounded **propose → score → refine** loop. Claude
  proposes batches of CBC expressions, the released scorer ranks them by predicted AUC, and the
  loop refines until the predicted score plateaus. The measured AUC is logged but never shown to
  the proposing model, so the search tests the distilled prior alone.
- **`biomarker-explainer`** — explains one expression: exact Shapley attribution of every
  blood-count feature, a glass-box figure, and a literature review classifying each component
  *Expected* / *Surprising* with verified citations, packaged as a one-page report.

```bash
pip install -e gat_agent_tool[all]      # scorer + MCP server
pip install -r requirements.txt         # adds scikit-learn, asteval, matplotlib

mkdir -p .claude/skills                 # Claude Code discovers skills here
ln -s "$PWD/skills/biomarker-discovery" .claude/skills/
ln -s "$PWD/skills/biomarker-explainer" .claude/skills/
```

Launch Claude Code from the repository root (so `.mcp.json`'s relative paths resolve), then invoke
`/biomarker-discovery` or `/biomarker-explainer`.

**What is not included.** No patient data ships here. Scoring an expression needs nothing beyond
this clone, but every step that measures a *real* AUC — bootstrap CIs and all Shapley attribution —
requires a labelled cohort **you** supply (MIMIC-IV, EHRShot and NHANES are obtained under their own
data use agreements). The paper's head-to-head against third-party LLM-tool expressions is not
reproducible here, as those candidate sets are not redistributable. See
[`skills/README.md`](skills/README.md) for the full capability matrix, the required parquet schema,
and the limitations.

---

## Privacy statement

The released artifacts — the trained GAT checkpoints and the scoring code — contain **no
patient-level data of any kind**. The GAT is a pure symbolic-to-scalar function: its only
input at inference time is the graph encoding of a candidate arithmetic expression, and
scoring is a single forward pass over that graph. Patient records are consulted only
during model *training* (on the private Clalit cohort) and, separately, during any final
classifier fit a user may choose to run on their *own* cohort against the small set of
top-ranked candidates — never during the ranking step performed by this tool. This is the
core privacy property of the method: private EHR evidence is distilled into a shareable
ranking prior, and the prior can be released and reused without re-exposing the data it
was learned from.

---

## Licensing

Licensing in this repository spans three components; the terms below are stated as they
currently stand.

- **Source code** (the `Code/` model definitions and the `gat_agent_tool/` package source)
  is released under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
- **Known mismatch to resolve before public release:** the `gat_agent_tool/` package's
  `pyproject.toml` currently declares its license as **MIT**. This disagrees with the
  repository's Apache-2.0 `LICENSE` and is a known inconsistency to be unified prior to a
  public release.
- **Trained GAT checkpoints** (the `.pt` weights under `GAT/`) are released under the
  **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license —
  see `MODEL_LICENSE.md`.

If you redistribute or build on this work, retain the corresponding notices for each
component.
