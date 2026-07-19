---
name: biomarker-discovery
description: Iteratively discover **CBC-derived (complete blood count)** biomarker arithmetic expressions for a target disease (ICD code) using a multi-disease GAT scoring MCP server. Conducts a literature-grounded propose-score-refine loop where Claude proposes CBC lab-feature expressions in a medical-researcher framing (mirroring the prompt style of `Code/e2e_2M_CBC_OPT_crohn.py`), the GAT predicts AUC, and final candidates are validated on every available external cohort against published baselines. Trigger on any request to discover, generate, search for, or evolve novel CBC biomarker expressions for a disease; to beat a published external-method baseline (GPT/Gemini/SciSpace) on a cohort; or to run the multi-iteration GAT-scored discovery loop. Common phrasings include "discover biomarkers for", "find CBC expressions for ICD", "find new biomarker candidates", "biomarker discovery for <disease>", "beat the GPT baseline for <disease>", or explicit /biomarker-discovery invocation. Do NOT trigger on one-off "score this expression" requests, single-call GAT inference, generic literature reviews unrelated to the GAT feature set, or requests to just list features / explore the model.
version: 0.1.0
---

# Biomarker Discovery (CBC-derived)

End-to-end pipeline: **literature review → propose-score-refine via GAT → external-cohort validation against published baselines**, all driven by Claude in a single Claude Code session.

The skill wraps a trained Graph Attention Network (one per disease, 13 in total) as an external MCP scoring tool, and orchestrates Claude through three phases. Every disease's GAT is trained on **complete-blood-count (CBC) lab features** — `lab_NEUTpct_last`, `lab_LYMpct_last`, `lab_PLT_last`, `lab_HB_last`, `lab_RDW_last`, etc. — so every prompt in Phase 2 is medically framed as "biomarker discovery from CBC features", mirroring the canonical orchestrator [`Code/e2e_2M_CBC_OPT_crohn.py`](../../Code/e2e_2M_CBC_OPT_crohn.py) (the proven prompt template the manuscript pipeline uses). Result: a single best CBC biomarker expression with a measured AUC on every available external cohort, plus a side-by-side comparison vs. published external-method baselines (GPT Deep Research, Gemini, SciSpace).

> **🛑 LOAD-BEARING INVARIANT — silent `real_auc`**
> The `real_auc` column in `iterative_details.csv` exists for the final report only. **Never include `real_auc` values in any prompt to Claude during Phase 2.** Only `gat_score` may appear in the iteration history block. If `real_auc` leaks into a prompt, the propose-score-refine loop becomes contaminated and the entire experiment is invalid. This mirrors the experimental control at `Code/iterative_llm_gat.py:386`.

## When to use

Trigger this skill when the user wants to **discover** new biomarker candidates for a disease — not when they just want to score an existing expression (use the MCP tools directly for that).

Good triggers:
- "Discover biomarkers for rheumatoid arthritis"
- "Find biomarker expressions for ICD 250"
- "/biomarker-discovery 714"
- "Run the full biomarker pipeline on Crohn's"

Bad triggers (use `mcp__gat-multi-scorer__*` tools directly instead):
- "Score this expression"
- "What's the GAT score for X?"
- "Show me the feature list for ICD Y"

## Expected runtime

The default budget is **100 iterations × 200 expressions per iter** (up to 20,000 expressions). Realistic wall-clock breakdown:
- Phase 1 lit-review depth (PubMed/bioRxiv/Open Targets queries) — 10–20 min
- Phase 2 iteration count and Claude's response latency — ~1–3 min per iter (200-expression batches take longer to generate + score than the legacy 10-batch); 100 iters would be 2–5 h, but in practice plateau detection (default patience `5`) typically stops at iter 8–20 when the GAT's surrogate AUC saturates → **30 min – 1.5 h** end-to-end
- Phase 3 bootstrap (n_boot=500) on the cohort parquet, evaluated for the top-by-GAT, first-expression, and iteration-winner picks across all available cohorts — 2–10 min depending on cohort count

Drop `--iterations` to 5–10 for a quick sanity check (~10–20 min). With `pop_per_iter=200` and `plateau_patience=5`, you almost never reach the 100-iter cap.

## MANDATORY: Setup gate

Before doing anything else, verify the environment in this order. Fail fast on any missing prerequisite — DO NOT silently proceed.

### 0. Working directory is the repo root

The `.mcp.json` arguments (`--gat-root GAT`, env `GAT_AGENT_TOOL_CODE_DIR=Code`) are RELATIVE paths. Claude Code must have been launched from the root of this repository (the directory containing `.mcp.json`, `GAT/`, and `Code/`); otherwise the MCP server can't find the GAT checkpoints or the `Code/` helpers. Verify with `pwd` (via Bash). If wrong, ask the user to restart Claude Code from the repo root.

### 1. MCP server `gat-multi-scorer` is live

Call:
```
mcp__gat-multi-scorer__list_diseases()
```

Expected: dict with `diseases: [...]` containing at least 13 ICD codes (the canonical set: `242, 250, 277, 340, 555, 556, 696, 714, 2452, 5790, 7100, 7101, 7102`).

If the call errors with "tool not found" or similar:
- Check `./.mcp.json` exists and contains the `gat-multi-scorer` server entry.
- The expected snippet:
  ```json
  {
    "mcpServers": {
      "gat-multi-scorer": {
        "command": "gat-agent-multi-mcp",
        "args": ["--gat-root", "GAT", "--checkpoint-filename", "best_by_loss.pt", "--server_name", "gat-multi-scorer", "--device", "auto", "--log_level", "INFO"],
        "env": {"GAT_AGENT_TOOL_CODE_DIR": "Code"}
      }
    }
  }
  ```
- After adding/fixing the file, the user must restart Claude Code so the MCP server reconnects. Tell them.
- If the `gat-agent-multi-mcp` console script is missing, install with `pip install -e gat_agent_tool[all]` from the repo root.

### 2. Disease selected

Default: `disease_id="714"` (RheumatoidArthritis) when the user invokes `/biomarker-discovery` with no args.

If the user supplies an arg, validate it's in the result of `list_diseases()`. If not, print available IDs from `references/diseases.md` and stop.

### 3. External cohort parquet present

Default cohort: `mimic`. Resolve the parquet path from `references/diseases.md` (e.g., `Data/MIMIC/preprocessed/714_RheumatoidArthritis_mimic.parquet`).

If the file does not exist on disk, print which preprocessor to run (e.g., `python Code/preprocess_mimic_all.py --data_dir Data/MIMIC --output_dir Data/MIMIC/preprocessed`) and stop.

### 4. NHANES restriction

If `--cohort nhanes` is passed, the disease must be one of `{242, 250, 696, 714, 2452}` (per `Code/preprocess_nhanes_external.py:NHANES_CASE_DEFINITIONS`). Note: `242` and `2452` both use the "any-thyroid problem" label, so results for those two are conflated; surface this caveat in the final report. Other diseases lack NHANES labels — reject with a clear error and the supported list.

### 5. Helper scripts importable

Confirm `Code/_external_scoring.py` and `Code/_ehrshot_disease_map.py` exist. Both `scripts/silent_real_auc.py` and `scripts/finalize.py` import from them via `sys.path` patch — missing files will crash mid-run.

### 6. Output directory

Resolve to `Results/<icd>/biomarker_discovery/<YYYYMMDD-HHMM>/` using `datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")`. Create it. All Phase 1/2/3 artifacts go inside.

## Subcommands

| Invocation | Effect |
|---|---|
| `/biomarker-discovery` | Full pipeline on RA (714) on MIMIC. |
| `/biomarker-discovery <icd>` | Same, but for a different disease (e.g., `/biomarker-discovery 250`). |
| `/biomarker-discovery <icd> --iterations N` | Override iteration count (default 100). |
| `/biomarker-discovery <icd> --pop-per-iter K` | Override expressions-per-iter (default 200). |
| `/biomarker-discovery <icd> --skip-literature` | Skip Phase 1; the loop just runs without the literature_review.md context. The first-expression baseline is unaffected (still iter 1 rank 0). |
| `/biomarker-discovery <icd> --cohort {mimic,ehrshot,nhanes}` | Override cohort (default `mimic`). `nhanes` is supported only for `{242, 250, 696, 714, 2452}`. |
| `/biomarker-discovery <icd> --checkpoint-variant {best_by_loss,best_by_rank,gat_embed_model_v5_1}` | Choose a different GAT checkpoint variant (default `best_by_loss`). |
| `/biomarker-discovery <icd> --plateau-patience N` | Iterations without GAT improvement before early-stop (default 5). |
| `/biomarker-discovery <icd> --n-bootstrap N` | Bootstrap resamples for Phase 3 CI (default 500). |

## Workflow phases

### Phase 1 — Literature review via a parallel Opus 4.8 Workflow (`references/literature_review.md`)

Conduct deep, literature-grounded research for the target disease by launching the **`Workflow` tool** to fan out **3 parallel Opus 4.8 research agents at `max` effort** — one `agent()` call per research angle, each with `model: "opus"` and `effort: "max"`. Each agent works in its own context window:

1. **Mechanism agent** — Open Targets disease→target evidence + per-feature mechanistic narrative.
2. **Established-ratios agent** — PubMed search for published biomarker ratios (NLR, PLR, MLR, RDW, MCV, etc.) since 2015.
3. **Preprints / novel-composites agent** — bioRxiv + recent PubMed (2022+) for novel composites (AISI, SII, PIV, ML scores).

The Workflow returns the 3 markdown reports to the main thread, which synthesizes them into `output_dir/literature_review.md` (with 3 sidecar `_agent_*.md` files for audit). Output: 5–10 seed expressions in the GAT's exact DSL form (validated via `parse_candidates`), per-feature mechanistic rationale, and PMID/preprint citations.

> **A skill invoking `Workflow` is an explicit opt-in** — the Workflow tool is permitted here because these skill instructions direct you to call it (no separate "ultracode" toggle needed). If the `Workflow` tool is unavailable in the running harness, fall back to 3 parallel `Agent` tool calls (`subagent_type: "general-purpose"`, `model: "opus"`) dispatched in a single message; the `Agent` tool can't pin `effort: "max"`, so it inherits the session effort. The agent prompts in `references/literature_review.md` are identical either way.

**Why Opus 4.8 at max effort, not Sonnet**: literature synthesis benefits from the strongest model holding dozens of abstracts in working memory and prioritizing citation quality over quantity; `max` effort buys deeper cross-checking of PMIDs and mechanistic plausibility. `model: "opus"` always resolves to the latest Opus version (**Opus 4.8**) regardless of the user's session model.

**Why a Workflow**: 3 research angles in one parallel wall-clock window (20–40 min total) instead of serial (60–120 min), with deterministic fan-out, per-agent progress in `/workflows`, and a single structured return the main thread synthesizes.

Read [`references/literature_review.md`](references/literature_review.md) for the full procedural detail (the Workflow script, agent prompt templates, synthesis logic, quality checklist).

Skip this phase only when `--skip-literature` is set; Phase 2's iter-1 baseline quality drops without context but the loop still runs.

### Phase 2 — Iteration loop (`references/iteration_loop.md`)

**Hand-off contract**: Phase 2 reads `<output_dir>/literature_review.md` as **context only** — Claude has its content in working memory when generating iter 1's batch, but the literature seeds are NOT scored at iter 0. Under the new flow, **iter 0 does not exist**; the loop starts at `iter == 1` with a 200-expression batch from `assets/initial_prompt.md`. If `literature_review.md` is missing AND `--skip-literature` was not passed, Phase 2 fails fast with a stderr message. With `--skip-literature` and no file, the loop simply runs without literature context (the prompt asset is unchanged).

Drive the propose-score-refine loop:
- **Iter 1**: 200-expression initial batch from `assets/initial_prompt.md` (verbatim user prompt with the 14-feature CBC panel and 4-rule constraint set). The first valid expression (`rank_in_iter == 0`) becomes Phase 3's **first-expression baseline**.
- **Iter 2..N**: 200-expression refinement batches from `assets/iteration_prompt.md`, with a `{history_block}` of prior GAT scores injected. The history block contains ONLY `gat_score` and the expression — never `real_auc` (load-bearing experimental control).
- Per iteration: validate via `gat_agent_tool.validation.parse_candidates`, dedupe, score in batch via `mcp__gat-multi-scorer__score_expressions`, append to `iterative_details.csv`, fill in real univariate AUC silently via `scripts/silent_real_auc.py`, check plateau (default `plateau_patience=5`), repeat.

Read [`references/iteration_loop.md`](references/iteration_loop.md) for the full procedural detail.

### Phase 3 — Final delivery (`scripts/finalize.py`)

**Hand-off contract**: Phase 3 reads `<output_dir>/iterative_details.csv` (Phase 2 output). The CSV must have at least one `is_valid=True` row, otherwise Phase 3 exits 5.

Sort `iterative_details.csv` by `gat_score` desc, take row 0 as `top_by_gat`. Bootstrap CI on its expression against the **primary cohort** parquet (the one passed via `--parquet`/`--cohort`) via `_external_scoring.bootstrap_auc_ci`. Read each baseline CSV from `Results/<icd>/evaluation_results_external_v2/<cohort>/<Tool>_eval_details.csv`, take the max `univariate_auc` per tool, take the overall max as `baseline_best`. PASS if `top_by_gat`'s primary-cohort point AUC > `baseline_best` (strict). If no baseline files are found at all, auto-PASS with a `(no baselines found)` note in the stdout line — surface this transparently in the final report so the user knows the comparison was vacuous.

**First expression as baseline**. Phase 3 treats the row at **`rank_in_iter == 0` in the lowest iter present** (typically `iter == 1` under the new flow) — the LLM's #1 self-ranked pick from the initial 200-batch — as the canonical **first-expression baseline**. The iteration winner is the highest-`gat_score` row among iters strictly *after* the first batch. The per-iteration `Δ real AUC = iteration_winner − first_expression` is the primary scoring signal of the loop, isolating the value the GAT-feedback iterations add over the LLM's first 200-batch first guess. This Δ is emitted as a `BASELINE_DELTA` stdout line by `finalize.py`. (The published-baselines `PASS/FAIL` gate stays as a secondary check for back-compat with existing tooling that grep's the stdout line.)

**Multi-cohort evaluation**. With `--auto-cohorts` (on by default), Phase 3 *additionally* evaluates the same three picks (top-by-GAT, baseline, iteration winner) on every other available external cohort:
- It probes `Data/{MIMIC,EHRShot,NHANES}/preprocessed/<icd>_<name>_<cohort>.parquet` and skips cohorts whose parquet doesn't exist.
- NHANES is skipped automatically for diseases not in `{242, 250, 696, 714, 2452}` (per `Code/preprocess_nhanes_external.py:NHANES_CASE_DEFINITIONS`).
- Per-cohort baseline directories are auto-resolved (`evaluation_results_external_v2_full/<cohort>` → `evaluation_results_external_v2/<cohort>` → `evaluation_results_unified/external_validation_<cohort>`).
- Each cohort gets its own bootstrap CI (n_boot=500) on top-by-GAT, first biomarker, and iteration winner.

The rendered `final_report.md` adds a "Cross-cohort evaluation" section with two sub-tables:
1. **Top-by-GAT real AUC per cohort** — per-cohort point AUC + 95% CI + baseline_best + Δ.
2. **Before vs after the GAT iteration per cohort** — per-cohort first-biomarker vs iteration-winner with their CIs and the Δ that isolates the value the propose-score-refine loop added over the literature seed on each cohort.

The JSON sidecar (`final_report.json`) gets a structured `cohorts: [...]` array with all metrics per cohort. The single-line PASS/FAIL stdout signal continues to use only the primary cohort's AUC vs `baseline_best`, so existing automation that grep's the stdout line is unaffected.

Render `final_report.md` from the template at `assets/template_final_report.md` and write a machine-readable `final_report.json`. Print a single PASS/FAIL stdout line as the acceptance signal.

Invoke (the `--auto-cohorts` default does the multi-cohort eval; pass `--no-auto-cohorts` for primary-only):
```bash
python skills/biomarker-discovery/scripts/finalize.py \
    --csv <output_dir>/iterative_details.csv \
    --parquet <primary_cohort_parquet> \
    --target-col icd_<icd> \
    --baseline-dir Results/<icd>/evaluation_results_external_v2/<cohort> \
    --out-dir <output_dir> \
    --disease-id <icd> --disease-name <name> --cohort <cohort> \
    --gat-checkpoint GAT/<icd_disk>/trained_models/best_by_loss.pt \
    --checkpoint-variant best_by_loss \
    --n-bootstrap 500 --random-seed 42
```

(Use the `_disk` folder for the checkpoint path: ICD 2452 → folder 2542, all others → folder == ICD.)

## Outputs

```
Results/<icd>/biomarker_discovery/<YYYYMMDD-HHMM>/
├── literature_review.md       # Phase 1 deliverable
├── iterative_details.csv      # Phase 2 log (one row per scored expression)
├── prompts/                   # Phase 2 audit trail
│   ├── iter_01.txt            # the exact prompt Claude saw at iter 1
│   ├── iter_01_response.txt   # Claude's raw response
│   └── ...
├── final_report.md            # Phase 3 human-readable deliverable
└── final_report.json          # Phase 3 machine-readable summary
```

The final report contains the chosen expression, its measured cohort AUC + 95% bootstrap CI, head-to-head comparison vs. the 3 published external-method baselines, top-10 trajectory table, and a PASS/FAIL verdict.

## Constraints / contracts (Claude must respect)

1. **Silent real_auc**: `real_auc` values are NEVER included in iteration-loop prompts. Only `gat_score` may appear in the prompt's history block. (Mirrors `Code/iterative_llm_gat.py:386`.)
2. **Validation always runs**: every batch of LLM-proposed expressions is filtered through `gat_agent_tool.validation.parse_candidates` BEFORE being sent to `score_expressions`. This filters out numeric literals, unknown features, and Markdown noise.
3. **One scoring path**: `mcp__gat-multi-scorer__score_expressions(disease_id, expressions)` is the ONLY scoring path. No direct GAT calls, no fallback to Python eval.
4. **DSL rules** are taken verbatim from [`references/expression_dsl.md`](references/expression_dsl.md) and pasted into iteration prompts. Don't paraphrase.
5. **Checkpoint variant** stays consistent across the run — if `--checkpoint-variant=best_by_loss` (default), use it for both the registry-loaded GAT (Phase 2) and the `gat_checkpoint` field in `final_report.json` (Phase 3).
6. **Disk-folder typo**: ICD 2452 lives at `GAT/2542/` on disk. The MCP server's registry handles this transparently. Always pass `disease_id="2452"` (the public ID), never `"2542"`.
7. **Output directory immutability**: each run creates a fresh `<YYYYMMDD-HHMM>` directory. Don't overwrite prior runs.

## References

- [`references/diseases.md`](references/diseases.md) — 13-disease lookup table (ICD ↔ folder ↔ name ↔ cohort parquet ↔ NHANES availability)
- [`references/expression_dsl.md`](references/expression_dsl.md) — DSL rules + worked examples + LLM-mistake catalog
- [`references/literature_review.md`](references/literature_review.md) — Phase 1 procedural detail
- [`references/iteration_loop.md`](references/iteration_loop.md) — Phase 2 procedural detail
- [`scripts/silent_real_auc.py`](scripts/silent_real_auc.py) — Phase 2 helper: fills `real_auc` column
- [`scripts/finalize.py`](scripts/finalize.py) — Phase 3 main: bootstrap + baseline comparison + report rendering
- [`assets/template_final_report.md`](assets/template_final_report.md) — Phase 3 markdown template

## Known limitations

- Bootstrap CI is bias-corrected only via `max(AUC, 1-AUC)` direction-flipping; no BCa.
- Baseline files are hand-collected (not produced by this skill); they live under `Results/<icd>/evaluation_results_external_v2/<cohort>/`. If a baseline file is missing for a (cohort, disease) cell, that tool is excluded from `baseline_best`.
- The `--skip-literature` mode falls back to a minimal seed set if no prior `literature_review.md` exists; this seed set is generic and may underperform vs. a fresh literature review.
- When NHANES has fewer than 50 positives for the target disease, bootstrap CIs are wide (commonly ±0.05 AUC). Interpret PASS/FAIL with care for low-positive cohorts.

## See also

- `Code/iterative_llm_gat.py` — the canonical HF-driven reference implementation this skill mirrors
- `gat_agent_tool/gat_agent_tool/multi_mcp_server.py` — the MCP server this skill calls
- `gat_agent_tool/gat_agent_tool/validation.py` — `parse_candidates` (used in Phase 2)
- `Code/_external_scoring.py` — `eval_expression`, `univariate_auc`, `bootstrap_auc_ci` (used in Phases 2 and 3)
