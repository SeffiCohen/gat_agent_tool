# Phase 1 — literature review (deep research via a parallel Opus 4.8 Workflow)

## Purpose

Ground the iter-1 200-batch in published evidence rather than blind LLM guessing. Without literature context, Phase 2's first 200 wastes itself rediscovering well-known biomarker ratios (NLR, PLR, MLR) instead of building on them. This phase frontloads that knowledge — and does it with the **`Workflow` tool fanning out 3 parallel Opus 4.8 agents at `max` effort** (one `agent()` call per research angle) so we cover the disease's pathophysiology, established biomarker literature, and recent preprints in one wall-clock window.

## Inputs

- `disease_id` — the ICD code passed via the skill's CLI (e.g. `714` for rheumatoid arthritis).
- `output_dir` — already resolved by `SKILL.md` (typically `Results/<disease_id>/biomarker_discovery/<YYYYMMDD-HHMM>/`).

## Outputs

A single file: `output_dir/literature_review.md`. Synthesized by the main thread from the 3 subagent reports. Must contain:
1. The full GAT feature list (verbatim from the MCP tool).
2. 5–10 known biomarker ratios with PMID citations, expressed in the GAT's exact DSL.
3. Per-feature mechanistic rationale tying each allowed analyte to the specific disease.
4. 5–10 seed expressions in the GAT's exact DSL (will be visible to Claude when generating the iter-1 200-batch in Phase 2 — they're context, NOT scored as iter 0 under the new flow).
5. A numbered references list (PubMed + bioRxiv).

Plus 3 sidecar files for audit:
- `output_dir/literature_review_agent_mechanism.md`
- `output_dir/literature_review_agent_established.md`
- `output_dir/literature_review_agent_preprints.md`

## Expected runtime

20–40 minutes wall-clock. The 3 agents run **in parallel** inside one Workflow (a single `parallel()` fan-out over 3 `agent()` calls), so the slowest agent gates the wall-clock. Each agent typically takes 8–20 min depending on how many PMIDs/preprints it pulls.

## Why a Workflow (and why Opus 4.8 at max effort)

- **Parallelism + determinism**: the `Workflow` tool runs the 3 research angles concurrently as one deterministic fan-out (`parallel()` over 3 `agent()` calls), surfaces per-agent progress in `/workflows`, and returns a single structured array the main thread synthesizes. Same context budget on the main thread; 3× the work in the same wall-clock.
- **Strongest model at max effort** (`model: "opus"`, `effort: "max"`): `model: "opus"` always resolves to the latest Opus (**Opus 4.8**), and `effort: "max"` puts each agent on the deepest reasoning tier — together the **"Opus 4.8 max"** configuration. Literature synthesis benefits from a model that can hold dozens of abstracts in working memory and prioritize quality citations over quantity. Sonnet is faster but consistently produces shallower lit reviews — defaults to skimming the top-5 PubMed hits without cross-checking. Opus 4.8 at max effort reads more, cites more carefully, and catches mechanistic implausibilities (e.g., "this PMID is about COVID-related thrombocytopenia, not your target disease").
- **Independent context windows**: each agent has its own window, so the main thread doesn't get bloated with raw PubMed XML. Only the synthesized markdown returns.
- **Trust boundary**: the main thread can't see what's *inside* each agent's reasoning, only the returned markdown report. So the synthesis step (Step 4 below) has a chance to catch hallucinated PMIDs by cross-checking IDs that appear in 2+ agent reports.

## MCP tools used (BY the Workflow agents, not the main thread)

The Workflow agents call these — the main thread should NOT call them directly:
- `mcp__plugin_bio-research_pubmed__search_articles` + `get_article_metadata` — peer-reviewed evidence.
- `mcp__plugin_bio-research_biorxiv__search_preprints` + `get_preprint` — pre-publication candidates.
- `mcp__plugin_bio-research_ot__search_entities` + `query_open_targets_graphql` — target-disease evidence.

Each agent runs on **Opus 4.8 at max effort** (set via `model: "opus"` + `effort: "max"` on the `agent()` call inside the Workflow). These three literature MCP servers (PubMed, bioRxiv, Open Targets) are unauthenticated, so Workflow agents can reach them on demand via tool search.

The MAIN thread runs only:
- `mcp__gat-multi-scorer__get_feature_list` — to fetch the feature allowlist before spawning agents (so each agent receives the EXACT 14-or-15-feature list and never proposes off-allowlist features).

---

## Step 1: Main thread — get the GAT feature list

```
mcp__gat-multi-scorer__get_feature_list(disease_id="<disease_id>")
```

Returns:
```
{
  "features": ["lab_NEUT_abs_last", "lab_LYMpct_last", "lab_PLT_last", ...],
  "operators": ["+", "-", "*", "/"],
  "max_depth": 3,
  ...
}
```

CACHE the `features` list — it will be **embedded in each subagent's prompt** so they can't propose off-allowlist features.

For RA (`disease_id="714"`), the list is 15 `lab_*_last` columns. For an exact per-disease feature inventory, see `references/diseases.md`.

## Step 2: Main thread — get disease metadata

Read `references/diseases.md` and look up the row for the given `disease_id`. Capture:
- Descriptive disease name (e.g. `RheumatoidArthritis`, `T1Diabetes`, `CrohnDisease`).
- ICD-9 and ICD-10 codes.

Use the **descriptive name** in subagent prompts — search engines understand "rheumatoid arthritis" much better than "ICD 714".

## Step 3: Launch the Phase-1 Workflow (3 parallel Opus 4.8 agents)

**Critical**: run the 3 research angles as a single `Workflow` fan-out (one `parallel()` over 3 `agent()` calls). Do NOT serialize them — that defeats the parallelism that makes this phase finish in 20–40 min instead of 60–120 min.

Each `agent()` call:
- Uses `model: "opus"` — resolves to the latest Opus (**Opus 4.8**), overriding the session model.
- Uses `effort: "max"` — the deepest reasoning tier (the **"Opus 4.8 max"** configuration). The standalone `Agent` tool has no `effort` knob, so the Workflow path is the only way to pin max effort for these subagents.
- Uses `agentType: "general-purpose"` — that registry type has `*` tools, including the `mcp__plugin_bio-research_*` family; Workflow agents reach session-connected MCP servers on demand.
- Receives the SAME shared header (disease name + 14-or-15-feature allowlist + DSL constraints) plus its own focused research question (the three prompts below).
- Returns a self-contained markdown report as its final message.

**Invoking a `Workflow` from a skill is an explicit opt-in** — these instructions directing you to call it satisfy the Workflow tool's opt-in requirement (no separate "ultracode" toggle needed).

Assemble the script below — inlining the three agent prompt bodies from §"Agent 1/2/3" verbatim into the `AGENTS` array — and invoke it:

```
Workflow({
  script: "<the script below, with the 3 prompt bodies inlined>",
  args: {
    disease_id:    "<icd>",
    disease_name:  "<descriptive name, e.g. rheumatoid arthritis>",
    features_list: "<newline-joined allowlist from Step 1>",
    today:         "<YYYY-MM-DD>"   // resolve via Bash `date +%F`; the Workflow
                                    // sandbox has no clock (new Date() throws)
  }
})
```

```javascript
export const meta = {
  name: 'biomarker-litreview',
  description: '3 parallel Opus 4.8 research agents (mechanism / established ratios / preprints) for CBC biomarker discovery',
  phases: [{ title: 'Research', detail: '3 parallel Opus 4.8 agents at max effort', model: 'opus' }],
}

// args is passed verbatim by the main thread (see the invocation above).
const { disease_id, disease_name, features_list, today } = args

const HEADER = `You are doing deep research on ${disease_name} (ICD ${disease_id})
for downstream CBC biomarker discovery. The downstream pipeline accepts ONLY
arithmetic combinations of these CBC features (case-sensitive, exact match):

  ${features_list}

Operators: + - * / (no numeric literals, no functions, depth <= 3).`

// Each `prompt` is HEADER + the verbatim "Agent N" body documented below this
// script. The preprints body interpolates ${today} into its bioRxiv interval.
const AGENTS = [
  { key: 'mechanism',   label: 'litrev:mechanism',   prompt: `${HEADER}\n\n<<Agent 1 — Mechanism body>>` },
  { key: 'established', label: 'litrev:established', prompt: `${HEADER}\n\n<<Agent 2 — Established body>>` },
  { key: 'preprints',  label: 'litrev:preprints',  prompt: `${HEADER}\n\n<<Agent 3 — Preprints body, using ${today}>>` },
]

phase('Research')

// No `schema` opt => each agent() resolves to its final text (the markdown
// report) as a string. Do NOT add a schema here: Step 4 writes that string to
// the sidecar files verbatim, so a schema would silently break the contract.
const reports = await parallel(AGENTS.map(a => () =>
  agent(a.prompt, {
    label:     a.label,
    phase:     'Research',
    model:     'opus',             // -> latest Opus = Opus 4.8
    effort:    'max',              // deepest reasoning tier ("Opus 4.8 max")
    agentType: 'general-purpose',  // * tools incl. bio-research MCP
  }).then(markdown => ({ agent: a.key, markdown })),
))

// One object per angle. filter(Boolean) drops any agent that died (parallel()
// resolves a failed thunk to null). The main thread synthesizes them in Step 4.
return reports.filter(Boolean)
```

The Workflow returns `[{agent:"mechanism",markdown:"..."}, {agent:"established",markdown:"..."}, {agent:"preprints",markdown:"..."}]` to the main thread once all three agents complete.

> **Fallback (no `Workflow` tool):** if the running harness lacks the `Workflow` tool, dispatch the same 3 prompts as parallel `Agent` tool calls (`subagent_type: "general-purpose"`, `model: "opus"`) in a single message. The `Agent` tool can't pin `effort: "max"` (it inherits the session effort), so Opus 4.8 still runs but not necessarily at max effort.

### Agent 1 — Mechanism + Open Targets (`agent()` prompt — `model: "opus"`, `effort: "max"`)

```
description: "Disease mechanism + target genes for {disease_name}"

prompt: |
  You are doing deep research on {disease_name} (ICD {disease_id}) for downstream
  biomarker discovery. The downstream pipeline accepts ONLY arithmetic
  combinations of these CBC features (case-sensitive, exact match):

    {features_list}

  Operators: + - * / (no numeric literals, no functions, depth ≤ 3).

  Your job (focus area: MECHANISM):
  1. Use mcp__plugin_bio-research_ot__search_entities to find the EFO id for
     "{disease_name}".
  2. Use mcp__plugin_bio-research_ot__query_open_targets_graphql to fetch the
     top 25 associated targets (genes) for that EFO id.
  3. For each of the top 10 targets, write a 1-2 sentence note tying the gene
     to a CBC compartment perturbation. Example: IL6 → acute-phase response
     → elevated PLT, RDW, NEUTpct (via NEUTpct/LYMpct). CXCR1 → neutrophil
     trafficking → elevated NEUTpct, lower LYMpct.
  4. For each ALLOWED feature in the list above, write 1-3 sentences on how
     {disease_name} mechanistically perturbs it (or why it's expected to be
     uninformative). If you can't tie a feature to the disease, say so explicitly
     — do NOT fabricate a connection.

  Output: a single markdown document with two sections: "## Top targets" and
  "## Per-feature mechanistic notes". Include EFO id and any PMIDs you cite.
  Under 1500 words. Save to your final response — main thread will write it
  to output_dir/literature_review_agent_mechanism.md.
```

### Agent 2 — Established CBC biomarker ratios (`agent()` prompt — `model: "opus"`, `effort: "max"`)

```
description: "Established CBC biomarker ratios for {disease_name}"

prompt: |
  You are doing deep research on established CBC-derived biomarker ratios for
  {disease_name} (ICD {disease_id}). The downstream pipeline accepts ONLY
  arithmetic combinations of these CBC features (case-sensitive, exact match):

    {features_list}

  Operators: + - * / (no numeric literals, no functions, depth ≤ 3).

  Your job (focus area: ESTABLISHED LITERATURE):
  1. Use mcp__plugin_bio-research_pubmed__search_articles to find peer-reviewed
     papers on biomarker ratios for {disease_name} since 2015. Search terms:
       '"{disease_name}" AND ("NLR" OR "neutrophil-to-lymphocyte" OR "PLR" OR
        "platelet-to-lymphocyte" OR "MLR" OR "monocyte-to-lymphocyte" OR "RDW"
        OR "MCV" OR "complete blood count")'
     mindate=2015, max_results=20.
  2. Use mcp__plugin_bio-research_pubmed__get_article_metadata on the top
     5-10 PMIDs to get abstracts.
  3. For each well-supported ratio (≥1 PMID with positive findings), output a
     row in this format:
       | DSL form | Effect size from abstract | PMID(s) | Notes |
       | --- | --- | --- | --- |
       | lab_NEUT_abs_last / lab_LYMpct_last | NLR > 3.0 → ... | 28748147 | ... |
     CRITICAL: every leaf in "DSL form" MUST be in the allowlist above. If a
     ratio uses a feature not in the allowlist, DROP it (don't try to translate
     it to a different feature).
  4. Then list 5–10 "seed expressions" — DSL strings only, one per line, that
     a downstream LLM should consider as starting points. Same allowlist
     constraint.

  Output: markdown with sections "## Established ratios" (the table) and
  "## Seed expressions" (the line list). Include PMIDs as a footer references
  list. Under 1500 words. Save to your final response — main thread will write
  it to output_dir/literature_review_agent_established.md.
```

### Agent 3 — Recent preprints + novel composites (`agent()` prompt — `model: "opus"`, `effort: "max"`)

```
description: "Recent preprints / novel CBC composites for {disease_name}"

prompt: |
  You are doing deep research on RECENT (2022+) and PREPRINT-stage CBC-derived
  biomarker composites for {disease_name} (ICD {disease_id}). The downstream
  pipeline accepts ONLY arithmetic combinations of these CBC features
  (case-sensitive, exact match):

    {features_list}

  Operators: + - * / (no numeric literals, no functions, depth ≤ 3).

  Your job (focus area: PREPRINTS + NOVEL COMPOSITES):
  1. Use mcp__plugin_bio-research_biorxiv__search_preprints with
       server="biorxiv", terms="{disease_name} biomarker complete blood count",
       interval="2022-01-01/<today>"
     If bioRxiv is sparse for this disease, also try server="medrxiv" with
     same terms.
  2. Use mcp__plugin_bio-research_biorxiv__get_preprint on the top 3-5 hits.
  3. ALSO call mcp__plugin_bio-research_pubmed__search_articles with
       '"{disease_name}" AND ("AISI" OR "SII" OR "PIV" OR "composite biomarker"
        OR "machine learning" OR "predictive score")'
       mindate=2022, max_results=10.
  4. For each candidate composite (≥3 features combined), output a row with the
     DSL form, the source (PMID or bioRxiv DOI), and the reported AUC/effect.
     CRITICAL: drop any candidate whose DSL form requires features not in the
     allowlist above.
  5. Then list 5–10 "novel seed expressions" — combinations YOU think are worth
     exploring based on the literature (still allowlist-only).

  Output: markdown with sections "## Recent composites from preprints + 2022+
  PubMed" and "## Novel seed expressions". Footer references. Under 1500 words.
  Save to your final response — main thread will write it to
  output_dir/literature_review_agent_preprints.md.
```

> **Why `agentType: "general-purpose"` for all three?** That registry type has access to `*` tools (including the `mcp__plugin_bio-research_*` family), and Workflow agents resolve `agentType` from the same registry as the `Agent` tool. `general-purpose` is the safest default — it works in every Claude Code installation that has the bio-research plugin enabled.

> **Why `model: "opus"` + `effort: "max"` and not the defaults?** The agent model otherwise inherits from the parent session — whatever the user opened Claude Code with, possibly Sonnet. Pinning `model: "opus"` ensures the literature synthesis runs on the strongest available model regardless of the session model, and `"opus"` always resolves to the latest Opus version (**Opus 4.8** at the time of this writing). `effort: "max"` then puts each agent on the deepest reasoning tier — together this is the **"Opus 4.8 max"** configuration. Both knobs are Workflow `agent()` options; the standalone `Agent` tool exposes `model` but not `effort`.

## Step 4: Main thread — synthesize the 3 agent reports

After the Workflow completes, the main thread receives the returned array — one `{agent, markdown}` object per research angle. Do this:

1. **Save each agent report verbatim** (the `markdown` field, keyed by `agent`) to `output_dir/literature_review_agent_{mechanism,established,preprints}.md` (audit trail; they're cited from the synthesized doc).

2. **Cross-validate citations**. List every PMID that appears in ≥2 agent reports — those are highest-confidence (multiple agents independently found the same source). PMIDs from a single agent are kept but flagged.

3. **Synthesize into `output_dir/literature_review.md`**:

   ````markdown
   # Literature review — {disease_name} (ICD {disease_id})

   *Generated by 3 parallel Opus 4.8 agents (max effort) via the Workflow tool on {timestamp}.*

   ## Allowed features (from GAT)
   - `lab_NEUT_abs_last`
   - `lab_LYMpct_last`
   - ... (verbatim from Step 1)

   ## Per-feature mechanistic rationale
   *Source: Agent 1 (mechanism)*
   - `lab_NEUT_abs_last` — Neutrophil count. {1-2 sentences from Agent 1's per-feature notes, citing PMID/EFO}.
   - ... (one bullet per allowed feature, only those Agent 1 could mechanistically tie to the disease)

   ## Known biomarker ratios (with citations)
   *Source: Agent 2 (established)*
   {Agent 2's table verbatim, filtered to drop allowlist-violating rows}

   ## Recent composites from preprints + 2022+ PubMed
   *Source: Agent 3 (preprints)*
   {Agent 3's table verbatim, filtered to drop allowlist-violating rows}

   ## Seed expressions for iter 1 context
   These 5-10 expressions are **NOT** scored as iter 0 (under the new 200-batch
   flow there is no iter 0). They serve as context Claude can read when
   generating iter 1's 200-batch.

   ```
   1.  lab_NEUT_abs_last / lab_LYMpct_last      # NLR — Agent 2
   2.  lab_PLT_last / lab_LYMpct_last           # PLR — Agent 2
   3.  lab_MONOpct_last / lab_LYMpct_last       # MLR — Agent 2
   4.  (lab_PLT_last * lab_NEUT_abs_last) / lab_LYMpct_last  # SII — Agent 3
   5.  lab_RDW_last / lab_HB_last               # anemia composite — Agent 1
   ... (5-10 total, sourced from across the 3 agents)
   ```

   ## References
   ### Highest-confidence (cited by ≥2 agents)
   [1] PMID 28748147 — ...

   ### Single-agent citations
   [2] PMID 29876543 — ... (Agent 2 only)
   [3] bioRxiv 2023.05.12.540712 — ... (Agent 3 only)
   ````

4. **Validate seeds before writing** — for each line in the "Seed expressions" block, run it through `gat_agent_tool.validation.parse_candidates` to confirm the leaves are in the allowlist and the DSL is clean. Drop any seed that fails. Log dropped seeds to stderr.

## Quality checklist (run before considering Phase 1 complete)

- [ ] All 3 agent sidecar files exist under `output_dir/literature_review_agent_*.md` and are non-empty.
- [ ] `output_dir/literature_review.md` exists and contains all 5 required sections.
- [ ] Every "Seed expressions" line passes `parse_candidates` (allowlist-only, no numeric literals).
- [ ] At least 5 distinct seed expressions across all 3 agents (after dedup + allowlist filtering). If <5, abort the run with a clear error — the literature was sparse enough that Phase 2 won't get useful context.
- [ ] References list has ≥3 PMIDs from peer-reviewed PubMed (preprints alone are a soft signal — peer-reviewed grounding is required for Phase 3's narrative section).
- [ ] No banned substrings in the synthesized doc: `0.000001`, `epsilon`, `divide-by-zero` (the GAT rejects numeric literals — these strings would suggest a hallucinated CBC composite that uses one).

## Handoff to Phase 2

`output_dir/literature_review.md` exists. Phase 2 reads it as **context only** — the LLM has its content in working memory when generating the iter-1 200-batch from `assets/initial_prompt.md`. The seeds are not scored as iter 0 (no iter 0 exists under the new flow), but their presence in Claude's working memory raises the floor on iter-1 proposal quality.

Skip Phase 1 only when `--skip-literature` is set; Phase 2 runs without context in that case (lower iter-1 baseline quality but still functional).
