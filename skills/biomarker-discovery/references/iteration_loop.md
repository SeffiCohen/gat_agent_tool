# Phase 2 — iteration loop

You are driving a propose-score-refine loop: you propose 200 biomarker expressions per iteration, the GAT scores them via MCP tools, the scores feed back into your next prompt, you refine. Repeat until iteration cap or plateau.

**Inputs:**
- `disease_id` — ICD folder string (e.g. `"714"`).
- `output_dir` — directory containing Phase 1's `literature_review.md` and where this phase writes `iterative_details.csv`.
- `iterations` — max outer loop count (default `100`). The propose-score-refine loop will run up to this many iterations, but `plateau_patience` (default `5`) typically stops it earlier once the GAT signal saturates.
- `pop_per_iter` — expressions to propose per iteration (default `200`). Yes, two hundred — this is intentional. The initial batch (`iter == 1`) is the LLM's "first pass" guess at the disease's CBC arithmetic space; subsequent batches refine with GAT-score feedback.
- `plateau_patience` — stop if `best_gat_ever` does not strictly improve for this many consecutive iterations (default `5`).

**Output:** `output_dir/iterative_details.csv` with these exact columns (matching `Code/iterative_llm_gat.py:19-20`):

```
iter, rank_in_iter, expression, gat_score, real_auc, is_valid, duplicate_of
```

**Expected runtime:** 30 min – 2 h depending on `iterations`, `pop_per_iter`, and how often the loop plateaus early. With `plateau_patience=5` and a typical disease, expect 8–20 iterations before convergence.

**The reference implementation** is the HuggingFace-driven propose-score-refine loop described in the paper. This phase reproduces the same loop logic but with Claude as the LLM, calling MCP tools instead of `model.generate()`, and with a larger per-iteration batch (200 vs 10) and no iter-0 literature scoring (the literature review is context-only).

---

> **EXPERIMENTAL CONTROL — DO NOT VIOLATE**: The `real_auc` column in `iterative_details.csv` is computed by `scripts/silent_real_auc.py` for audit and final-report purposes only. **You must NEVER include `real_auc` values in the prompt block of any subsequent iteration.** Only `gat_score` may appear in history shown to the LLM (yourself, in the next iteration's reasoning). This invariant is enforced silently by `iterative_llm_gat.py:386` in the reference implementation. Mirror it here. If you accidentally let `real_auc` leak into the prompt, the entire experiment is invalidated.

---

## Step 0 — Initialize state

Set up scratch state for the run:

```bash
mkdir -p "${OUTPUT_DIR}/prompts"
```

- Create `${OUTPUT_DIR}/iterative_details.csv` with header row only:
  ```
  iter,rank_in_iter,expression,gat_score,real_auc,is_valid,duplicate_of
  ```
- Initialize `seen_exprs = set()` — in-memory dedup of expression strings already scored.
- Initialize `history = []` — list of `{iter, rank_in_iter, expression, gat_score}` dicts. This is the ONLY thing allowed in prompts. Do not add `real_auc` here.
- Initialize `best_gat_ever = -inf`, `no_improve_count = 0`.
- Cache `features` and `operators` once via:
  ```
  mcp__gat-multi-scorer__get_feature_list(disease_id=<disease_id>)
  ```
  Hold them for the rest of the run; do not re-fetch each iteration.
- **Read `${OUTPUT_DIR}/literature_review.md` as context** — its content informs the LLM's first-pass proposals but is NOT scored as iter 0. Under the new flow, iter 0 does not exist; the literature review is reference material that primes Claude's understanding before the iter-1 batch is generated.

> **No iter 0 under the new flow.** Earlier versions of this skill scored 5–10 literature seeds at `iter==0`. That step is gone. The LLM now starts directly at `iter == 1` with a 200-batch from `assets/initial_prompt.md`. The "first expression" baseline (used by Phase 3's before-vs-after-GAT comparison) is the row at `iter == 1, rank_in_iter == 0` — the LLM's #1 self-ranked pick from the very first batch.

## Step 1 — Iterate from `it = 1` to `iterations`

For each iteration `it` from `1` upward:

### Step 1a — Build prompt

Two prompts, branched on `it`:

- **`it == 1`** (initial 200-batch): use the verbatim asset at `assets/initial_prompt.md`. The placeholder `{disease_name}` is filled at runtime with the descriptive disease name (from `references/diseases.md`).
- **`it >= 2`** (GAT-feedback iterations): use the verbatim asset at `assets/iteration_prompt.md`. Two placeholders: `{disease_name}` and `{history_block}` (constructed per Step 1b below).

Both prompts share:
- The 14-feature canonical CBC panel (`lab_BASO_pct_last`, `lab_EOS_pct_last`, `lab_HB_last`, `lab_HCT_last`, `lab_LYMpct_last`, `lab_MCH_last`, `lab_MCHC_last`, `lab_MCV_last`, `lab_MONOpct_last`, `lab_NEUTpct_last`, `lab_PLT_last`, `lab_RBC_last`, `lab_RDW_last`, `lab_WBC_last`).
- The 4-rule constraint set: only `+ - * /`, only the listed variables, prefer ratios/differences/normalized combinations, interaction terms are fine.
- The output format: 200 expressions, one per line, ordered LLM-best-first, no numbering or prose.

> **Why a separate asset?** The verbatim prompts live in `assets/initial_prompt.md` and `assets/iteration_prompt.md` so they're auditable, regression-tested (`tests/verify_initial_prompt.py`), and re-renderable by tooling without re-parsing this markdown. Don't paraphrase them inline — load and `.format(...)` the asset.

> **Per-disease feature override**: 11 of 13 diseases use exactly the 14-feature panel above. The two exceptions — RA (714) and Hashimoto (2452) — *add* `lab_NEUT_abs_last` (absolute neutrophil count) as a 15th feature. The actual allowlist enforced by `parse_candidates` and the GAT comes from the runtime `get_feature_list(disease_id)` call (Step 0 cache). When the disease's allowlist has 15 features, append the 15th line under the canonical 14 in the rendered prompt before sending. The asset file holds the canonical 14; the iteration code adds extras.

#### Step 1b — `{history_block}` construction (it >= 2 only)

Mirrors `iterative_llm_gat.py:115-136`. Skip when `it == 1`.

1. `scored = [h for h in history if h["gat_score"] is not None]`. Skip if empty.
2. `half = max(1, max_history_shown // 2)` where `max_history_shown = 20` by default (10 top-by-GAT + 10 most-recent — keeps the prompt short while showing both global best and recency).
3. `top = sorted(scored, key=lambda h: h["gat_score"], reverse=True)[:half]` — global best.
4. `last_iter = max(h["iter"] for h in history)`; `recent = [h for h in history if h["iter"] == last_iter][:half]` — recency signal.
5. Dedupe by expression string, keeping the higher-scored copy: `shown = {}; for h in top + recent: shown.setdefault(h["expression"], h)`.
6. Sort the merged set by `gat_score` desc.
7. Format each line as `f"  {h['gat_score']:.3f}   {h['expression']}"`. Single space-padding gives a readable two-column table.

**CRITICAL**: do NOT include `real_auc` anywhere in this block. Only `gat_score`. This is the experimental control restated for emphasis — it is the load-bearing invariant of the entire study.

### Step 1c — Save the prompt to disk

```
${OUTPUT_DIR}/prompts/iter_{it:02d}.txt
```

This is the audit trail; downstream verification reads from here. Save the FINAL rendered prompt (after `{disease_name}` and `{history_block}` substitution), not the raw template.

### Step 1d — Generate `pop_per_iter` expressions

You (the LLM driving the skill) write 200 expressions inline in your response, one expression per line, no numbering or bullets, no prose around them, ordered from highest-confidence to lowest-confidence (the LLM's self-ranking). Save your raw output (every line you produced for this iteration's batch, before any cleanup) to:

```
${OUTPUT_DIR}/prompts/iter_{it:02d}_response.txt
```

> **Self-ranking matters.** The first row (`rank_in_iter == 0`) of `iter == 1` is later used by Phase 3 as the **first-expression baseline** — the comparison point that isolates the iteration loop's actual added value. Order by genuine confidence, not alphabetically or randomly.

### Step 1e — Validate

Pipe the raw response through `gat_agent_tool.validation.parse_candidates`. This strips bullets, comments, and code fences; rejects expressions containing numeric literals; rejects expressions referencing unknown features; deduplicates while preserving order:

```bash
python -c "
import json, sys
from gat_agent_tool.validation import parse_candidates
feats = json.loads(sys.argv[1])
ops = ['+','-','*','/']
print('\n'.join(parse_candidates(sys.stdin.read(), feats, ops)))
" "$FEATURES_JSON" < "${OUTPUT_DIR}/prompts/iter_${it_padded}_response.txt" > "${OUTPUT_DIR}/prompts/iter_${it_padded}_validated.txt"
```

`$FEATURES_JSON` is the JSON-encoded list cached in Step 0. See `references/expression_dsl.md` for the DSL rules `parse_candidates` enforces.

> **Order preservation matters.** `parse_candidates` deduplicates while preserving order. The first VALID expression from your response stays at index 0 of the validated list — this is what gets `rank_in_iter == 0` and becomes the Phase 3 first-expression baseline. Don't reorder after validation.

### Step 1f — Dedupe against `seen_exprs`

```python
new_exprs = [e for e in validated if e not in seen_exprs]
```

If `new_exprs` is empty (every proposal was a duplicate of something already scored), increment `no_improve_count` and continue to the next iteration — do not call the GAT, do not write any rows.

### Step 1g — Score new expressions in batch

```
results = mcp__gat-multi-scorer__score_expressions(
    disease_id=<disease_id>,
    expressions=new_exprs,
)
```

Append one CSV row per result with `iter=it`, `rank_in_iter=0..len(new_exprs)-1`, the expression, the `gat_score`, empty `real_auc` (filled in Step 1h), `is_valid` from the GAT response, and `duplicate_of=""` (these are by construction not duplicates).

> **rank_in_iter assignment.** `rank_in_iter` is the position in the LLM's self-ranked list AFTER validation+dedup, NOT after GAT scoring. So `iter==1, rank_in_iter==0` is the first expression that survived `parse_candidates` — which is the LLM's #1 pick if it parsed cleanly, or the next valid expression if its #1 had e.g. a numeric literal.

Update state:
- `seen_exprs.update(new_exprs)`
- For each scored expr: `history.append({"iter": it, "rank_in_iter": k, "expression": e, "gat_score": s})`

### Step 1h — Silent real_auc

```bash
python ${SKILL_DIR}/scripts/silent_real_auc.py \
    --csv "${OUTPUT_DIR}/iterative_details.csv" \
    --parquet "${COHORT_PARQUET}" \
    --target-col "icd_${DISEASE_ID}"
```

The script appends/updates `real_auc` for every CSV row that doesn't already have one. Do NOT read its stdout into prompt context. The CSV update is the only channel.

### Step 1i — Plateau detection (early stopping)

```python
this_iter_scores = [r["score"] for r in results if r["score"] is not None]
if not this_iter_scores:
    no_improve_count += 1
else:
    best_this_iter = max(this_iter_scores)
    if best_this_iter > best_gat_ever:
        best_gat_ever = best_this_iter
        no_improve_count = 0
    else:
        no_improve_count += 1

if no_improve_count >= plateau_patience:   # default 5
    break  # convergence — early stop
```

The loop terminates when **5 consecutive iterations** fail to strictly improve `best_gat_ever`. With `pop_per_iter=200` and 14–15 features, the GAT's surrogate AUC saturates relatively fast (typically by iter 8–15), so the default `plateau_patience=5` is calibrated to neither cut off promising trajectories early nor waste budget on stale exploration.

## Step 2 — Final summary

After the loop ends (whether by `iterations` cap or plateau), print to console — and ONLY to console, never into prompt context:

```
Iteration loop complete: {iter_done} iters, {len(seen_exprs)} unique expressions scored,
{n_valid} valid, best_gat_ever={best_gat_ever:.4f}.
Calling Phase 3 (finalize.py) next.
```

Then invoke Phase 3 — see `SKILL.md` for the exact `scripts/finalize.py` invocation. Phase 3 will then:
- Pick the **first expression** (row at `iter==1, rank_in_iter==0`) as the baseline.
- Pick the **iteration winner** (highest gat_score among `iter > 1` rows).
- Bootstrap CIs for both on every available external cohort.
- Emit two stdout signals: the legacy `PASS/FAIL` line (vs published external-method baselines) AND the new `BASELINE_DELTA` line (iter_winner real AUC vs first_expression real AUC).

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| All expressions parse-fail (`is_valid=False` everywhere) | LLM is using wrong feature names. | Verify against `get_feature_list(disease_id)` cache; check case (feature names are case-sensitive). |
| All `gat_score=None` | Parse failures at the GAT layer (often implicit multiplication, `lab_Xlab_Y`). | Add explicit `*` between adjacent features; see `references/expression_dsl.md`. |
| `gat_score` plateaus at iteration 1 | LLM not exploring the space — same ratios, same operators. | Add an explicit dedupe instruction to the prompt; rebalance the history block toward recent-iter entries. With `pop_per_iter=200`, this is rare unless the LLM is ignoring the "200 NEW expressions" instruction. |
| Plateau detected too early | `plateau_patience=5` may still be tight for noisy disease signals. | Pass `--plateau-patience 8` or `10` for harder problems. |
| MCP tool timeouts on first call | GAT checkpoint slow to load. | Pre-warm at startup: call `mcp__gat-multi-scorer__list_diseases` once before the loop begins; the server caches per-disease checkpoints after first load. |
| Duplicate-heavy iterations (Step 1f empties `new_exprs`) | LLM regenerating known-good ratios with the same numerator/denominator orientation. | Add an explicit dedupe instruction to the prompt; rebalance the history block toward recent-iter entries (lower the global-best half) to reduce the LLM's anchoring on top scorers. |
| `real_auc` accidentally appears in the prompt | History dict was extended with `real_auc` somewhere. | Audit `history.append(...)` calls. The dict must contain ONLY `iter, rank_in_iter, expression, gat_score`. Re-grep `iter_*.txt` under `${OUTPUT_DIR}/prompts/` for `real_auc` to confirm cleanup. |
| Phase 3 reports `first_expression` as a low-confidence pick | The LLM's #1 self-ranked output (its first line) was rejected by `parse_candidates` (numeric literal, unknown feature, etc.) and the next-valid expression became `rank_in_iter==0`. | Inspect `prompts/iter_01_response.txt` vs `prompts/iter_01_validated.txt` to see which of the LLM's top picks were dropped. The new flow's baseline-quality is sensitive to the LLM's first few lines parsing cleanly. |
