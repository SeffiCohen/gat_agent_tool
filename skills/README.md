# Agent skills

Two Claude Code *skills* that turn the released Graph-Attention scorer into an end-to-end
research workflow:

| Skill | Question it answers | Stage |
|---|---|---|
| [`biomarker-discovery/`](biomarker-discovery/) | *Which CBC arithmetic expressions look most predictive for this disease?* | search |
| [`biomarker-explainer/`](biomarker-explainer/) | *Why does this expression work, and does the literature agree?* | interpretation |

They are **optional companions** to the scorer, not part of it. The scorer itself
(`gat_agent_tool/`, `Code/`, `GAT/`) is fully usable without either skill — see the
top-level [`README.md`](../README.md). These skills package the propose–score–refine
protocol described in the paper so that a researcher can reproduce it on their *own*
cohort.

---

## 1. What each skill does

### `biomarker-discovery`

A literature-grounded **propose → score → refine** loop, driven by Claude in a single
Claude Code session:

1. **Literature review.** Parallel research agents survey mechanism evidence, established
   haematological ratios (NLR, PLR, RDW, …) and recent preprints for the target disease,
   and emit seed expressions in the scorer's expression grammar.
2. **Iteration loop.** Claude proposes batches of candidate expressions; each batch is
   validated (`gat_agent_tool.validation.parse_candidates`) and scored **only** through
   `mcp__gat-multi-scorer__score_expressions`. Predicted AUCs are fed back into the next
   prompt. The loop early-stops on a plateau of the predicted AUC.
3. **Finalisation.** The winning expression is evaluated on *your* labelled cohort with a
   bootstrap 95 % CI, compared against the first-guess expression from iteration 1, and
   written up as `final_report.md` / `final_report.json`.

> **Load-bearing experimental control.** The measured (real) AUC is logged to the run CSV
> but is **never** shown to the proposing model — only the GAT's *predicted* AUC is. The
> loop is a test of whether the distilled prior alone can steer search. Preserve this if
> you modify the skill; leaking real AUC into a prompt invalidates the experiment.

### `biomarker-explainer`

Given one expression, produces two deliverables:

1. **Glass-box Shapley figure.** The expression is fit as a one-predictor logistic model on
   a labelled cohort, and that prediction is attributed back to each *raw* blood-count
   feature by exact interventional Shapley values against a median reference patient. Sign
   and magnitude therefore reflect how each feature is actually used inside the formula
   (numerator vs denominator, orientation included). Rendered as an expression tree with
   each leaf coloured by its contribution.
2. **Expected-vs-surprising literature report.** A deep, fan-out literature review
   classifies every component as **Expected** (model direction concordant with established
   clinical literature), **Surprising** (discordant, or no recognised link despite
   non-trivial impact — a hypothesis-generating signal) or **Unclear**. A dedicated
   verification pass re-resolves every PMID/DOI and drops anything it cannot confirm.

Packaged as a one-page `.tex` + `.pdf` per biomarker.

---

## 2. What ships here — and what does not

This repository is a **code + weights** artifact. Consistent with the premise of the
paper, it contains no patient-level data of any kind.

**Included:** the two skills, the trained per-disease GAT checkpoints (`GAT/`), the model
and expression-graph code (`Code/`), and the scorer package with its MCP server
(`gat_agent_tool/`).

**Not included — you must supply it:**

| Missing | Why | Consequence |
|---|---|---|
| `Data/**` — labelled cohort parquets | Patient-level data; MIMIC-IV and EHRShot are credentialed-access | Every step that measures a *real* AUC (bootstrap CIs, Shapley attribution) needs one |
| `Results/**` — pipeline outputs | Produced by the private research pipeline | Baseline head-to-head comparison is unavailable (see §7) |
| Third-party LLM-tool comparison expressions | Hand-collected outputs of external commercial services; not redistributable | The published-baseline gate cannot be evaluated |

The 13 discovered expressions reported in the paper **are** included, as
[`biomarker-explainer/data/winner_expressions.csv`](biomarker-explainer/data/winner_expressions.csv)
(`disease_folder,disease_name,winner_expression`). These are model outputs, not patient data, so the
explainer's `--icd` shortcut resolves offline; you still supply the cohort it is attributed on.

---

## 3. Prerequisites

### 3.1 The scorer and its Python dependencies

```bash
# from the repository root
pip install -r requirements.txt
pip install -e gat_agent_tool[all]      # provides the `gat-agent-multi-mcp` console script
```

Beyond the core scorer dependencies (`numpy`, `pandas`, `torch`, `torch-geometric`), the
skill scripts additionally need:

```bash
pip install "scikit-learn>=1.3" "asteval>=0.9.31" "matplotlib>=3.6"
```

`scikit-learn` and `asteval` back the AUC/expression-evaluation helpers in
`Code/_external_scoring.py`; `matplotlib` renders the Shapley figure. (The `shap` package
is *not* required — Shapley values are computed exactly, in closed form over the
expression.)

### 3.2 The MCP scoring server

`biomarker-discovery` scores **exclusively** through the `gat-multi-scorer` MCP server.
The repository ships a ready-to-use [`.mcp.json`](../.mcp.json). Its arguments
(`--gat-root GAT`, `GAT_AGENT_TOOL_CODE_DIR=Code`) are **relative**, so Claude Code must be
launched from the repository root — the directory containing `GAT/`, `Code/` and
`.mcp.json`. Verify the server is live with:

```
mcp__gat-multi-scorer__list_diseases()
```

which should return the 13 released disease IDs. If the tool is not found, confirm
`gat-agent-multi-mcp` is on `PATH` (step 3.1) and restart Claude Code so it reconnects.

### 3.3 System binaries (explainer PDF packaging only)

| Binary | Package | Needed for |
|---|---|---|
| `pandoc` | `pandoc` | Markdown → LaTeX |
| `pdflatex` (or `xelatex`) + `latexmk` | a TeX distribution | LaTeX → PDF |
| `pdfunite` | `poppler-utils` | merging the batch compendium (optional) |

Stage 1 (the Shapley figure and findings JSON) needs none of these.

### 3.4 Harness features

Both skills call agentic harness features that are **not** distributed with this
repository and may not exist in your environment:

- the **`Workflow`** tool, for parallel fan-out research (`biomarker-discovery` documents a
  fallback to parallel `Agent` calls; `biomarker-explainer` currently has no fallback and
  its Stage 2 is unavailable without it);
- literature MCP servers (PubMed, bioRxiv, Consensus, Open Targets) and/or
  `WebSearch`/`WebFetch`.

Without these, Stage 1 of the explainer and Phases 2–3 of discovery still run; the
literature phases do not.

### 3.5 Your cohort parquet

Every measured-AUC step reads a single parquet with one row per patient:

| Column | Meaning |
|---|---|
| `icd_<code>` | binary target, e.g. `icd_714` — 1 = case, 0 = control |
| `lab_<ANALYTE>_<agg>` | CBC features, e.g. `lab_HB_last`, `lab_RDW_last`, `lab_NEUTpct_last` |
| `subject_id`, `age`, `gender`/`sex` | optional, carried through |

Feature names are `lab_` + a CBC analyte code + an aggregation suffix (usually `_last`).
Get the **exact** vocabulary a checkpoint expects from
`mcp__gat-multi-scorer__get_feature_list(disease_id)` — it is authoritative, and it is not
identical across diseases (RA/714 and Hashimoto/2452 expose 15 features; the others 14).

The convention the skills assume by default is
`Data/<COHORT>/preprocessed/<icd>_<name>_<cohort>.parquet`, but any path can be passed
explicitly with `--parquet` / `--target-col`.

### 3.6 Verify the install

A 30-second check that the checkpoints load and the grammar behaves. This is a real transcript from
a clean checkout:

```python
from gat_agent_tool.registry import GatScorerRegistry
reg = GatScorerRegistry(gat_root="GAT")

sorted(reg.list_diseases())
# ['242', '2452', '250', '277', '340', '555', '556', '5790', '696', '7100', '7101', '7102', '714']

len(reg.feature_names("714"))                              # 15  (14 for most diseases)

reg.score("714", "lab_NEUT_abs_last / lab_LYMpct_last")    # 0.5324542373418808

# The rheumatoid-arthritis expression reported in the paper reproduces its published GAT score:
reg.score("714", "(lab_RDW_last - lab_RBC_last) * (lab_MCV_last + lab_RDW_last - lab_MCH_last)"
                 " / (lab_HB_last * lab_MCHC_last - lab_RDW_last)")
# 0.6327788233757019        (paper Table 4 reports 0.633)

reg.score("714", "lab_PLT_last * 0.5")                     # None — numeric literals are not in the grammar
reg.score("714", "lab_NEUT_abs / lab_LYMPH_abs")           # None — names must match the vocabulary exactly
```

Note the Python API is `reg.feature_names(icd)`; the MCP tool of the same purpose is
`get_feature_list(disease_id)`. Feature names are case-sensitive and exact — always read them from
the checkpoint rather than guessing, and expect `None` (not an exception) for anything unparseable.

---

## 4. Installing the skills for Claude Code

Claude Code discovers skills under `.claude/skills/` in the project. From the repository
root:

```bash
mkdir -p .claude/skills
ln -s ../../skills/biomarker-discovery .claude/skills/biomarker-discovery
ln -s ../../skills/biomarker-explainer .claude/skills/biomarker-explainer
```

(or `cp -R skills/biomarker-* .claude/skills/` if you prefer copies). Restart Claude Code;
`/biomarker-discovery` and `/biomarker-explainer` then appear as slash commands. Script paths inside
both `SKILL.md` files are written relative to the repository root as `skills/...`, so they resolve
whether you symlink or run the CLIs directly.

The Python scripts are ordinary CLIs and can also be run directly from `skills/…` without
installing anything — see the worked examples below.

---

## 5. Worked example — discovery

Launch Claude Code **from the repository root**, then:

```
/biomarker-discovery 714 --iterations 10 --cohort mimic
```

`714` is rheumatoid arthritis (the default if no argument is given); `--iterations 10` is a
short sanity run — the full default budget is 100 iterations × 200 expressions, which
plateau-stops in practice around iteration 8–20.

The skill will walk through its setup gate (server live, disease valid, cohort parquet
present), run the literature phase, then the scored iteration loop, writing everything to
`Results/714/biomarker_discovery/<YYYYMMDD-HHMM>/`.

The final phase is a plain script you can also run yourself against any run directory:

```bash
python skills/biomarker-discovery/scripts/finalize.py \
    --csv     Results/714/biomarker_discovery/<run>/iterative_details.csv \
    --parquet Data/MIMIC/preprocessed/714_RheumatoidArthritis_mimic.parquet \
    --target-col icd_714 \
    --baseline-dir Results/714/evaluation_results_external_v2/mimic \
    --out-dir Results/714/biomarker_discovery/<run> \
    --disease-id 714 --disease-name RheumatoidArthritis --cohort mimic \
    --n-bootstrap 500 --random-seed 42
```

**Produces**

```
Results/<icd>/biomarker_discovery/<YYYYMMDD-HHMM>/
├── literature_review.md     seeds + mechanistic rationale + citations
├── iterative_details.csv    one row per scored expression (predicted AUC + silent real AUC)
├── prompts/                 exact prompt and raw response per iteration (audit trail)
├── final_report.md          winner, bootstrap CI, before/after Δ, per-cohort table
└── final_report.json        the same, machine-readable
```

The headline number is **Δ real AUC = iteration winner − first expression**: how much the
GAT-guided iterations improved on the model's own first guess, measured on held-out data
the model never saw scores from.

---

## 6. Worked example — explainer

The route that needs no data beyond your own cohort is `--expr`:

```bash
python skills/biomarker-explainer/scripts/shap_biomarker.py \
    --expr "(RDW-RBC)*(MCV+RDW-MCH)/(HB*MCHC-RDW)" \
    --disease "Rheumatoid arthritis" \
    --parquet Data/MIMIC/preprocessed/714_RheumatoidArthritis_mimic.parquet \
    --target-col icd_714 \
    --outdir Results/714/biomarker_explainer --stem 714
```

**Produces** (under `--outdir`): `714_shap_tree.png` (the glass-box figure),
`714_findings.json` (per-feature direction, role, relative impact — the input to Stage 2)
and `714_shap_impact.csv`.

Then, inside Claude Code, the skill launches the literature workflow with
`714_findings.json` as its arguments, writes `714_explained.md`, re-renders the figure with
the verdicts overlaid (`--annotate`), and packages a one-page PDF:

```bash
python skills/biomarker-explainer/scripts/build_report_pdf.py \
    --md      Results/714/biomarker_explainer/714_explained.md \
    --figure  Results/714/biomarker_explainer/714_shap_tree_annotated.png \
    --outdir  Results/714/biomarker_explainer --stem 714
```

The `--icd <code>` shortcut resolves the paper's discovered expression for that disease from the
shipped [`data/winner_expressions.csv`](biomarker-explainer/data/winner_expressions.csv), so
`--expr` is optional; you still pass `--parquet`/`--target-col` for your own cohort (override the
table with `--iter-audit <csv>` if you have your own). Record provenance with
`--cohort-label "MIMIC-IV"`, which is written into `findings.json`. The batch driver
`run_all_biomarkers.py prep|finalize` runs the same path across all 12 labelled diseases.

---

## 7. What you must provide — summary

| Capability | Runs out of the box | Needs your cohort parquet | Not available publicly |
|---|---|---|---|
| GAT scoring via MCP (`score_expression(s)`, `get_feature_list`) | ✅ | | |
| Expression validation / grammar checking | ✅ | | |
| Discovery Phase 1 literature review | ✅ (needs harness research tools, §3.4) | | |
| Discovery Phase 2 propose–score–refine loop | ✅ | | |
| Silent real AUC, bootstrap CIs, Phase 3 report | | ✅ | |
| Shapley figure + findings JSON | | ✅ | |
| Explainer literature review + PDF | ✅ (needs `Workflow` + pandoc/TeX) | | |
| Explainer `--icd` lookup of the paper's discovered expressions | ✅ (shipped CSV) | ✅ to attribute them | |
| Head-to-head vs third-party LLM-tool baselines | | | ❌ |

Skill runs write under `Results/` in the repository root; add it to your `.gitignore` if
you work in a clone.

---

## 8. Limitations and honest caveats

- **A vacuous PASS is still a PASS.** Discovery Phase 3 compares the winner against
  third-party LLM-tool comparison expressions read from `Results/<icd>/…`. Those files are
  hand-collected outputs of external commercial services and are **not** distributed here,
  so an external run will find none and report PASS with a *"no baselines found"* note.
  Treat that verdict as **not evaluated**, not as a win. The meaningful signal in a public
  run is the before/after Δ within the run itself.
- **The predicted AUC is a prior, not a measurement.** The GAT ranks expression *structure*;
  it is a search heuristic distilled from one health system's cohort. Rank correlation
  against measured out-of-sample AUC is modest, and the strict argmax of the predicted
  score is not reliably the best expression. Always re-rank the shortlist by measured AUC
  on your own cohort before drawing any conclusion.
- **Distribution shift is real.** The weights were trained on one national EHR cohort;
  transfer to ICU-derived (MIMIC-IV) or survey (NHANES) populations is imperfect, and
  several CBC-derived signals are partly age-confounded. The explainer prints these
  guardrails on every report; do not strip them.
- **Interpretation is in-sample and univariate.** Shapley attribution uses a one-predictor
  logistic link fit on the same cohort being explained, with no covariate adjustment. It
  explains *the formula*, not the disease.
- **A feature appearing in both numerator and denominator** may be a structural artifact of
  the search rather than biology. The explainer flags this; take it seriously.
- **Bootstrap CIs are percentile**, with direction-agnostic `max(AUC, 1−AUC)` folding — no
  BCa correction. Cohorts with few positives (common on NHANES) give wide intervals.
- **Cohort caveats.** Disease definitions are ICD-derived and imperfect; NHANES cannot
  distinguish hyperthyroidism (242) from Hashimoto (2452) and uses the same
  any-thyroid-problem label for both.
- **These skills are research instruments, not clinical tools.** Nothing they produce is
  validated for diagnosis, screening or any clinical decision.

---

## 9. Licensing

The skills are **source code** and are released under the Apache License 2.0, the same as
the rest of the code in this repository ([`LICENSE`](../LICENSE)).

Running them invokes the trained weights under `GAT/`, which are released separately under
**CC BY-NC 4.0** ([`MODEL_LICENSE.md`](../MODEL_LICENSE.md)) — so outputs produced by
running these skills inherit that non-commercial restriction.

Third-party tool names referenced in comparison tables are trademarks of their respective
owners; no affiliation or endorsement is implied.

If you use these skills, please cite the companion paper:

> **Distilling private EHR evidence into a public agentic tool for CBC biomarker discovery.**
