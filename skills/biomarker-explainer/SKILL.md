---
name: biomarker-explainer
description: >-
  Explain a discovered CBC (complete-blood-count) biomarker expression for a disease: render the
  glass-box expression tree coloured by each blood test's exact Shapley impact on the disease
  prediction, then run an INTENSIVE deep literature review (via the Workflow tool) that classifies
  every component as clinically EXPECTED or SURPRISING and backs each call with adversarially-verified
  paper citations, and finally packages it into a CONCISE ONE-PAGE self-contained LaTeX (.tex) and PDF
  (the visualization + ≤2 sentences per component + references). Use this whenever the user wants to explain, interpret, justify, or "make sense of"
  a discovered biomarker / arithmetic lab expression; asks why a formula works or which features drive
  it; wants SHAP / feature-impact of a biomarker; asks which parts are expected vs surprising/novel;
  wants a literature review or references supporting (or contradicting) a discovered biomarker; or
  references the candJ / glass-box SHAP tree figure for a single disease. Trigger on phrasings like
  "explain the biomarker for <disease>", "why does this expression predict <disease>", "is RDW expected
  in RA", "literature support for this biomarker", "what's surprising about this formula", "SHAP tree
  for ICD <code>", or an explicit /biomarker-explainer. Do NOT trigger to DISCOVER or search for new
  biomarkers (that is the biomarker-discovery skill) or for one-off "score this expression" requests.
---

# Biomarker explainer

Turn one discovered CBC biomarker into (1) a publication-grade **glass-box SHAP figure** and (2) an
**evidence-graded expected-vs-surprising literature report**. The point is interpretation, not
discovery: you are handed a formula and you explain *what each blood test contributes and whether the
science agrees*.

The two deliverables come from two stages that you run in order. Stage 1 is deterministic Python.
Stage 2 is a deep, fan-out **Workflow** — this skill is an explicit opt-in to use the Workflow tool, so
launch it as described; do not approximate it with a few inline searches.

## What "a biomarker" means here

A depth-≤3 arithmetic expression over CBC features (operators `+ − × ÷`, no numeric literals), e.g. RA:
`(RDW−RBC)·(MCV+RDW−MCH) / (HB·MCHC−RDW)`. The 13 manuscript diseases live in
`skills/biomarker-explainer/data/winner_expressions.csv` (`disease_folder` → `winner_expression`); the
labelled external cohorts are the MIMIC-IV parquets in `Data/MIMIC/preprocessed/`.

## Stage 1 — Shapley impact figure + structured findings

Resolve which biomarker the user means, then run the bundled script. It computes **exact interventional
Shapley** values (the whole expression is fit as a one-predictor logistic disease model
`P=σ(β₀+β₁·standardised E)`, and that prediction is attributed back to each *raw* feature using the
cohort median as the reference patient — so a feature's sign/magnitude reflect how it is actually used
inside the formula, numerator vs denominator and orientation included).

```bash
# By disease code (looks up the expression + MIMIC cohort automatically):
python skills/biomarker-explainer/scripts/shap_biomarker.py \
    --icd 714 --outdir Results/714/biomarker_explainer

# Or an arbitrary expression on any labelled cohort:
python skills/biomarker-explainer/scripts/shap_biomarker.py \
    --expr "(RDW-RBC)*(MCV+RDW-MCH)/(HB*MCHC-RDW)" --disease "Rheumatoid arthritis" \
    --parquet Data/MIMIC/preprocessed/714_RheumatoidArthritis_mimic.parquet --target-col icd_714 \
    --outdir Results/714/biomarker_explainer
```

Outputs (under `--outdir`, stem = `--icd` or `--stem`):
- `<stem>_shap_tree.png` — the glass-box tree, leaves coloured red↑ (raises predicted risk) → blue↓
  (lowers it), scaled to the strongest driver in **this** formula, direction triangle on each chip.
- `<stem>_findings.json` — the per-feature directions, roles, magnitudes — **this is the input to Stage 2.**
- `<stem>_shap_impact.csv` — same, tabular.

Show the user `<stem>_shap_tree.png` before moving on. If the disease has no MIMIC labels (e.g. FMF/277)
or the user gives a brand-new expression with no cohort, pass `--parquet`/`--target-col`; if truly no
labelled cohort exists, Shapley can't run — say so and proceed to Stage 2 using the structure (the
feature list + numerator/denominator roles) instead of SHAP directions.

## Stage 2 — intensive deep literature review (Workflow)

Read `<stem>_findings.json`, then launch the bundled workflow, passing the findings as `args`. The
workflow fans out: deep multi-source research per feature → adversarial citation verification → a
composite/ratio surprise hunt → synthesis. It is deliberately thorough; let it run.

```
Workflow({
  scriptPath: "skills/biomarker-explainer/scripts/deep_litreview.workflow.js",
  args: <the parsed contents of <stem>_findings.json>
})
```

(Use the script's absolute path if the skill is installed outside this repo. Pass `args` as a real JSON
object — the parsed findings — not a stringified blob.)

The workflow's agents reach PubMed / Consensus / bioRxiv / web search through `ToolSearch`; their job
is to decide, per component, whether the **model's direction is concordant or discordant with the
established clinical literature**:
- **Expected** — the SHAP direction matches the known direction of that blood test in the disease
  (e.g. RA: RDW↑ raises risk — RDW is an established inflammation/anaemia marker elevated in RA; HB↑
  lowers risk — anaemia of chronic disease).
- **Surprising** — the SHAP direction contradicts the established direction, **or** the feature has no
  recognized link to the disease yet carries non-trivial impact (a hypothesis-generating signal).
- **Unclear** — evidence too thin or conflicting.

The verification phase exists because fabricated citations are the worst failure mode: it re-looks-up
every PMID/DOI and drops anything it can't confirm. Trust only verified citations downstream.

## Write the report and present

The workflow returns `{ components, composites, report }`. `report` is a **concise one-page** GitHub-flavoured
Markdown explainer — the biomarker formula, then ≤2 sentences per component (direction + Expected/Surprising
verdict + one-line mechanism + a bracketed citation), a one-line caveat, and a short numbered reference list.
No summary table, no long narrative sections — it must fit a single page under the figure. Write it next to
the figure and present both:

```bash
# write the returned report to:
Results/<icd>/biomarker_explainer/<stem>_explained.md
```

Then re-render the figure with the literature verdicts overlaid (a small ● = expected, ★ = surprising
marker per chip) so the figure and report agree at a glance:

```bash
# build {"RDW":"expected","HB":"expected","MCV":"surprising",...} from the verified verdicts → annot.json
python skills/biomarker-explainer/scripts/shap_biomarker.py \
    --icd 714 --outdir Results/714/biomarker_explainer --annotate Results/714/biomarker_explainer/annot.json
# → also writes <stem>_shap_tree_annotated.png
```

## Build the LaTeX + PDF

Always finish by packaging everything into a self-contained **one-page `.tex` and `.pdf`** — the figure,
the ≤2-sentence-per-component explanation, and the verified references in one shareable page. Use the
annotated figure when it exists. The compact defaults (`--figwidth 38 --margin 0.75 --fontsize 9`, a
raised title block, tight `parskip`) keep it to one page for up to ~8 components:

```bash
python skills/biomarker-explainer/scripts/build_report_pdf.py \
    --md Results/714/biomarker_explainer/714_explained.md \
    --figure Results/714/biomarker_explainer/714_shap_tree_annotated.png \
    --outdir Results/714/biomarker_explainer --stem 714
# → 714_explained.tex  and  714_explained.pdf
```

It sanitises report unicode to robust LaTeX (`\ensuremath{...}` / `\textsuperscript{...}` so adjacent
symbols like ×10⁶/µL never collide into stray `$$`), embeds the figure under the title via pandoc, and
compiles with latexmk. Needs pandoc + a LaTeX engine (pdflatex by default; `--engine xelatex` works
too). If the build fails, report it — do not hand-doctor the `.tex`; fix the sanitiser instead.

In your chat reply, give the user: the figure path, the report path, the **PDF path**, and a tight summary — the
top 1–2 expected drivers and the top 1–2 surprising findings, each with its key citation. Keep the
honesty guardrails visible: in-sample logistic link, discovery on Clalit vs MIMIC validation, modest
out-of-sample AUC, age confounding for several CBC features, and that a feature recurring in
numerator+denominator may be an overfit artifact rather than biology.

## Run all biomarkers (batch)

When the user wants every discovered biomarker explained at once, don't loop the single path 12×
by hand — use the batch driver, which runs one master Workflow that fans the validated single-biomarker
review over all diseases under one shared concurrency cap (≈16 agents), then packages a per-disease PDF
plus a merged compendium. Three steps:

```bash
# 1) prep: compute SHAP figure + findings.json for the 12 labelled diseases, assemble the Workflow args
python skills/biomarker-explainer/scripts/run_all_biomarkers.py prep
#    → Results/_aggregate/biomarker_explainer/all_findings.json   (277/FMF is auto-skipped: 0 MIMIC cases)
```
```
# 2) launch the master Workflow with that file's contents as args:
Workflow({ scriptPath: ".../scripts/deep_litreview_all.workflow.js",
           args: <parsed contents of all_findings.json> })
#    args is ~13 KB; trim each feature to {feature,pretty,role,shap_direction,rel_impact,
#    standalone_signed_corr} if you want a smaller call. Wait for the completion notification
#    (~190 subagents, ~12–15M tokens, ~15–30 min). Then save the result's `result` object to
#    Results/_aggregate/biomarker_explainer/workflow_result.json   (the .output file nests it under `result`).
```
```bash
# 3) resolve every cited PMID against PubMed  ← REQUIRED before finalize
python skills/biomarker-explainer/scripts/fetch_pubmed_records.py \
    --workflow-output Results/_aggregate/biomarker_explainer/workflow_result.json \
    --bib Results/_aggregate/biomarker_explainer/biomarker_explainer_refs.bib
#    → data/pubmed_records.json (the cache every reference is rendered from) + a BibTeX file.
#    Any PMID it cannot resolve, and anything retracted, is printed — chase those before shipping.

# 4) finalize: per-disease MD + annotated figure + PDF, then a merged compendium + summary CSV
python skills/biomarker-explainer/scripts/run_all_biomarkers.py \
    finalize --workflow-output Results/_aggregate/biomarker_explainer/workflow_result.json
#    add --reuse-figures to rebuild the reports without re-running SHAP (no cohort access needed)

# 5) OPTIONAL: one unified LaTeX source for the whole supplement, instead of merged PDFs
python skills/biomarker-explainer/scripts/build_supplement_tex.py \
    --workflow-output Results/_aggregate/biomarker_explainer/workflow_result.json \
    --outdir <dir> --compile
```

Step 4 merges independently compiled per-disease PDFs with `pdfunite`, so there is no single
editable source for the compendium. Step 5 builds one: a standalone article with a page per
disease, relative figure paths, and a single shared bibliography driven by `\cite` keys that
`citations.assign_keys` also uses for the `.bib`, so a key can never drift between a citation and
the entry it points at. Use it when the supplement has to be edited by hand or uploaded to
Overleaf; the merged-PDF path remains fine when it does not.

**Never render a reference from the workflow's own citation dicts.** The research pass returns an
author string but usually no PMID, while the verification pass returns the PMID but no authors, so
joining them on PMID silently produces authorless entries — and any renderer that then falls back to
the title's first word emits fabrications like "Hematologic et al.". Step 3 exists to make that
impossible: `citations.py` renders both the reference lists and the BibTeX from the PubMed cache, and
anything missing from it is marked `[unverified]` rather than guessed.

Batch outputs land in `Results/_aggregate/biomarker_explainer/`: `all_biomarkers_explained.pdf` (cover +
all per-disease reports), `expected_surprising_summary.csv` (every component's verdict + key citation),
and `all_findings.json`. Per-disease artifacts stay in `Results/<icd>/biomarker_explainer/`. Report any
diseases whose child review returned `{error}` rather than silently dropping them.

## Scope notes

- Default disease when none is given: **714 (rheumatoid arthritis)** on MIMIC, the worked example.
- This skill *explains* a given biomarker. To *find* new ones, use `biomarker-discovery`.
- Everything is aggregate/out-of-sample — no patient rows leave the cohort; figure + report are shareable.
