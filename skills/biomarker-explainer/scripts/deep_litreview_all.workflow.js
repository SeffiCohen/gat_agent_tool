export const meta = {
  name: 'biomarker-deep-litreview-all',
  description: 'Run the intensive per-biomarker deep literature review for many discovered CBC biomarkers at once: fans out the validated single-biomarker workflow over every disease under one shared concurrency cap, returning an expected-vs-surprising report (with adversarially-verified citations) per disease.',
  phases: [
    { title: 'Per-biomarker review' },
  ],
}

// args = { single_workflow_path: "<abs path to deep_litreview.workflow.js>",
//          biomarkers: [ <each disease's findings.json>, ... ] }
let A = args || {}
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
const BIOS = A.biomarkers || []
const SINGLE = A.single_workflow_path
if (!BIOS.length || !SINGLE) {
  return { error: 'need args.single_workflow_path and a non-empty args.biomarkers[]' }
}

log(`reviewing ${BIOS.length} biomarkers: ${BIOS.map((b) => `${b.disease}(${b.icd})`).join(', ')}`)

// Each child is the already-validated single-biomarker review. workflow() nesting
// shares this run's concurrency cap + token budget, so all diseases' per-feature
// research/verify agents interleave under one cap instead of oversubscribing. A
// child that dies degrades to {icd, error} so one bad disease never sinks the batch.
const results = await parallel(BIOS.map((b) => () =>
  workflow({ scriptPath: SINGLE }, b)
    .then((r) => ({ icd: b.icd, disease: b.disease, expression: b.expression_pretty, ...r }))
    .catch((e) => ({ icd: b.icd, disease: b.disease, error: String(e && e.message || e) }))))

const ok = results.filter((r) => r && r.report && !r.error)
const failed = results.filter((r) => !r || r.error || !r.report)
log(`done: ${ok.length}/${BIOS.length} produced a report` +
    (failed.length ? `; failed: ${failed.map((r) => r && r.icd).join(', ')}` : ''))

return { results, n_ok: ok.length, n_failed: failed.length }
