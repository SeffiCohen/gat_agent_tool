export const meta = {
  name: 'biomarker-deep-litreview',
  description: 'Intensive deep literature review of a discovered CBC biomarker: per-feature expected-vs-surprising classification against the clinical literature, with adversarially-verified citations, plus a composite/ratio surprise hunt and a synthesized report.',
  phases: [
    { title: 'Component research' },
    { title: 'Adversarial verify' },
    { title: 'Composite & surprise hunt' },
    { title: 'Synthesis' },
  ],
}

// ---- inputs (from shap_biomarker.py findings.json, passed as Workflow args) ----
let A = args || {}
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
if (A && typeof A.findings === 'object') A = A.findings   // accept {findings:{...}} too
const disease = A.disease || 'the target disease'
const icd = A.icd || ''
const exprP = A.expression_pretty || A.expression_raw || ''
const wholeAuc = A.whole_auc_mimic
const features = A.features || []

// Cohort provenance, written by shap_biomarker.py as
// "<validation> (out-of-sample vs <discovery> discovery cohort)". Naming the real cohorts keeps
// the caveat truthful when the skill runs on data other than this project's.
const cohortM = /^\s*([^(]+?)\s*\(\s*out-of-sample vs\s+(.+?)\s+discovery cohort\s*\)\s*$/i
  .exec(A.cohort || '')
const validationCohort = cohortM ? cohortM[1].trim() : (A.cohort || '').trim()
const discoveryCohort = cohortM ? cohortM[2].trim() : ''
const caveatLine = 'Caveats: '
  + (validationCohort
      ? 'the logistic link is fit on ' + validationCohort + ' (in-sample)'
      : 'the logistic link is fit in-sample on the cohort it explains')
  + (discoveryCohort ? '; the expression was discovered on ' + discoveryCohort : '')
  + '; age confounds several red-cell indices'
  + (wholeAuc ? '; the out-of-sample AUC is modest (~' + wholeAuc + ')' : '')
  + '; a feature recurring in numerator+denominator may be an overfit artifact rather than biology.'
if (!features.length) {
  return { error: 'no args.features — run shap_biomarker.py first and pass its findings.json' }
}

const TOOLS_NOTE = [
  'LITERATURE ACCESS — you MUST ground every claim in real, retrievable sources.',
  'Load tools with ToolSearch before using them, e.g.:',
  '  ToolSearch "select:mcp__plugin_bio-research_pubmed__search_articles,mcp__plugin_bio-research_pubmed__get_article_metadata,mcp__plugin_bio-research_pubmed__get_full_text_article,mcp__plugin_bio-research_pubmed__find_related_articles"',
  '  ToolSearch "select:mcp__plugin_bio-research_consensus__search"',
  '  ToolSearch "select:WebSearch,WebFetch"',
  '  ToolSearch "biorxiv medrxiv preprint clinical trials" (keyword search for more sources)',
  'Run SEVERAL query angles (disease + feature, disease + mechanism, the composite index name, recent reviews).',
  'Prefer systematic reviews, meta-analyses, and large cohort studies over single small studies.',
  'ONLY cite papers you actually retrieved through a tool: copy the real PMID / DOI / title / year straight from the tool result. NEVER invent or guess a citation, PMID, or DOI — fabricated references are the single worst failure mode here and they will be caught and dropped in verification.',
].join('\n')

const CITATION = {
  type: 'object',
  properties: {
    title: { type: 'string' }, authors: { type: 'string' }, year: { type: 'integer' },
    journal: { type: 'string' }, pmid: { type: 'string' }, doi: { type: 'string' }, url: { type: 'string' },
    finding: { type: 'string', description: 'one sentence: what this paper actually showed, relevant to this feature+disease' },
    retrieved_via: { type: 'string', description: 'which tool returned it (e.g. pubmed.search_articles)' },
  },
  required: ['title', 'year', 'finding'],
}

const COMPONENT_SCHEMA = {
  type: 'object',
  properties: {
    feature: { type: 'string' },
    established_direction: { type: 'string', enum: ['elevated_in_disease', 'reduced_in_disease', 'biphasic_or_mixed', 'no_consistent_association', 'unknown'] },
    evidence_strength: { type: 'string', enum: ['strong', 'moderate', 'weak', 'none'] },
    mechanism: { type: 'string', description: 'why the feature moves in that direction in this disease (pathophysiology)' },
    shap_vs_literature: { type: 'string', enum: ['concordant', 'discordant', 'no_literature_basis'], description: 'does the SHAP direction (given to you) agree with the established clinical direction?' },
    classification: { type: 'string', enum: ['expected', 'surprising', 'unclear'] },
    rationale: { type: 'string' },
    citations: { type: 'array', items: CITATION },
  },
  required: ['feature', 'established_direction', 'evidence_strength', 'shap_vs_literature', 'classification', 'rationale', 'citations'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    verified_citations: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' }, year: { type: 'integer' }, pmid: { type: 'string' }, doi: { type: 'string' }, url: { type: 'string' },
          exists: { type: 'boolean', description: 'did you confirm this paper is real via a tool lookup?' },
          supports_claim: { type: 'boolean', description: 'does it actually support the stated finding/direction?' },
          note: { type: 'string' },
        },
        required: ['title', 'exists', 'supports_claim'],
      },
    },
    contradicting_evidence: { type: 'array', items: CITATION },
    final_classification: { type: 'string', enum: ['expected', 'surprising', 'unclear'] },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    verdict_note: { type: 'string', description: 'why you upheld or changed the classification after scrutiny' },
  },
  required: ['verified_citations', 'final_classification', 'confidence', 'verdict_note'],
}

const COMPOSITE_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    observations: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          composite_observed: { type: 'string', description: 'the structural pattern in the formula (e.g. RDW in numerator AND denominator; RDW/HB ratio; RDW used 3x)' },
          known_index: { type: 'string', description: 'the established clinical index it resembles, if any (NLR, PLR, RPR, MLR, SII, RDW/RBC ...) or "none known"' },
          literature_finding: { type: 'string' },
          classification: { type: 'string', enum: ['expected', 'surprising', 'unclear'] },
          citations: { type: 'array', items: CITATION },
        },
        required: ['composite_observed', 'classification'],
      },
    },
  },
  required: ['lens', 'observations'],
}

// ---- phase 1: deep per-feature research -------------------------------------
const researchPrompt = (f) => [
  `You are a hematology / clinical-pathology evidence analyst. Disease: ${disease} (ICD ${icd}).`,
  `It was found that a discovered CBC biomarker uses the blood-count feature ${f.pretty} (${f.feature}).`,
  `Inside the discovered formula  ${exprP}  this feature sits in the ${f.role}.`,
  `The model attribution (exact Shapley on out-of-sample MIMIC, whole-biomarker AUC≈${wholeAuc}) says:`,
  `  → ${f.shap_direction} (relative impact ${f.rel_impact}; its standalone signed rank-biserial in the same cohort is ${f.standalone_signed_corr}).`,
  '',
  `TASK: establish, from the clinical literature, the KNOWN association between ${f.pretty} and ${disease}:`,
  `  • Is ${f.pretty} typically elevated, reduced, mixed, or not consistently associated in ${disease}? With what mechanism?`,
  `  • How strong is the evidence (systematic reviews / meta-analyses / large cohorts vs anecdotal)?`,
  `Then decide whether the model's direction (${f.shap_direction}) is EXPECTED or SURPRISING:`,
  `  - "expected": the model direction matches the established clinical direction of ${f.pretty} in ${disease}.`,
  `  - "surprising": the model direction contradicts the established direction, OR ${f.pretty} has no recognized link to ${disease} yet carries non-trivial model impact (a hypothesis-generating signal).`,
  `  - "unclear": evidence too thin/conflicting to judge.`,
  `Capture 2–5 of the STRONGEST real citations that ground your call (with PMIDs/DOIs from the tools).`,
  '',
  TOOLS_NOTE,
  '',
  'Return ONLY the structured object. Be specific and honest; do not overstate weak evidence.',
].join('\n')

const verifyPrompt = (r, f) => [
  `Adversarially audit this literature finding for ${f.pretty} in ${disease}. Your job is to catch fabricated or mis-cited references and over-claims.`,
  '',
  `CLAIM UNDER REVIEW:\n${JSON.stringify(r, null, 2)}`,
  '',
  'Do this:',
  '1. For EACH cited paper, look it up with the literature tools (ToolSearch the pubmed tools / WebSearch). Confirm it EXISTS (real title+year, resolvable PMID/DOI) and that it ACTUALLY supports the stated finding/direction. Mark exists + supports_claim honestly; a plausible-sounding but unverifiable citation is exists=false.',
  '2. Actively search for CONTRADICTING evidence — papers showing the opposite direction. Record any you find.',
  '3. Re-decide the final classification (expected / surprising / unclear) after this scrutiny, and your confidence. If the original call relied on citations you could not verify, lower confidence or change the verdict and say so.',
  '',
  TOOLS_NOTE,
  '',
  'Default to skepticism: if you cannot verify a citation, do not pass it. Return ONLY the structured object.',
].join('\n')

const components = (await pipeline(
  features,
  (f) => agent(researchPrompt(f), { phase: 'Component research', label: `res:${f.pretty}`, schema: COMPONENT_SCHEMA, effort: 'high' }),
  (r, f) => (r
    ? agent(verifyPrompt(r, f), { phase: 'Adversarial verify', label: `vfy:${f.pretty}`, schema: VERIFY_SCHEMA, effort: 'high' })
      .then((v) => ({ feature: f.feature, pretty: f.pretty, role: f.role, shap_direction: f.shap_direction, rel_impact: f.rel_impact, research: r, verify: v }))
    : null),
)).filter(Boolean)

// ---- phase 2: composite / ratio surprise hunt (whole-expression level) -------
const LENSES = [
  { key: 'ratios_denominator', focus: 'What does the NUMERATOR/DENOMINATOR split encode physiologically? Search literature on the implied ratios (e.g. a red-cell-distribution term over a haemoglobin/MCHC term ≈ anaemia-of-inflammation contrast).' },
  { key: 'known_cbc_indices', focus: 'Map the structure onto ESTABLISHED CBC-derived composite indices and search each in this disease: neutrophil-to-lymphocyte ratio (NLR), platelet-to-lymphocyte ratio (PLR), monocyte-to-lymphocyte ratio (MLR), RDW-to-platelet ratio (RPR), RDW/RBC, systemic immune-inflammation index (SII).' },
  { key: 'repeats_and_products', focus: 'Which features RECUR or MULTIPLY (e.g. a feature appearing in both numerator and denominator, or squared), and is such an interaction biologically meaningful or likely an overfit artifact? Search for evidence either way.' },
]
const compPrompt = (L) => [
  `Disease: ${disease} (ICD ${icd}). Discovered CBC biomarker:  ${exprP}`,
  `Per-feature model directions: ${JSON.stringify(features.map((f) => ({ f: f.pretty, role: f.role, dir: f.shap_direction, rel: f.rel_impact })))}`,
  '',
  `LENS — ${L.key}: ${L.focus}`,
  '',
  'For each structural observation under this lens, say whether the literature makes it EXPECTED (a known/plausible composite signal in this disease) or SURPRISING (novel or counter to known indices), with real citations. If a pattern looks like an overfit artifact rather than biology, say so plainly.',
  '',
  TOOLS_NOTE,
  '',
  'Return ONLY the structured object.',
].join('\n')

const composites = (await parallel(
  LENSES.map((L) => () => agent(compPrompt(L), { phase: 'Composite & surprise hunt', label: `comp:${L.key}`, schema: COMPOSITE_SCHEMA, effort: 'high' })),
)).filter(Boolean)

// ---- phase 3: synthesis into a report ---------------------------------------
const onlyVerified = components.map((c) => ({
  feature: c.pretty, role: c.role, shap_direction: c.shap_direction, rel_impact: c.rel_impact,
  established_direction: c.research.established_direction, evidence_strength: c.research.evidence_strength,
  mechanism: c.research.mechanism, classification: c.verify.final_classification, confidence: c.verify.confidence,
  rationale: c.research.rationale, verdict_note: c.verify.verdict_note,
  citations: (c.verify.verified_citations || []).filter((x) => x.exists && x.supports_claim),
  contradicting: c.verify.contradicting_evidence || [],
}))

const synthPrompt = [
  `Write a rigorous, publication-grade explainer of the discovered CBC biomarker for ${disease} (ICD ${icd}).`,
  `Biomarker:  ${exprP}   (whole-biomarker out-of-sample AUC on MIMIC ≈ ${wholeAuc}).`,
  '',
  'PER-FEATURE EVIDENCE (use ONLY the verified citations listed here — do not add new references; if a feature has zero verified citations, say the evidence is unverified rather than inventing any):',
  JSON.stringify(onlyVerified, null, 2),
  '',
  'COMPOSITE/STRUCTURE EVIDENCE:',
  JSON.stringify(composites, null, 2),
  '',
  'Produce a SUPER-CONCISE one-page explainer in GitHub-flavoured Markdown, EXACTLY this shape:',
  '',
  '# Biomarker explained: <disease> (ICD <icd>)',
  '',
  '**Biomarker:** <formula>  ·  out-of-sample AUC (MIMIC) ≈ <auc>',
  '',
  'Then ONE short paragraph per feature (ordered by |rel_impact|, largest first). Each paragraph is AT MOST TWO sentences: begin with the bold feature name and its role, e.g. "**RDW** (numerator+denominator).", say whether higher levels raise or lower the predicted risk and give the Expected / Surprising / Unclear verdict with confidence, then the one-line mechanism, and end with a single bracketed citation number like [1]. Never exceed two sentences for a feature.',
  '',
  'End with this exact italic line, reproduced verbatim: *' + caveatLine + '*',
  '',
  '## References',
  'A numbered list of ONLY the citations you referenced by [n] (First-author et al. Short title. Year. PMID). One key reference per feature is enough.',
  '',
  'HARD CONSTRAINTS: the whole document must fit on ONE page — NO summary table, NO section headings other than "## References", NO per-feature "##" subheadings, at most two sentences per feature. Use ONLY the verified citations above; never invent one. Return ONLY the Markdown.',
].join('\n')

const report = await agent(synthPrompt, { phase: 'Synthesis', label: 'synthesis', effort: 'high' })

return { disease, icd, expression: exprP, whole_auc_mimic: wholeAuc, components, composites, report }
