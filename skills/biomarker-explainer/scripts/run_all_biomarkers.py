#!/usr/bin/env python3
r"""run_all_biomarkers.py — batch driver for the biomarker-explainer skill.

Two stages around one master Workflow (deep_litreview_all.workflow.js):

  prep      For every target ICD, run shap_biomarker.py (figure + findings.json +
            csv), then assemble Results/_aggregate/biomarker_explainer/all_findings.json
            = { single_workflow_path, biomarkers:[<each findings.json>] }  -> the
            args you hand to the master Workflow.

  finalize  Given the master Workflow's result JSON, for every disease that produced
            a report: write <icd>_explained.md, derive the literature verdicts into
            annot.json, re-render the verdict-annotated figure, build the per-disease
            PDF (build_report_pdf.py), then pdfunite a cover + all per-disease PDFs
            into all_biomarkers_explained.pdf and emit expected_surprising_summary.csv.

Reuses shap_biomarker.py / build_report_pdf.py as subprocesses — no duplicated logic.

Usage:
  python run_all_biomarkers.py prep
  # ... launch Workflow(scriptPath=deep_litreview_all.workflow.js, args=<all_findings.json>) ...
  # ... save its `result` object to workflow_result.json ...
  python run_all_biomarkers.py finalize --workflow-output <.../workflow_result.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parent
SHAP = SCRIPTS / "shap_biomarker.py"
PDF = SCRIPTS / "build_report_pdf.py"
SINGLE_WF = SCRIPTS / "deep_litreview.workflow.js"

# clinically-grouped order (matches the candJ panel); 277/FMF excluded (no MIMIC cases)
DEFAULT_ICDS = ["7100", "714", "2452", "242", "696", "7102", "7101", "556", "555", "340", "250", "5790"]


def find_repo() -> Path:
    for up in [Path.cwd(), *Path.cwd().parents, HERE, *HERE.parents]:
        if (up / "Code" / "_external_scoring.py").exists():
            return up
    raise SystemExit("Could not locate the repository root (no Code/_external_scoring.py found above this script); set GAT_AGENT_TOOL_REPO to the repository root.")


def agg_dir(repo: Path) -> Path:
    d = repo / "Results" / "_aggregate" / "biomarker_explainer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def disease_dir(repo: Path, icd: str) -> Path:
    d = repo / "Results" / icd / "biomarker_explainer"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- prep
def prep(icds, repo):
    bios, ok, fail = [], [], []
    for icd in icds:
        out = disease_dir(repo, icd)
        print(f"[prep] {icd} ...", flush=True)
        r = subprocess.run([sys.executable, str(SHAP), "--icd", icd, "--outdir", str(out)],
                           capture_output=True, text=True)
        fjson = out / f"{icd}_findings.json"
        if r.returncode != 0 or not fjson.exists():
            print(f"[prep] SKIP {icd}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no findings'}")
            fail.append(icd)
            continue
        bios.append(json.load(open(fjson)))
        ok.append(icd)
    payload = {"single_workflow_path": str(SINGLE_WF), "biomarkers": bios}
    af = agg_dir(repo) / "all_findings.json"
    json.dump(payload, open(af, "w"), indent=2)
    print(f"\n[prep] {len(ok)} ok ({', '.join(ok)})" + (f"; {len(fail)} skipped ({', '.join(fail)})" if fail else ""))
    print(f"[prep] wrote {af}")
    print(f"[prep] -> launch Workflow(scriptPath='{SINGLE_WF.parent / 'deep_litreview_all.workflow.js'}', args=<contents of {af}>)")


# ---------------------------------------------------------------- finalize helpers
def clean_report(md):
    """Drop any LLM preamble before the first H1 ('I'll write this directly...').
    The synthesis template always opens with '# Biomarker explained:'."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("# ") and not ln.startswith("## "):
            return "\n".join(lines[i:])
    return md


_ESTPHRASE = {
    "elevated_in_disease": "elevated in", "reduced_in_disease": "reduced in",
    "biphasic_or_mixed": "variably altered in", "no_consistent_association": "not consistently linked to",
    "unknown": "of unclear direction in",
}


def _first_sentence(txt, maxlen=210):
    import re
    txt = (txt or "").strip()
    parts = re.split(r"(?<=[.!?])\s+", txt)
    s = (parts[0] if parts else txt).strip()
    if len(s) > maxlen:
        s = s[:maxlen].rsplit(" ", 1)[0].rstrip(",;: ") + "..."
    return s


def _first_author(cite, authmap):
    pmid = cite.get("pmid") or ""
    auth = authmap.get(pmid) or cite.get("authors") or ""
    if auth:
        first = auth.split(",")[0].strip().split(" ")[0]
    else:
        first = (cite.get("title") or "Ref").split(" ")[0]
    return first


def build_concise_report(disease, icd, expr, auc, components):
    """Deterministic one-page explainer from the verified, cached findings:
    formula + <=2 sentences per component (direction + verdict + mechanism) + numbered refs."""
    refs, ref_idx = [], {}   # pmid -> number
    # global pmid -> authors (verified_citations drop the author list; research.citations keep it)
    gauth = {}
    for c in components:
        for x in ((c.get("research") or {}).get("citations") or []):
            if x.get("pmid") and x.get("authors"):
                gauth[x["pmid"]] = x["authors"]

    def cite_num(cite):
        key = cite.get("pmid") or cite.get("doi") or cite.get("title")
        if key not in ref_idx:
            ref_idx[key] = len(refs) + 1
            refs.append(cite)
        return ref_idx[key]

    lines = [f"# Biomarker explained: {disease} (ICD {icd})", "",
             f"**Biomarker:** {expr}  ·  out-of-sample AUC (MIMIC) ≈ {auc}", ""]
    for c in components:
        pretty = c.get("pretty", c.get("feature", "?"))
        role = c.get("role", "")
        rdir = "raises" if "RAIS" in (c.get("shap_direction", "").upper()) else "lowers"
        research = c.get("research") or {}
        verify = c.get("verify") or {}
        verdict = (verify.get("final_classification") or research.get("classification") or "unclear").lower()
        conf = verify.get("confidence") or research.get("evidence_strength") or ""
        est = _ESTPHRASE.get(research.get("established_direction", "unknown"), "of unclear direction in")
        if verdict == "expected":
            s1 = f"Higher {pretty} {rdir} the predicted risk — **expected**, consistent with {pretty} being {est} {disease}"
        elif verdict == "surprising":
            s1 = f"Higher {pretty} {rdir} the predicted risk — **surprising**, since {pretty} is typically {est} {disease}"
        else:
            s1 = f"Higher {pretty} {rdir} the predicted risk — **unclear**, as the literature on {pretty} in {disease} is limited"
        if conf:
            s1 += f" ({conf} confidence)."
        else:
            s1 += "."
        s2 = _first_sentence(research.get("mechanism") or research.get("rationale") or "", maxlen=112)
        authmap = {x.get("pmid"): x.get("authors") for x in (research.get("citations") or []) if x.get("pmid")}
        vcs = [x for x in (verify.get("verified_citations") or []) if x.get("exists") and x.get("supports_claim")]
        cites = vcs or (research.get("citations") or [])
        tag = f" [{cite_num(cites[0])}]" if cites else " (no verified citation)"
        line = f"**{pretty}** ({role}). {s1} {s2}{tag}".replace("  ", " ")
        lines.append(line)
        lines.append("")
    lines.append(f"*Caveats: the logistic link is fit on MIMIC (in-sample); the expression was discovered on "
                 f"Clalit; age confounds several red-cell indices; the out-of-sample AUC is modest "
                 f"(≈{auc}); and a feature recurring in numerator+denominator may be an overfit "
                 f"artifact rather than biology.*")
    lines.append("")
    lines.append("## References")
    for i, cite in enumerate(refs, 1):
        auth = _first_author(cite, gauth)
        title = (cite.get("title") or "").strip().rstrip(".")
        if len(title) > 70:
            title = title[:67].rstrip(",;: ") + "..."
        yr = cite.get("year") or ""
        pmid = cite.get("pmid") or ""
        doi = cite.get("doi") or ""
        tail = f"PMID {pmid}" if pmid else (doi or "")
        lines.append(f"{i}. {auth} et al. {title}. {yr}. {tail}.")
    return "\n".join(lines)


def best_citation(comp):
    """Return a short 'Title (Year, PMID)' from the first VERIFIED citation, else ''."""
    vr = (comp.get("verify") or {}).get("verified_citations") or []
    vr = [c for c in vr if c.get("exists") and c.get("supports_claim")]
    cand = vr[0] if vr else None
    if not cand:
        rc = (comp.get("research") or {}).get("citations") or []
        cand = rc[0] if rc else None
    if not cand:
        return ""
    title = (cand.get("title") or "").strip().rstrip(".")
    if len(title) > 60:
        title = title[:57] + "..."
    yr = cand.get("year") or ""
    pmid = cand.get("pmid") or cand.get("doi") or ""
    tail = ", ".join(str(x) for x in (yr, f"PMID {pmid}" if cand.get("pmid") else pmid) if x)
    return f"{title} ({tail})" if tail else title


def finalize(workflow_output, repo):
    res = json.load(open(workflow_output))
    results = res.get("results") if isinstance(res, dict) else res
    if results is None:
        sys.exit(f"no .results in {workflow_output}")
    findings_by_icd = {}
    af = agg_dir(repo) / "all_findings.json"
    if af.exists():
        for b in json.load(open(af)).get("biomarkers", []):
            findings_by_icd[b["icd"]] = b

    summary_rows, pdfs, done, skipped = [], {}, [], []
    for r in results:
        icd = str(r.get("icd", "?"))
        if not r.get("report") or r.get("error"):
            print(f"[finalize] SKIP {icd}: {r.get('error', 'no report')}")
            skipped.append(icd)
            continue
        out = disease_dir(repo, icd)
        disease = r.get("disease", findings_by_icd.get(icd, {}).get("disease", icd))
        expr = r.get("expression") or findings_by_icd.get(icd, {}).get("expression_pretty", "")
        auc = r.get("whole_auc_mimic", findings_by_icd.get(icd, {}).get("whole_auc_mimic", ""))
        # 1) concise one-page report, deterministically assembled from the verified findings
        (out / f"{icd}_explained.md").write_text(
            build_concise_report(disease, icd, expr, auc, r.get("components") or []))
        # 2) verdict annotations + per-feature summary rows
        annot = {}
        comps = r.get("components") or []
        fmeta = {f["feature"]: f for f in findings_by_icd.get(icd, {}).get("features", [])}
        for c in comps:
            feat = c.get("feature") or c.get("pretty")
            verdict = (c.get("verify") or {}).get("final_classification")
            if feat and verdict:
                annot[feat] = verdict
            fm = fmeta.get(feat, {})
            summary_rows.append({
                "icd": icd, "disease": disease, "feature": c.get("pretty", feat),
                "role": fm.get("role", c.get("role", "")),
                "model_direction": fm.get("shap_direction", c.get("shap_direction", "")),
                "rel_impact": fm.get("rel_impact", c.get("rel_impact", "")),
                "verdict": verdict or "",
                "confidence": (c.get("verify") or {}).get("confidence", ""),
                "key_citation": best_citation(c),
            })
        json.dump(annot, open(out / "annot.json", "w"), indent=2)
        # 3) annotated figure (re-render with verdict markers)
        subprocess.run([sys.executable, str(SHAP), "--icd", icd, "--outdir", str(out),
                        "--annotate", str(out / "annot.json")],
                       capture_output=True, text=True)
        fig = out / f"{icd}_shap_tree_annotated.png"
        if not fig.exists():
            fig = out / f"{icd}_shap_tree.png"
        # 4) per-disease PDF
        p = subprocess.run([sys.executable, str(PDF), "--md", str(out / f"{icd}_explained.md"),
                            "--figure", str(fig), "--outdir", str(out), "--stem", icd],
                           capture_output=True, text=True)
        pdf = out / f"{icd}_explained.pdf"
        if p.returncode == 0 and pdf.exists():
            pdfs[icd] = pdf
            done.append(icd)
            print(f"[finalize] {icd} {disease}: report + figure + PDF ok")
        else:
            print(f"[finalize] {icd}: PDF FAILED: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else '?'}")

    # summary CSV
    agg = agg_dir(repo)
    with open(agg / "expected_surprising_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["icd", "disease", "feature", "role", "model_direction",
                                           "rel_impact", "verdict", "confidence", "key_citation"])
        w.writeheader()
        w.writerows(summary_rows)

    # cover page + merged compendium
    _build_compendium(repo, agg, pdfs, summary_rows, findings_by_icd)

    print(f"\n[finalize] {len(done)} PDFs built ({', '.join(done)})" +
          (f"; skipped {', '.join(skipped)}" if skipped else ""))
    print(f"[finalize] summary -> {agg / 'expected_surprising_summary.csv'}")


def _build_compendium(repo, agg, pdfs, summary_rows, findings_by_icd):
    if not pdfs:
        print("[finalize] no per-disease PDFs — skipping compendium")
        return
    import shutil
    import datetime as _dt
    # per-disease verdict counts
    counts = {}
    for row in summary_rows:
        c = counts.setdefault(row["icd"], {"disease": row["disease"], "exp": 0, "sur": 0, "unc": 0})
        v = (row["verdict"] or "").lower()
        c["exp"] += v == "expected"
        c["sur"] += v == "surprising"
        c["unc"] += v == "unclear"
    order = [i for i in DEFAULT_ICDS if i in pdfs] + [i for i in pdfs if i not in DEFAULT_ICDS]
    lines = ["---", 'title: "Discovered CBC biomarkers — explained"',
             'subtitle: "Glass-box Shapley attribution + deep literature review (12 diseases)"',
             'author: "biomarker-explainer skill"', f'date: "{_dt.date.today().isoformat()}"',
             "geometry: margin=1in", "---", "",
             "Each of the following sections is a self-contained explainer: the discovered biomarker as a "
             "glass-box tree coloured by each blood test's Shapley impact on the disease prediction, an "
             "expected-vs-surprising classification of every component, and the supporting literature with "
             "verified citations. SHAP is computed out-of-sample on MIMIC-IV; FMF (277) is omitted (no MIMIC cases).", "",
             "| Disease (ICD) | AUC (MIMIC) | Expected | Surprising | Unclear |",
             "|---|---|---|---|---|"]
    for icd in order:
        c = counts.get(icd, {"disease": icd, "exp": 0, "sur": 0, "unc": 0})
        auc = findings_by_icd.get(icd, {}).get("whole_auc_mimic", "")
        lines.append(f"| {c['disease']} ({icd}) | {auc} | {c['exp']} | {c['sur']} | {c['unc']} |")
    cover_md = agg / "_cover.md"
    cover_md.write_text("\n".join(lines))
    cover_pdf = agg / "_cover.pdf"
    cov = subprocess.run(["pandoc", str(cover_md), "-o", str(cover_pdf),
                          "--pdf-engine=pdflatex", "-V", "geometry:margin=1in"],
                         capture_output=True, text=True)
    merged = agg / "all_biomarkers_explained.pdf"
    inputs = ([str(cover_pdf)] if (cov.returncode == 0 and cover_pdf.exists()) else []) + \
             [str(pdfs[i]) for i in order]
    if shutil.which("pdfunite"):
        subprocess.run(["pdfunite", *inputs, str(merged)], check=False)
        print(f"[finalize] compendium -> {merged}  ({len(order)} diseases)")
    else:
        print("[finalize] pdfunite missing — per-disease PDFs are in Results/<icd>/biomarker_explainer/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--icds", nargs="*", default=DEFAULT_ICDS)
    f = sub.add_parser("finalize")
    f.add_argument("--workflow-output", required=True)
    args = ap.parse_args()
    repo = find_repo()
    if args.cmd == "prep":
        prep(args.icds, repo)
    else:
        finalize(Path(args.workflow_output), repo)


if __name__ == "__main__":
    main()
