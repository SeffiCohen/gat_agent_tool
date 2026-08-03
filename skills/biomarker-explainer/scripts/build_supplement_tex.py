#!/usr/bin/env python3
r"""build_supplement_tex.py — one unified LaTeX source for the whole one-pager supplement.

The per-disease PDFs are compiled separately and merged with pdfunite, so there is no
single editable source for the compendium. This builds one: a standalone article with a
cover, one page-per-disease section, portable relative figure paths, and a single real
bibliography driven by \cite keys shared with biomarker_explainer_refs.bib.

Content comes from the same verified workflow output the one-pagers are built from, so
the two stay in step; references are rendered from the PubMed cache (citations.py).

Usage:
  python build_supplement_tex.py --workflow-output <workflow_result.json> \
      --outdir <dir> [--stem biomarker_onepagers_supp] [--compile]

Writes: <outdir>/<stem>.tex, <outdir>/biomarker_explainer_refs.bib,
        <outdir>/figures_onepagers/<icd>.png  (+ naturemag.bst and the PDF with --compile)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))

from build_report_pdf import sanitize  # noqa: E402
from citations import assign_keys, load_records, write_bib  # noqa: E402
from run_all_biomarkers import _ESTPHRASE, _first_sentence, DEFAULT_ICDS, find_repo  # noqa: E402

_TEX_SPECIAL = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def tex(s):
    """Escape LaTeX specials, then map prose unicode to LaTeX.

    Order matters: escaping runs first, on raw text that contains no LaTeX, so it cannot
    mangle the commands sanitize() then emits (\\ensuremath{...}, \\textsuperscript{...})."""
    s = str(s or "")
    for a, b in _TEX_SPECIAL:
        s = s.replace(a, b)
    return sanitize(s)


def component_sentences(c, disease):
    """The two sentences the one-pagers use, in LaTeX: direction + verdict, then mechanism."""
    pretty = c.get("pretty", c.get("feature", "?"))
    research, verify = c.get("research") or {}, c.get("verify") or {}
    rdir = "raises" if "RAIS" in (c.get("shap_direction", "").upper()) else "lowers"
    verdict = (verify.get("final_classification") or research.get("classification") or "unclear").lower()
    conf = verify.get("confidence") or research.get("evidence_strength") or ""
    est = _ESTPHRASE.get(research.get("established_direction", "unknown"), "of unclear direction in")
    if verdict == "expected":
        s1 = (f"Higher {tex(pretty)} {rdir} the predicted risk --- \\textbf{{expected}}, consistent "
              f"with {tex(pretty)} being {tex(est)} {tex(disease)}")
    elif verdict == "surprising":
        s1 = (f"Higher {tex(pretty)} {rdir} the predicted risk --- \\textbf{{surprising}}, since "
              f"{tex(pretty)} is typically {tex(est)} {tex(disease)}")
    else:
        s1 = (f"Higher {tex(pretty)} {rdir} the predicted risk --- \\textbf{{unclear}}, as the "
              f"literature on {tex(pretty)} in {tex(disease)} is limited")
    s1 += f" ({tex(conf)} confidence)." if conf else "."
    s2 = tex(_first_sentence(research.get("mechanism") or research.get("rationale") or "", maxlen=112))
    return pretty, s1, s2


def first_pmid(c):
    research, verify = c.get("research") or {}, c.get("verify") or {}
    vcs = [x for x in (verify.get("verified_citations") or [])
           if x.get("exists") and x.get("supports_claim")]
    for cite in (vcs or (research.get("citations") or [])):
        return (cite.get("pmid") or "").strip()
    return ""


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,top=2.0cm,bottom=2.0cm,left=2.0cm,right=2.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}\fancyhead[R]{S\thepage}
\renewcommand{\headrulewidth}{0pt}\setlength{\headheight}{14pt}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}

\title{\vspace{-1.5cm}\textbf{Supplementary Data 1}\\[2pt]
\large Discovered CBC biomarkers explained: Shapley attribution and literature concordance}
\date{}
"""

INTRO = r"""Each section below is a self-contained explainer for one discovered expression: the
biomarker drawn as a glass-box tree whose leaves are coloured by each blood test's Shapley
contribution to the disease prediction, an Expected / Surprising / Unclear classification of
every component against the clinical literature, and the supporting evidence. Shapley values are
computed out-of-sample on MIMIC-IV; familial Mediterranean fever (277) is omitted because the
MIMIC-IV cohort contains no labelled cases.

Every citation was resolved against PubMed by identifier and independently re-verified; the
bibliography at the end is shared across all twelve explainers.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow-output", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stem", default="biomarker_onepagers_supp")
    ap.add_argument("--bst", help="bibliography style file to copy in (default: find naturemag.bst)")
    ap.add_argument("--compile", action="store_true", help="run pdflatex/bibtex to produce the PDF")
    args = ap.parse_args()

    repo = find_repo()
    outdir = Path(args.outdir)
    figdir = outdir / "figures_onepagers"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(exist_ok=True)

    res = json.load(open(args.workflow_output))
    results = {str(r["icd"]): r for r in (res.get("results") if isinstance(res, dict) else res)
               if r.get("components") and r.get("report")}
    order = [i for i in DEFAULT_ICDS if i in results] + [i for i in results if i not in DEFAULT_ICDS]

    pubmed = load_records()
    if not pubmed:
        sys.exit("no PubMed record cache; run fetch_pubmed_records.py first")

    # Only cited records go in the bibliography, so no uncited entry can appear.
    cited = []
    for icd in order:
        for c in results[icd].get("components") or []:
            p = first_pmid(c)
            if p in pubmed and p not in [r["pmid"] for r in cited]:
                cited.append(pubmed[p])
    keys = assign_keys(cited)
    write_bib(cited, outdir / "biomarker_explainer_refs.bib")

    body, missing_figs = [], []
    for icd in order:
        r = results[icd]
        disease, expr = r.get("disease", icd), r.get("expression", "")
        auc = r.get("whole_auc_mimic", "")
        src = repo / "Results" / icd / "biomarker_explainer" / f"{icd}_shap_tree_annotated.png"
        if not src.exists():
            src = src.with_name(f"{icd}_shap_tree.png")
        if src.exists():
            shutil.copy(src, figdir / f"{icd}.png")
        else:
            missing_figs.append(icd)

        body.append(f"\\section*{{{tex(disease)} (ICD {tex(icd)})}}")
        body.append(f"\\addcontentsline{{toc}}{{section}}{{{tex(disease)} ({tex(icd)})}}")
        if src.exists():
            body.append(r"\begin{center}")
            body.append(f"\\includegraphics[width=0.62\\textwidth]{{figures_onepagers/{icd}.png}}")
            body.append(r"\end{center}")
            body.append(r"{\footnotesize Each blood-test leaf is coloured by its Shapley impact on the "
                        r"disease prediction (red = raises risk, blue = lowers it; triangle = direction; "
                        r"circle = expected, star = surprising).\par}")
        body.append(r"\vspace{2pt}")
        body.append(f"\\textbf{{Biomarker:}} {tex(expr)} $\\cdot$ out-of-sample AUC (MIMIC) "
                    f"$\\approx$ {tex(auc)}\\par")
        for c in r.get("components") or []:
            pretty, s1, s2 = component_sentences(c, disease)
            p = first_pmid(c)
            cite = f" \\cite{{{keys[p]}}}" if p in keys else ""
            role = tex(c.get("role", ""))
            body.append(f"\\textbf{{{tex(pretty)}}} ({role}). {s1} {s2}{cite}\\par")
        body.append(r"\vspace{2pt}")
        body.append(rf"{{\footnotesize\emph{{Caveats: the logistic link is fit on MIMIC (in-sample); "
                    rf"the expression was discovered on Clalit; age confounds several red-cell indices; "
                    rf"the out-of-sample AUC is modest ($\approx${tex(auc)}); and a feature recurring in "
                    rf"numerator+denominator may be an overfit artifact rather than biology.}}\par}}")
        body.append(r"\clearpage")

    bst = Path(args.bst) if args.bst else None
    if bst is None:
        for cand in repo.glob("*/naturemag.bst"):
            bst = cand
            break
    style = "naturemag"
    if bst and bst.exists():
        shutil.copy(bst, outdir / "naturemag.bst")
    else:
        style = "unsrt"
        print("naturemag.bst not found — falling back to \\bibliographystyle{unsrt}", file=sys.stderr)

    doc = "\n".join([PREAMBLE, r"\begin{document}", r"\maketitle", r"\thispagestyle{fancy}",
                     INTRO, r"\clearpage", *body,
                     r"\bibliographystyle{" + style + "}",
                     r"\bibliography{biomarker_explainer_refs}", r"\end{document}", ""])
    tex_path = outdir / f"{args.stem}.tex"
    tex_path.write_text(doc, encoding="utf-8")
    print(f"wrote {tex_path}  ({len(order)} diseases, {len(cited)} references)")
    if missing_figs:
        print(f"WARNING: no figure found for {', '.join(missing_figs)}", file=sys.stderr)

    if args.compile:
        for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                    ["bibtex", args.stem],
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]):
            p = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True)
            if p.returncode != 0 and cmd[0] == "pdflatex":
                tail = "\n".join(p.stdout.splitlines()[-25:])
                sys.exit(f"{cmd[0]} failed:\n{tail}")
        print(f"wrote {outdir / (args.stem + '.pdf')}")


if __name__ == "__main__":
    main()
