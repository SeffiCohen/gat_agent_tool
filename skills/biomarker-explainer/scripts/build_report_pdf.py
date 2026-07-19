#!/usr/bin/env python3
r"""build_report_pdf.py — turn the biomarker-explainer Markdown report into a
self-contained LaTeX (.tex) + PDF that embeds the glass-box SHAP figure, the full
expected-vs-surprising explanation, the summary table, and the verified references.

Pipeline: the synthesised Markdown report (from deep_litreview.workflow.js) is
lightly sanitised — a few prose unicode glyphs (Greek, ×, ≤, superscripts ...) are
rewritten to robust LaTeX (\ensuremath{...} / \textsuperscript{...}, which pandoc's
raw_tex passes through and which never collide when adjacent) so the document
compiles on a plain pdflatex. The figure is injected under the title, pandoc emits
a standalone .tex, and we compile that .tex to PDF ourselves (latexmk).

Usage:
  python build_report_pdf.py --md <report.md> --figure <tree.png> \
      --outdir <dir> --stem <name> [--title "..."] [--engine pdflatex]

Writes:  <outdir>/<stem>_explained.tex  and  <outdir>/<stem>_explained.pdf
Requires: pandoc + a LaTeX engine (pdflatex / xelatex / lualatex) + latexmk (optional).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path


def _em(cmd: str) -> str:
    return r"\ensuremath{" + cmd + "}"


# prose unicode -> robust LaTeX. \ensuremath / \textsuperscript survive being placed
# next to each other (e.g. ×10⁶/µL) without creating stray $...$ / $$ that pandoc
# would mis-read as display math. pandoc's raw_tex (on for -f markdown) passes them
# through verbatim. The report's intentional $$...$$ formula is pure ASCII, untouched.
SUBS = {
    "−": "-", "–": "--", "—": "---",
    "≈": _em(r"\approx"), "≤": _em(r"\le"), "≥": _em(r"\ge"), "≡": _em(r"\equiv"),
    "≠": _em(r"\ne"), "∼": _em(r"\sim"), "∝": _em(r"\propto"), "∈": _em(r"\in"),
    "×": _em(r"\times"), "·": _em(r"\cdot"), "÷": _em(r"\div"),
    "→": _em(r"\rightarrow"), "←": _em(r"\leftarrow"), "↔": _em(r"\leftrightarrow"),
    "±": _em(r"\pm"), "∞": _em(r"\infty"), "√": _em(r"\surd"), "∆": _em(r"\Delta"),
    "α": _em(r"\alpha"), "β": _em(r"\beta"), "γ": _em(r"\gamma"), "δ": _em(r"\delta"),
    "ε": _em(r"\epsilon"), "θ": _em(r"\theta"), "κ": _em(r"\kappa"), "λ": _em(r"\lambda"),
    "μ": _em(r"\mu"), "µ": _em(r"\mu"), "σ": _em(r"\sigma"), "χ": _em(r"\chi"), "ω": _em(r"\omega"),
    "Δ": _em(r"\Delta"), "Σ": _em(r"\Sigma"), "Ω": _em(r"\Omega"),
    "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}",
    "°": r"\textdegree{}", "’": "'", "‘": "'", "“": '"', "”": '"',
    "…": "...", " ": " ",
}
# chr()-keyed supplement (keeps this source ASCII). Applied with SUBS.
_SUPP = {
    chr(0x00B9): r"\textsuperscript{1}", chr(0x2070): r"\textsuperscript{0}",
    chr(0x2074): r"\textsuperscript{4}", chr(0x2075): r"\textsuperscript{5}",
    chr(0x2076): r"\textsuperscript{6}", chr(0x2077): r"\textsuperscript{7}",
    chr(0x2078): r"\textsuperscript{8}", chr(0x2079): r"\textsuperscript{9}",
    chr(0x2080): r"\textsubscript{0}", chr(0x2081): r"\textsubscript{1}",
    chr(0x2082): r"\textsubscript{2}", chr(0x2083): r"\textsubscript{3}", chr(0x2084): r"\textsubscript{4}",
    chr(0x2261): _em(r"\equiv"), chr(0x2264): _em(r"\le"), chr(0x2265): _em(r"\ge"),
    chr(0x2248): _em(r"\approx"), chr(0x2260): _em(r"\ne"), chr(0x221D): _em(r"\propto"),
    chr(0x2208): _em(r"\in"), chr(0x221E): _em(r"\infty"), chr(0x00F7): _em(r"\div"),
    chr(0x2032): "'", chr(0x2033): "''", chr(0x2044): "/", chr(0x2030): r"\textperthousand{}",
    chr(0x0394): _em(r"\Delta"), chr(0x03A3): _em(r"\Sigma"), chr(0x03A9): _em(r"\Omega"),
    chr(0x03B5): _em(r"\epsilon"), chr(0x03B8): _em(r"\theta"), chr(0x03BB): _em(r"\lambda"),
    chr(0x03C3): _em(r"\sigma"), chr(0x03C7): _em(r"\chi"), chr(0x03C9): _em(r"\omega"),
    chr(0x2212): "-",
}
_KEEP = "".join(chr(c) for c in (0x2013, 0x2014, 0x2019, 0x2018, 0x201C, 0x201D, 0x2026))


def _fold_math_italic(ch: str) -> str:
    """Fold ONLY the Mathematical Alphanumeric Symbols block (𝜇/𝛼/𝐀) to its base
    letter. Precomposed accented Latin (é, ñ) is deliberately NOT folded — pandoc
    translates it fine, and decomposing it would leave a lone combining mark that
    pdflatex rejects."""
    if 0x1D400 <= ord(ch) <= 0x1D7FF:
        d = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        return d or ch
    return ch


def sanitize(text: str) -> str:
    for k, v in SUBS.items():
        text = text.replace(k, v)
    for k, v in _SUPP.items():
        text = text.replace(k, v)
    text = "".join(_fold_math_italic(ch) for ch in text)
    for k, v in SUBS.items():      # re-map Greek the math-italic fold just exposed
        text = text.replace(k, v)
    for k, v in _SUPP.items():
        text = text.replace(k, v)
    # final net: keep ASCII, real letters (precomposed accents é/ñ + leftover Greek),
    # and pandoc-safe punctuation; drop stray combining marks and unmapped symbols.
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ord(ch) < 128 or ch in _KEEP or cat.startswith("L"):
            out.append(ch)
        elif cat.startswith("M"):
            continue
        else:
            out.append(_SUPP.get(ch, ""))
    return "".join(out)


def split_title(md: str):
    """Pull the first '# ...' line out as the document title; return (title, body).
    Anything before that H1 (e.g. an LLM 'I'll write this directly...' preamble that
    slipped past the synthesis instruction) is dropped — the report template always
    opens with '# Biomarker explained:'."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("# ") and not ln.strip().startswith("## "):
            title = ln.strip()[2:].strip()
            return title, "\n".join(lines[i + 1:])
    return None, md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", required=True, help="the synthesised report markdown")
    ap.add_argument("--figure", help="glass-box SHAP figure PNG to embed (annotated preferred)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--stem", required=True, help="output filename stem")
    ap.add_argument("--title", help="override document title")
    ap.add_argument("--subtitle", default="Glass-box Shapley explanation + literature review",
                    help="document subtitle")
    ap.add_argument("--figwidth", type=int, default=38, help="figure width as %% of text width")
    ap.add_argument("--margin", type=float, default=0.75, help="page margin in inches")
    ap.add_argument("--fontsize", type=int, default=9, help="body font size in pt")
    ap.add_argument("--engine", default="pdflatex", choices=["pdflatex", "xelatex", "lualatex"])
    args = ap.parse_args()

    if not shutil.which("pandoc"):
        sys.exit("pandoc not found — install pandoc (and a LaTeX engine) to build the PDF.")
    if not shutil.which(args.engine):
        for alt in ("pdflatex", "xelatex", "lualatex"):
            if shutil.which(alt):
                args.engine = alt
                break
        else:
            sys.exit("no LaTeX engine (pdflatex/xelatex/lualatex) found.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md = Path(args.md).read_text()
    title, body = split_title(md)
    title = sanitize(args.title or title or "Biomarker explained")

    # inject the figure right under the title (before the first section)
    fig_block = ""
    if args.figure and Path(args.figure).exists():
        figabs = str(Path(args.figure).resolve())
        cap = ("Each blood-test leaf is coloured by its Shapley impact on the disease prediction "
               "(red = raises risk, blue = lowers it; triangle = direction; "
               "circle = expected, star = surprising).")
        fig_block = f"\n\n![{cap}]({figabs}){{width={args.figwidth}%}}\n\n"

    body = sanitize(body)
    combined = body if not fig_block else (fig_block + body)

    today = _dt.date.today().isoformat()
    meta = ["---",
            f'title: "{title}"',
            f'subtitle: "{args.subtitle}  ({today})"',
            f"geometry: margin={args.margin}in",
            "colorlinks: true", "linkcolor: RoyalBlue", "urlcolor: RoyalBlue",
            f"fontsize: {args.fontsize}pt",
            "header-includes: |",
            "  \\usepackage{titling}",
            "  \\setlength{\\droptitle}{-4em}",
            "  \\setlength{\\parskip}{2pt}",
            "  \\setlength{\\parindent}{0pt}",
            "  \\posttitle{\\par\\end{center}\\vskip 0.3em}",
            "---", ""]
    src = "\n".join(meta) + combined

    with tempfile.NamedTemporaryFile("w", suffix=".md", dir=str(outdir), delete=False) as tf:
        tf.write(src)
        tmp_md = tf.name

    tex_out = outdir / f"{args.stem}_explained.tex"
    pdf_out = outdir / f"{args.stem}_explained.pdf"
    common = ["-f", "markdown", "--resource-path", str(Path(args.figure).resolve().parent) if args.figure else "."]
    build = Path(tempfile.mkdtemp(prefix="brp_"))
    try:
        # 1) standalone .tex (pandoc's engine-adaptive iftex preamble compiles anywhere)
        subprocess.run(["pandoc", tmp_md, *common, "-s", "-t", "latex", "-o", str(tex_out)], check=True)
        # 2) compile that .tex to PDF ourselves; latexmk handles the hyperref reruns.
        eng = args.engine
        if shutil.which("latexmk"):
            flag = {"pdflatex": "-pdf", "xelatex": "-xelatex", "lualatex": "-lualatex"}[eng]
            subprocess.run(["latexmk", flag, "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={build}", str(tex_out)], check=True, stdout=subprocess.DEVNULL)
        else:
            for _ in range(2):
                subprocess.run([eng, "-interaction=nonstopmode", "-halt-on-error",
                                f"-output-directory={build}", str(tex_out)], check=True, stdout=subprocess.DEVNULL)
        shutil.move(str(build / f"{tex_out.stem}.pdf"), str(pdf_out))
    finally:
        Path(tmp_md).unlink(missing_ok=True)
        shutil.rmtree(build, ignore_errors=True)

    print(f"wrote {tex_out}")
    print(f"wrote {pdf_out}  (engine={eng})")


if __name__ == "__main__":
    main()
