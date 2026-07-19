#!/usr/bin/env python3
r"""shap_biomarker.py — explain ONE discovered CBC biomarker with exact Shapley impact.

Given a discovered expression and a labelled cohort, this:
  1. fits the biomarker as a univariate disease model  P(disease)=logistic(z),
     z = (orientation-fixed expression value - mu)/sigma  (one predictor: the
     whole expression E, NOT the individual features);
  2. attributes that prediction back to each *raw* CBC feature with EXACT
     interventional Shapley values (cohort median = reference patient, all 2^d
     coalitions, d = number of distinct features <= ~8 so this is cheap and exact).
     Shapley runs end-to-end on  features -> E -> P(disease), so a feature's sign
     and magnitude reflect how it is actually used inside the formula (numerator
     vs denominator, products, orientation) -- which a raw feature-vs-disease
     correlation cannot capture;
  3. renders the glass-box expression tree, each leaf coloured by its signed
     Shapley impact scaled to the strongest driver in this formula
     (red/up = higher level raises predicted risk; blue/down = lowers it);
  4. emits findings.json -- the structured per-feature directions/magnitudes that
     the deep-literature-review workflow consumes to classify each component as
     expected vs surprising.

Two ways to point at a biomarker:
  --icd 714                         look up winner_expression + cohort from the
                                    frozen iter_audit + the MIMIC parquet map
  --expr "..." --disease "RA" \     supply an arbitrary expression + any labelled
      --parquet x.parquet --target-col icd_714      cohort parquet

Outputs (under --outdir):  <stem>_shap_tree.png, <stem>_shap_impact.csv,
                           <stem>_findings.json    (stem defaults to the icd/slug)
"""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import os
import re
import sys
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

PARQUET = {
    "242": "242_Hyperthyroidism_mimic.parquet", "250": "250_T1D_mimic.parquet",
    "277": "277_FMF_mimic.parquet", "340": "340_MS_mimic.parquet",
    "555": "555_Crohns_mimic.parquet", "556": "556_UlcerativeColitis_mimic.parquet",
    "696": "696_Psoriasis_mimic.parquet", "714": "714_RheumatoidArthritis_mimic.parquet",
    "2452": "2452_Hashimoto_mimic.parquet", "5790": "5790_Celiac_mimic.parquet",
    "7100": "7100_SLE_mimic.parquet", "7101": "7101_SystemicSclerosis_mimic.parquet",
    "7102": "7102_Sjogrens_mimic.parquet",
}
NAME = {"242": "Hyperthyroidism", "250": "Type 1 diabetes", "277": "Familial Mediterranean fever",
        "340": "Multiple sclerosis", "555": "Crohn's disease", "556": "Ulcerative colitis",
        "696": "Psoriasis", "714": "Rheumatoid arthritis", "2452": "Hashimoto thyroiditis",
        "5790": "Celiac disease", "7100": "Systemic lupus (SLE)", "7101": "Systemic sclerosis",
        "7102": "Sjogren's syndrome"}
PRETTY = {"NEUTpct": "NEUT%", "LYMpct": "LYM%", "MONOpct": "MONO%", "EOS_pct": "EOS%",
          "BASO_pct": "BASO%", "NEUT_abs": "NEUT#"}
PERCENT = {"NEUTpct", "LYMpct", "MONOpct", "EOS_pct", "BASO_pct"}
SEED, N_EXPL = 20260429, 4000


def pp(f):
    return PRETTY.get(f, f)


# ------------------------------------------------------------------ discovery
def find_code_dir(explicit):
    cands = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("GAT_AGENT_TOOL_CODE_DIR"):
        cands.append(Path(os.environ["GAT_AGENT_TOOL_CODE_DIR"]))
    here = Path(__file__).resolve()
    for base in [Path.cwd(), here.parent]:
        for up in [base, *base.parents]:
            cands.append(up / "Code")
    for c in cands:
        if (c / "_external_scoring.py").exists():
            return c.resolve()
    raise SystemExit("Could not find Code/_external_scoring.py — pass --code-dir")


def feats(expr):
    return sorted(set(re.findall(r"lab_([A-Za-z_]+?)_last", expr)))


def num_den_counts(expr):
    """{feature: [num_count, den_count]} via AST division-parity walk."""
    py = re.sub(r"lab_([A-Za-z_]+?)_last", r"\1", expr)
    out = {}

    def walk(node, par):
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div):
                walk(node.left, par); walk(node.right, par ^ 1)
            else:
                walk(node.left, par); walk(node.right, par)
        elif isinstance(node, ast.UnaryOp):
            walk(node.operand, par)
        elif isinstance(node, ast.Name):
            out.setdefault(node.id, [0, 0]); out[node.id][par] += 1
    walk(ast.parse(py, mode="eval").body, 0)
    return out


# ------------------------------------------------------------------ Shapley
def shapley(expr, df, target, eval_expression):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    cols = [f"lab_{f}_last" for f in feats(expr)]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return {}, {}
    d = len(cols)

    E_full = eval_expression(expr, df)
    if E_full is None:
        return {}, {}
    fin = np.isfinite(E_full) & np.isfinite(target)
    if fin.sum() < 20 or len(np.unique(target[fin])) < 2:
        return {}, {}
    auc_dir = roc_auc_score(target[fin], E_full[fin])
    sign = 1.0 if auc_dir >= 0.5 else -1.0
    Eo = sign * E_full[fin]
    mu, sigma = float(np.mean(Eo)), float(np.std(Eo))
    if not np.isfinite(sigma) or sigma == 0:
        return {}, {}
    whole_auc = float(max(auc_dir, 1 - auc_dir))
    lr = LogisticRegression(max_iter=1000).fit(((Eo - mu) / sigma).reshape(-1, 1),
                                                target[fin].astype(int))

    def g(E):
        E = np.asarray(E, dtype=np.float64)
        out = np.full(E.shape, np.nan)
        ok = np.isfinite(E)
        if ok.any():
            out[ok] = lr.predict_proba(((sign * E[ok] - mu) / sigma).reshape(-1, 1))[:, 1]
        return out

    idx = np.arange(len(df))
    if len(idx) > N_EXPL:
        idx = np.random.RandomState(SEED).choice(idx, N_EXPL, replace=False)
    sub = df.iloc[idx]
    N = len(sub)
    inst, med = {}, {}
    for c in cols:
        a = sub[c].to_numpy(dtype=np.float64)
        med[c] = float(np.nanmedian(df[c].to_numpy(dtype=np.float64)))
        inst[c] = np.where(np.isfinite(a), a, med[c])

    vcache = {}
    for r in range(d + 1):
        for S in itertools.combinations(range(d), r):
            Sset = set(S)
            data = {c: (inst[c] if j in Sset else np.full(N, med[c]))
                    for j, c in enumerate(cols)}
            vcache[S] = g(eval_expression(expr, pd.DataFrame(data)))

    phi = np.zeros((N, d))
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(d - r - 1) / factorial(d)
            for S in itertools.combinations(others, r):
                Swi = tuple(sorted(S + (i,)))
                phi[:, i] += w * (vcache[Swi] - vcache[S])

    nd = num_den_counts(expr)
    res = {}
    for j, c in enumerate(cols):
        f = c[len("lab_"):-len("_last")]
        col = phi[:, j]
        good = np.isfinite(col)
        if good.sum() < 3:
            continue
        mean_abs = float(np.mean(np.abs(col[good])))
        xi = inst[c][good]
        corr = float(np.corrcoef(xi, col[good])[0, 1]) if (np.std(xi) > 0 and np.std(col[good]) > 0) else 0.0
        sdir = 1 if corr >= 0 else -1
        ncnt, dcnt = nd.get(f, [0, 0])
        # standalone signed rank-biserial (direction the raw feature points, for context)
        scol = df[c].to_numpy(dtype=np.float64)
        sm = ~(np.isnan(scol) | np.isnan(target))
        scorr = float(2 * roc_auc_score(target[sm], scol[sm]) - 1) if sm.sum() >= 10 and len(np.unique(target[sm])) > 1 else float("nan")
        res[f] = {"mean_abs_shap": mean_abs, "shap_sign": sdir,
                  "signed_shap": sdir * mean_abs, "num_count": ncnt, "den_count": dcnt,
                  "standalone_signed_corr": scorr, "n": int(good.sum())}
    meta = {"whole_auc_mimic": whole_auc, "n_patients": N,
            "orientation_sign": int(sign)}
    return res, meta


# ------------------------------------------------------------------ layout
CW, CH, OPW, PW, BARGAP = 2.6, 1.15, 0.95, 0.5, 0.36
SHAP_CMAP = matplotlib.colormaps["RdBu_r"]
SHAP_NORM = Normalize(-1.0, 1.0)


def is_add(n):
    return isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub))


def measure(node):
    if isinstance(node, ast.Name):
        return {"k": "leaf", "name": node.id, "w": CW, "h": CH}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = measure(node.operand)
        if is_add(node.operand):
            inner = {"k": "paren", "inner": inner, "w": inner["w"] + 2 * PW, "h": inner["h"]}
        return {"k": "row", "parts": [("op", "−"), ("box", inner)], "w": OPW + inner["w"], "h": inner["h"]}
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            num, den = measure(node.left), measure(node.right)
            return {"k": "frac", "num": num, "den": den,
                    "w": max(num["w"], den["w"]), "h": num["h"] + den["h"] + 2 * BARGAP}
        opsym = {"Add": "+", "Sub": "−", "Mult": "·"}[type(node.op).__name__]
        L, R = measure(node.left), measure(node.right)
        if isinstance(node.op, ast.Mult):
            if is_add(node.left):
                L = {"k": "paren", "inner": L, "w": L["w"] + 2 * PW, "h": L["h"]}
            if is_add(node.right):
                R = {"k": "paren", "inner": R, "w": R["w"] + 2 * PW, "h": R["h"]}
        elif isinstance(node.op, ast.Sub):
            if is_add(node.right):
                R = {"k": "paren", "inner": R, "w": R["w"] + 2 * PW, "h": R["h"]}
        return {"k": "row", "parts": [("box", L), ("op", opsym), ("box", R)],
                "w": L["w"] + OPW + R["w"], "h": max(L["h"], R["h"])}
    raise ValueError(f"unhandled node {ast.dump(node)}")


def lum(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def shap_color(v):
    if v is None or not np.isfinite(v):
        return (0.84, 0.84, 0.84, 1.0)
    return SHAP_CMAP(SHAP_NORM(v))


CLASS_MARK = {"expected": ("o", "#1a9850"), "surprising": ("*", "#d73027"),
              "unclear": ("D", "#888888")}


def draw_chip(ax, xc, yc, feat, imp, annot):
    v = imp.get(feat)
    fc = shap_color(v)
    blank = v is None or not np.isfinite(v)
    x, y = xc - CW / 2, yc - CH / 2
    ax.add_patch(FancyBboxPatch((x, y), CW, CH, boxstyle="round,pad=0.04,rounding_size=0.14",
                                fc=fc, ec="#333", lw=0.9, hatch=("////" if blank else None), zorder=3))
    tc = "white" if (not blank and lum(fc) < 0.5) else "#15202b"
    ax.text(xc + 0.16, yc, pp(feat), ha="center", va="center", fontsize=11, color=tc, fontweight="bold", zorder=5)
    if not blank and abs(v) > 0.04:
        ax.scatter(x + 0.34, yc, marker=("^" if v > 0 else "v"), s=34, color=tc, zorder=7, linewidths=0)
    if annot and feat in annot:
        mk, mc = CLASS_MARK.get(annot[feat], CLASS_MARK["unclear"])
        ax.scatter(x + CW - 0.18, y + 0.2, marker=mk, s=70, color=mc,
                   edgecolors="white", linewidths=0.6, zorder=8)


def place(ax, box, xc, yc, imp, annot):
    k = box["k"]
    if k == "leaf":
        draw_chip(ax, xc, yc, box["name"], imp, annot)
    elif k == "paren":
        inner = box["inner"]
        fs = 13 + 5 * (inner["h"] > 1.5 * CH)
        ax.text(xc - box["w"] / 2 + PW * 0.45, yc, "(", ha="center", va="center", fontsize=fs, color="#555")
        ax.text(xc + box["w"] / 2 - PW * 0.45, yc, ")", ha="center", va="center", fontsize=fs, color="#555")
        place(ax, inner, xc, yc, imp, annot)
    elif k == "frac":
        num, den = box["num"], box["den"]
        barw = max(num["w"], den["w"])
        ax.plot([xc - barw / 2, xc + barw / 2], [yc, yc], color="#222", lw=1.6, zorder=2)
        place(ax, num, xc, yc - BARGAP - num["h"] / 2, imp, annot)
        place(ax, den, xc, yc + BARGAP + den["h"] / 2, imp, annot)
    elif k == "row":
        x = xc - box["w"] / 2
        for typ, payload in box["parts"]:
            if typ == "box":
                place(ax, payload, x + payload["w"] / 2, yc, imp, annot)
                x += payload["w"]
            else:
                ax.text(x + OPW / 2, yc, payload, ha="center", va="center", fontsize=12, color="#333")
                x += OPW


def parse_expr(expr):
    return ast.parse(re.sub(r"lab_([A-Za-z_]+?)_last", r"\1", expr), mode="eval").body


def render(expr, imp, disease, out, annot=None, sc=0.52):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    box = measure(parse_expr(expr))
    M, TITLEH, CBAR = 1.2, 2.2, 1.7
    totalW = box["w"] + 2 * M
    totalH = box["h"] + TITLEH + M
    fig = plt.figure(figsize=(max(totalW * sc, 5.5), totalH * sc + CBAR))
    cbar_frac = CBAR / (totalH * sc + CBAR)
    ax = fig.add_axes([0.0, cbar_frac, 1.0, 1.0 - cbar_frac - 0.01])
    ax.set_xlim(0, totalW)
    ax.set_ylim(totalH, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    pretty_expr = re.sub(r"lab_([A-Za-z_]+?)_last", lambda m: pp(m.group(1)), expr).replace("*", "·")
    ax.text(totalW / 2, 0.55, disease, ha="center", va="center", fontsize=15, fontweight="bold", color="#15202b")
    ax.text(totalW / 2, 1.45, pretty_expr, ha="center", va="center", fontsize=10.5, color="#445", style="italic")
    place(ax, box, totalW / 2, TITLEH + box["h"] / 2, imp, annot)

    cax = fig.add_axes([0.30, cbar_frac * 0.66, 0.40, cbar_frac * 0.10])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=SHAP_CMAP, norm=SHAP_NORM,
                                          orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["lowers risk", "negligible", "raises risk"], fontsize=7)
    cb.set_label("Shapley impact of the blood test on the disease prediction "
                 "(signed, scaled to the strongest driver in this formula)", fontsize=7.5)
    handles = [Line2D([0], [0], marker="^", color="w", markerfacecolor="#b2182b", label="higher level raises predicted risk", markersize=7),
               Line2D([0], [0], marker="v", color="w", markerfacecolor="#2166ac", label="higher level lowers predicted risk", markersize=7)]
    if annot:
        handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a9850", label="literature: expected", markersize=7),
                    Line2D([0], [0], marker="*", color="w", markerfacecolor="#d73027", label="literature: surprising", markersize=9)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, cbar_frac * 0.06),
               fontsize=7, frameon=False, ncol=2)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icd", help="disease folder/ICD (looks up iter_audit + MIMIC parquet)")
    ap.add_argument("--expr", help="explicit biomarker expression (lab_<F>_last form, or plain F)")
    ap.add_argument("--disease", help="disease display name (for --expr mode)")
    ap.add_argument("--parquet", help="labelled cohort parquet (for --expr mode)")
    ap.add_argument("--target-col", help="binary target column, e.g. icd_714 (for --expr mode)")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--stem", help="output filename stem (default: icd or slug)")
    ap.add_argument("--code-dir", help="path to repo Code/ (auto-detected otherwise)")
    ap.add_argument("--iter-audit", help="path to iter_audit.csv")
    ap.add_argument("--mimic-dir", help="path to Data/MIMIC/preprocessed")
    ap.add_argument("--annotate", help="JSON {feature: expected|surprising|unclear} -> annotated render")
    ap.add_argument("--cohort-label", default="user-supplied cohort",
                    help="provenance label recorded in findings.json (e.g. 'MIMIC-IV')")
    args = ap.parse_args()

    code_dir = find_code_dir(args.code_dir)
    sys.path.insert(0, str(code_dir))
    from _external_scoring import eval_expression  # noqa: E402
    repo = code_dir.parent
    iter_audit = (Path(args.iter_audit) if args.iter_audit
                  else Path(__file__).resolve().parents[1] / "data" / "winner_expressions.csv")
    mimic = Path(args.mimic_dir) if args.mimic_dir else repo / "Data/MIMIC/preprocessed"

    # resolve biomarker + cohort
    if args.expr:
        expr = args.expr
        # accept plain-feature form too: F -> lab_F_last for known CBC names
        if "lab_" not in expr:
            known = set(PRETTY) | {"RBC", "HB", "HCT", "MCV", "MCH", "MCHC", "RDW", "WBC", "PLT"} | set(PERCENT) | {"NEUT_abs"}
            expr = re.sub(r"\b([A-Za-z_]+)\b", lambda m: f"lab_{m.group(1)}_last" if m.group(1) in known else m.group(1), expr)
        disease = args.disease or "Disease"
        icd = args.target_col.replace("icd_", "") if args.target_col else "expr"
        if not args.parquet or not args.target_col:
            raise SystemExit("--expr mode needs --parquet and --target-col for the Shapley computation")
        df = pd.read_parquet(args.parquet)
        target = df[args.target_col].to_numpy(dtype=np.float64)
    else:
        if not args.icd:
            raise SystemExit("pass --icd OR --expr/--disease/--parquet/--target-col")
        icd = args.icd
        rows = {r["disease_folder"]: r for r in csv.DictReader(open(iter_audit))}
        if icd not in rows:
            raise SystemExit(f"icd {icd} not in {iter_audit}")
        expr = rows[icd]["winner_expression"]
        disease = NAME.get(icd, rows[icd]["disease_name"])
        pq = mimic / PARQUET.get(icd, "")
        if not pq.exists():
            raise SystemExit(f"no MIMIC cohort for {icd} ({pq}); SHAP needs a labelled cohort. "
                             f"Use --parquet to supply one.")
        df = pd.read_parquet(pq)
        target = df[f"icd_{icd}"].to_numpy(dtype=np.float64)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or icd

    res, meta = shapley(expr, df, target, eval_expression)
    if not res:
        raise SystemExit("Shapley computation failed (no labels / degenerate expression).")

    maxabs = max((v["mean_abs_shap"] for v in res.values()), default=0.0) or 1.0
    # rel impact (signed) in [-1,1], scaled to the formula's strongest driver
    imp = {f: v["signed_shap"] / maxabs for f, v in res.items()}

    # rank + assemble findings
    order = sorted(res.items(), key=lambda kv: -kv[1]["mean_abs_shap"])
    pretty_expr = re.sub(r"lab_([A-Za-z_]+?)_last", lambda m: pp(m.group(1)), expr).replace("*", "·")
    features = []
    for rank, (f, v) in enumerate(order, 1):
        role = ("numerator" if v["num_count"] and not v["den_count"]
                else "denominator" if v["den_count"] and not v["num_count"]
                else "numerator+denominator" if v["num_count"] and v["den_count"]
                else "—")
        features.append({
            "feature": f, "pretty": pp(f), "is_percent": f in PERCENT,
            "role": role, "num_count": v["num_count"], "den_count": v["den_count"],
            "shap_sign": v["shap_sign"],
            "shap_direction": "higher level RAISES predicted disease risk" if v["shap_sign"] > 0
                              else "higher level LOWERS predicted disease risk",
            "mean_abs_shap": round(v["mean_abs_shap"], 6),
            "rel_impact": round(imp[f], 4), "rank": rank,
            "standalone_signed_corr": (round(v["standalone_signed_corr"], 4)
                                       if np.isfinite(v["standalone_signed_corr"]) else None),
        })

    findings = {"icd": icd, "disease": disease, "expression_pretty": pretty_expr,
                "expression_raw": expr, "n_patients": meta["n_patients"],
                "whole_auc_mimic": round(meta["whole_auc_mimic"], 4),
                "cohort": args.cohort_label,
                "shap_method": "exact interventional Shapley, cohort-median reference, "
                               "univariate-logistic disease model on the standardised biomarker",
                "features": features}

    # write outputs
    with open(outdir / f"{stem}_shap_impact.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "pretty", "role", "num_count", "den_count", "shap_sign",
                    "shap_direction", "mean_abs_shap", "rel_impact", "rank", "standalone_signed_corr"])
        for ft in features:
            w.writerow([ft["feature"], ft["pretty"], ft["role"], ft["num_count"], ft["den_count"],
                        ft["shap_sign"], ft["shap_direction"], ft["mean_abs_shap"], ft["rel_impact"],
                        ft["rank"], ft["standalone_signed_corr"]])
    with open(outdir / f"{stem}_findings.json", "w") as fh:
        json.dump(findings, fh, indent=2)

    annot = None
    if args.annotate and Path(args.annotate).exists():
        annot = json.load(open(args.annotate))
    render(expr, imp, disease, outdir / f"{stem}_shap_tree.png", annot=annot)
    if annot:
        render(expr, imp, disease, outdir / f"{stem}_shap_tree_annotated.png", annot=annot)

    print(f"\n{disease} ({icd})  whole-AUC(MIMIC)={meta['whole_auc_mimic']:.3f}  n={meta['n_patients']}")
    for ft in features:
        arrow = "↑raises" if ft["shap_sign"] > 0 else "↓lowers"
        print(f"  #{ft['rank']} {ft['pretty']:<7} {ft['role']:<22} rel={ft['rel_impact']:+.2f}  {arrow} risk")
    print(f"\nfindings -> {outdir / f'{stem}_findings.json'}")


if __name__ == "__main__":
    main()
