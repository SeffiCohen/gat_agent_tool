"""Authoritative bibliographic records for the biomarker one-pagers.

The literature-review workflow returns citations in two half-complete halves: the
`research.citations` entries carry an author string but usually no PMID, while the
`verify.verified_citations` entries carry the PMID/DOI but drop the authors. Joining
them on PMID therefore fails silently, and any renderer that falls back to "first word
of the title" emits nonsense like "Hematologic et al.".

The fix is to stop reconstructing bibliographic data from the workflow output at all.
Every citation is resolved once against PubMed (see `fetch_pubmed_records.py`) into a
cache keyed by PMID; this module is the single place that turns a cached record into
either a BibTeX entry or a rendered reference line, so the two can never disagree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CACHE_NAME = "pubmed_records.json"


# --------------------------------------------------------------------------- cache

def cache_path(explicit=None):
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parent.parent / "data" / CACHE_NAME


def load_records(explicit=None):
    """Return {pmid: record}. Missing cache yields {} so callers can degrade loudly."""
    p = cache_path(explicit)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------- utilities

def expand_pages(pages):
    """PubMed compresses page ranges: '190-4' means 190-194, '1123-31' means 1123-1131."""
    pages = (pages or "").strip()
    m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", pages)
    if not m:
        return pages
    start, end = m.group(1), m.group(2)
    if len(end) < len(start):
        end = start[: len(start) - len(end)] + end
    return f"{start}-{end}"


def _fix_surname_case(last):
    """Pre-1960s MEDLINE records store surnames in caps ('JEFFREY'); that is a record-keeping
    artifact, not the author's name. Only touch a single all-caps alphabetic word, so genuine
    mixed-case names and collective author names are left exactly as PubMed has them."""
    if last.isupper() and last.isalpha() and len(last) > 1:
        return last.capitalize()
    return last


def _author_list(rec):
    return [{**a, "last": _fix_surname_case(a["last"])}
            for a in (rec.get("authors") or []) if a.get("last")]


def surname(rec):
    a = _author_list(rec)
    return a[0]["last"] if a else ""


# ------------------------------------------------------------------------- BibTeX

_BIB_SPECIAL = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def bib_escape(s):
    s = (s or "").replace("&amp;", "and").replace("&lt;", "<").replace("&gt;", ">")
    s = re.sub(r"</?(i|b|sub|sup|em|strong)>", "", s)
    for a, b in _BIB_SPECIAL:
        s = s.replace(a, b)
    return s


def cite_key(rec, taken=None):
    """firstauthor+year+firsttitleword, e.g. carli2015leukopenia — stable and readable."""
    last = re.sub(r"[^A-Za-z]", "", surname(rec)).lower() or "ref"
    year = re.sub(r"[^0-9]", "", str(rec.get("year") or "")) or "0000"
    stop = {"a", "an", "the", "of", "in", "on", "and", "for", "with", "is", "are", "to", "at"}
    word = "ref"
    for w in re.findall(r"[A-Za-z]+", rec.get("title") or ""):
        if w.lower() not in stop and len(w) > 2:
            word = w.lower()
            break
    key = f"{last}{year}{word}"
    if taken is not None:
        base, n = key, 1
        while key in taken:
            n += 1
            key = f"{base}{chr(ord('a') + n - 2)}"
        taken.add(key)
    return key


def to_bibtex(rec, key=None):
    """A complete @article entry: every author, unabbreviated title, DOI and PMID."""
    key = key or cite_key(rec)
    authors = " and ".join(
        f"{bib_escape(a['last'])}, {bib_escape(a.get('initials') or '')}".rstrip(", ")
        for a in _author_list(rec)
    )
    fields = [("author", authors),
              ("title", f"{{{bib_escape(rec.get('title') or '')}}}"),
              ("journal", bib_escape(rec.get("journal_iso") or rec.get("journal_full") or "")),
              ("year", str(rec.get("year") or "")),
              ("volume", str(rec.get("volume") or "")),
              ("number", str(rec.get("issue") or "")),
              ("pages", expand_pages(rec.get("pages")).replace("-", "--")),
              ("doi", rec.get("doi") or ""),
              ("pmid", rec.get("pmid") or "")]
    body = "".join(f"  {k:<8}= {{{v}}},\n" for k, v in fields if v)
    return f"@article{{{key},\n{body}}}"


def assign_keys(records):
    """{pmid: citekey} in a deterministic order (first author, then year).

    Shared by the .bib writer and anything emitting \\cite, so a key never drifts
    between the bibliography and the citations pointing at it."""
    taken, keys = set(), {}
    for rec in sorted(records, key=lambda r: (surname(r).lower(), str(r.get("year") or ""))):
        keys[rec.get("pmid") or id(rec)] = cite_key(rec, taken)
    return keys


def write_bib(records, path, header=None):
    """Write a .bib in a deterministic order (first author, then year)."""
    keys = assign_keys(records)
    ordered = sorted(records, key=lambda r: (surname(r).lower(), str(r.get("year") or "")))
    out = [to_bibtex(rec, keys[rec.get("pmid") or id(rec)]) for rec in ordered]
    head = header or ("% Verified bibliographic records for the biomarker one-pagers.\n"
                      "% Generated from PubMed by fetch_pubmed_records.py -- do not hand-edit.\n\n")
    Path(path).write_text(head + "\n\n".join(out) + "\n", encoding="utf-8")
    return len(out)


# --------------------------------------------------------------- rendered reference

def format_authors(rec, max_authors=3):
    """Nature style: list up to `max_authors`, then 'et al.'; two authors joined by '&'."""
    a = _author_list(rec)
    if not a:
        return ""
    def one(x):
        ini = (x.get("initials") or "").strip()
        ini = ", " + " ".join(f"{c}." for c in ini) if ini else ""
        return f"{x['last']}{ini}"
    if len(a) == 1:
        return one(a[0])
    if len(a) <= max_authors:
        return ", ".join(one(x) for x in a[:-1]) + " & " + one(a[-1])
    return one(a[0]) + " et al."


def format_reference(rec, max_authors=3, include_pmid=True):
    """One rendered reference line, faithful to the BibTeX entry for the same record."""
    bits = []
    auth = format_authors(rec, max_authors)
    if auth:
        bits.append(auth + ("" if auth.endswith(".") else "."))
    title = (rec.get("title") or "").strip().rstrip(".")
    if title:
        bits.append(title + ".")
    jrn = (rec.get("journal_iso") or rec.get("journal_full") or "").strip().rstrip(".")
    vol, pages = str(rec.get("volume") or "").strip(), expand_pages(rec.get("pages"))
    year = str(rec.get("year") or "").strip()
    loc = jrn
    if vol:
        loc += f" **{vol}**"
    if pages:
        loc += f", {pages}"
    if year:
        loc += f" ({year})"
    if loc.strip():
        bits.append(loc.strip() + ".")
    if include_pmid and rec.get("pmid"):
        bits.append(f"PMID {rec['pmid']}.")
    return " ".join(bits)
