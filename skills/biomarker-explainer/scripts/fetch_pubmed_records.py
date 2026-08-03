#!/usr/bin/env python3
"""Resolve every PMID cited by the literature-review workflow into an authoritative record.

Reads a workflow output JSON, collects the PMIDs actually cited, fetches each from
PubMed (NCBI efetch), and writes the cache that `citations.py` reads. Records already
in the cache are kept unless --refresh is given, so re-running is cheap.

    python fetch_pubmed_records.py --workflow-output workflow_result.json
    python fetch_pubmed_records.py --workflow-output ... --bib out/refs.bib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from citations import cache_path, load_records, write_bib  # noqa: E402

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def cited_pmids(workflow_output):
    """The PMIDs that actually reach a reference list: first citation of each component."""
    res = json.load(open(workflow_output))
    results = res.get("results") if isinstance(res, dict) else res
    pmids = []
    for r in results or []:
        for c in r.get("components") or []:
            research, verify = c.get("research") or {}, c.get("verify") or {}
            vcs = [x for x in (verify.get("verified_citations") or [])
                   if x.get("exists") and x.get("supports_claim")]
            for cite in (vcs or (research.get("citations") or [])):
                p = (cite.get("pmid") or "").strip()
                if p and p not in pmids:
                    pmids.append(p)
                break
    return pmids


def _text(node, path, default=""):
    el = node.find(path)
    return (el.text or default) if el is not None else default


def parse_article(art):
    """Parse one <PubmedArticle>.

    Every lookup below is scoped to an explicit path rather than iter(). A PubmedArticle
    embeds the article's own bibliography under PubmedData/ReferenceList, and those
    <Reference> entries carry their own <ArticleId IdType="doi">. An unscoped iter()
    therefore happily returns a cited paper's DOI as if it were this article's.
    """
    citation = art.find("MedlineCitation")
    article = citation.find("Article")
    journal = article.find("Journal")
    title_el = article.find("ArticleTitle")

    authors = []
    for a in article.findall("AuthorList/Author"):
        last, ini = a.findtext("LastName"), a.findtext("Initials")
        collective = a.findtext("CollectiveName")
        if last:
            authors.append({"last": last, "initials": ini or ""})
        elif collective:
            authors.append({"last": collective, "initials": ""})

    elocs = {e.get("EIdType"): (e.text or "").strip() for e in article.findall("ELocationID")}
    doi = elocs.get("doi", "")
    if not doi:
        ids = art.find("PubmedData/ArticleIdList")
        for aid in (ids.findall("ArticleId") if ids is not None else []):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break

    # Electronic-only articles have no page range; their article number stands in for one.
    pages = (article.findtext("Pagination/MedlinePgn")
             or article.findtext("Pagination/StartPage")
             or elocs.get("pii", ""))

    pubtypes = [e.text for e in article.findall("PublicationTypeList/PublicationType") if e.text]
    # A retracted paper is flagged either by its own type or by a RetractionIn back-pointer.
    retracted = (any("Retract" in p for p in pubtypes)
                 or any(c.get("RefType") == "RetractionIn"
                        for c in citation.findall("CommentsCorrectionsList/CommentsCorrections")))

    year = _text(journal, "JournalIssue/PubDate/Year") \
        or _text(journal, "JournalIssue/PubDate/MedlineDate")[:4]
    return {
        "pmid": citation.findtext("PMID") or "",
        # itertext() keeps text inside <i>/<sup> markup that PubMed embeds in titles
        "title": ("".join(title_el.itertext()).strip().rstrip(".") if title_el is not None else ""),
        "journal_full": _text(journal, "Title"),
        "journal_iso": _text(journal, "ISOAbbreviation"),
        "year": year,
        "volume": _text(journal, "JournalIssue/Volume"),
        "issue": _text(journal, "JournalIssue/Issue"),
        "pages": pages,
        "doi": doi,
        "authors": authors,
        "pubtypes": pubtypes,
        "retracted": retracted,
    }


def fetch(pmids, email, batch=50, pause=0.4):
    out = {}
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i + batch]
        params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml",
                  "tool": "biomarker-explainer"}
        if email:
            params["email"] = email
        q = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{EFETCH}?{q}", timeout=90) as f:
            root = ET.fromstring(f.read().decode("utf-8"))
        for art in root.iter("PubmedArticle"):
            rec = parse_article(art)
            if rec["pmid"]:
                out[rec["pmid"]] = rec
        time.sleep(pause)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow-output", help="workflow_result.json to harvest PMIDs from")
    ap.add_argument("--pmids", nargs="*", default=[], help="extra PMIDs to resolve")
    ap.add_argument("--cache", help="cache path (default: <skill>/data/pubmed_records.json)")
    ap.add_argument("--bib", help="also write a .bib of the whole cache here")
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL", ""),
                    help="contact address sent to NCBI (or set NCBI_EMAIL); optional, but "
                         "unidentified callers are rate-limited more aggressively")
    ap.add_argument("--refresh", action="store_true", help="re-fetch PMIDs already cached")
    args = ap.parse_args()

    wanted = list(args.pmids)
    if args.workflow_output:
        wanted = cited_pmids(args.workflow_output) + [p for p in wanted
                                                      if p not in cited_pmids(args.workflow_output)]
    if not wanted:
        sys.exit("nothing to fetch: pass --workflow-output and/or --pmids")

    cache = load_records(args.cache)
    todo = wanted if args.refresh else [p for p in wanted if p not in cache]
    print(f"{len(wanted)} cited PMIDs; {len(todo)} to fetch, {len(wanted) - len(todo)} cached")

    if todo:
        got = fetch(todo, args.email)
        missing = [p for p in todo if p not in got]
        if missing:
            print(f"WARNING: {len(missing)} PMIDs did not resolve: {', '.join(missing)}", file=sys.stderr)
        cache.update(got)

    out = cache_path(args.cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(cache)} records -> {out}")

    retracted = [p for p, r in cache.items() if r.get("retracted")]
    if retracted:
        print(f"RETRACTED, do not cite: {', '.join(retracted)}", file=sys.stderr)

    if args.bib:
        n = write_bib([cache[p] for p in wanted if p in cache], args.bib)
        print(f"wrote {n} BibTeX entries -> {args.bib}")


if __name__ == "__main__":
    main()
