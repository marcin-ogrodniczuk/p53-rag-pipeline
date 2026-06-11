"""Fetch p53 paper abstracts from PubMed via NCBI Entrez E-utilities.

Uses esearch to get PMIDs, then efetch to pull titles + abstracts + metadata.
No external deps beyond `requests`. Saves results to data/papers.json.
"""

import json
import time
import xml.etree.ElementTree as ET

import requests

import config

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _common_params():
    params = {"tool": config.ENTREZ_TOOL, "email": config.ENTREZ_EMAIL}
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    return params


def search_pmids(query: str, retmax: int) -> list[str]:
    """Return a list of PMIDs matching the query, restricted to papers with abstracts."""
    params = _common_params()
    params.update({
        "db": "pubmed",
        "term": f"{query} AND hasabstract[filter]",
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    })
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def _text(elem) -> str:
    """Flatten an element's text content, including nested tags."""
    return "".join(elem.itertext()).strip() if elem is not None else ""


def fetch_details(pmids: list[str]) -> list[dict]:
    """Fetch article metadata + abstracts for a batch of PMIDs."""
    papers = []
    batch_size = 50
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = _common_params()
        params.update({
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        })
        r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        for art in root.findall(".//PubmedArticle"):
            pmid = _text(art.find(".//PMID"))
            title = _text(art.find(".//ArticleTitle"))

            # Abstracts can be split into labeled sections.
            abstract_parts = []
            for ab in art.findall(".//Abstract/AbstractText"):
                label = ab.get("Label")
                txt = _text(ab)
                abstract_parts.append(f"{label}: {txt}" if label else txt)
            abstract = "\n".join(p for p in abstract_parts if p)

            journal = _text(art.find(".//Journal/Title"))
            year = _text(art.find(".//PubDate/Year")) or _text(art.find(".//PubDate/MedlineDate"))

            authors = []
            for a in art.findall(".//AuthorList/Author"):
                last = _text(a.find("LastName"))
                initials = _text(a.find("Initials"))
                if last:
                    authors.append(f"{last} {initials}".strip())

            if not abstract:
                continue

            papers.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": authors,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        print(f"  fetched {len(papers)} papers so far...")
        time.sleep(0.4)  # be polite to NCBI (<3 req/s without API key)

    return papers


def main():
    print(f"Searching PubMed for: {config.PUBMED_QUERY!r} (target {config.NUM_PAPERS} papers)")
    pmids = search_pmids(config.PUBMED_QUERY, config.NUM_PAPERS)
    print(f"Found {len(pmids)} PMIDs. Fetching details...")

    papers = fetch_details(pmids)
    print(f"Collected {len(papers)} papers with abstracts.")

    config.PAPERS_FILE.write_text(json.dumps(papers, indent=2, ensure_ascii=False))
    print(f"Saved to {config.PAPERS_FILE}")


if __name__ == "__main__":
    main()
