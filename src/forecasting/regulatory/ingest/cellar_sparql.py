"""CELLAR SPARQL client and query builder for EU legal acts.

This module builds SPARQL queries against the Publications Office CELLAR endpoint,
executes paginated requests with retries, and normalizes results to a tidy
pandas DataFrame suitable for downstream processing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd
import requests
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .parsing import (
    clean_text,
    eurlex_links_from_celex,
    extract_celex_from_uri,
    join_semicolon,
)


PREFIXES = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX dc:  <http://purl.org/dc/elements/1.1/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
""".strip()


@dataclass
class CellarQueryParams:
    """Parameters controlling the SPARQL query."""

    query_terms: List[str]
    eurovoc_uris: List[str]
    language: str
    date_from: str
    date_to: str
    limit: int
    offset: int


def _build_topic_filter(query_terms: List[str], eurovoc_uris: List[str]) -> str:
    """Builds the SPARQL snippet for topic filtering.

    - If `eurovoc_uris` is non-empty: returns a VALUES clause on ?eurovoc.
    - Else if `query_terms` is non-empty: returns a FILTER with CONTAINS checks on
      title and abstract.
    - Else: returns an empty string (no topic constraint).
    """
    if eurovoc_uris:
        uris = " ".join(f"<{u}>" for u in eurovoc_uris)
        return f"VALUES ?eurovoc {{ {uris} }}"
    terms = [t.strip().lower() for t in query_terms or [] if t and t.strip()]
    if terms:
        pieces = []
        for t in terms:
            t_json = json.dumps(t)  # safe quoting
            pieces.append(
                f"CONTAINS(LCASE(STR(?title)), {t_json}) || CONTAINS(LCASE(STR(?abstract)), {t_json})"
            )
        cond = " ||\n  ".join(pieces)
        return f"FILTER(\n  {cond}\n)"
    return ""


def build_query(
    query_terms: List[str],
    eurovoc_uris: List[str],
    language: str,
    date_from: str,
    date_to: str,
    limit: int,
    offset: int,
) -> str:
    """Builds a SPARQL query string for CELLAR legal resources.

    Args:
        query_terms: Keyword terms for title/abstract filtering (case-insensitive).
        eurovoc_uris: Optional EuroVoc concept URIs list; if provided, overrides
                      keyword filtering.
        language: CELLAR language code (e.g., "ENG", "ITA").
        date_from: Inclusive start date (YYYY-MM-DD).
        date_to: Inclusive end date (YYYY-MM-DD).
        limit: Page size.
        offset: Pagination offset.

    Returns:
        A SPARQL query string.
    """
    topic_filter = _build_topic_filter(query_terms, eurovoc_uris)

    # Base query adapted from the provided template. We omit ?item resolution here;
    # downstream links are derived from CELEX when available.
    query = f"""
{PREFIXES}

SELECT DISTINCT ?work ?celex ?title ?abstract ?date ?rtype ?eurovoc ?lang
WHERE {{
  ?work a cdm:legal-resource .
  ?work cdm:work_date_document ?date .
  OPTIONAL {{ ?work owl:sameAs ?celex . FILTER(CONTAINS(STR(?celex), "resource/celex/")) }}
  OPTIONAL {{
    ?expr cdm:expression_belongs_to_work ?work ;
          cdm:expression_uses_language ?langRes ;
          cdm:expression_title ?title .
    OPTIONAL {{ ?expr dcterms:abstract ?abstract . }}
    ?langRes dc:identifier ?lang .
    FILTER(STR(?lang) = {json.dumps(language)})
  }}
  OPTIONAL {{ ?work cdm:resource_legal_type ?rtype . }}
  OPTIONAL {{ ?work cdm:subject-matter ?eurovoc . }}

  FILTER ( ?date >= {json.dumps(date_from)}^^xsd:date && ?date <= {json.dumps(date_to)}^^xsd:date )

  {topic_filter}
}}
ORDER BY DESC(?date)
LIMIT {int(limit)}
OFFSET {int(offset)}
""".strip()
    return query


def _bindings_to_rows(bindings: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for b in bindings:
        def g(key: str) -> Optional[str]:
            v = b.get(key)
            return v.get("value") if isinstance(v, dict) else None

        rows.append(
            {
                "work": g("work"),
                "celex_uri": g("celex"),
                "title": g("title"),
                "abstract": g("abstract"),
                "date": g("date"),
                "rtype": g("rtype"),
                "eurovoc": g("eurovoc"),
                "lang": g("lang"),
            }
        )
    return rows


def run_page(endpoint: str, query: str, timeout_s: int) -> pd.DataFrame:
    """Executes a single SPARQL page and returns a DataFrame of raw results.

    Args:
        endpoint: CELLAR SPARQL endpoint URL.
        query: SPARQL query string.
        timeout_s: Request timeout in seconds.

    Returns:
        DataFrame with columns: work, celex_uri, title, abstract, date, rtype, eurovoc, lang
        Empty DataFrame on no results.
    """
    headers = {"Accept": "application/sparql-results+json"}
    resp = requests.post(endpoint, data={"query": query}, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    bindings = data.get("results", {}).get("bindings", [])
    rows = _bindings_to_rows(bindings)
    return pd.DataFrame(rows)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "work_uri",
                "celex",
                "title",
                "abstract",
                "date_document",
                "resource_type",
                "eurovoc",
                "link_html",
                "link_pdf",
                "language",
            ]
        )

    # Clean and transform columns
    df = df.copy()
    for col in ["title", "abstract", "rtype"]:
        if col in df.columns:
            df[col] = df[col].map(clean_text)

    # Group by work to aggregate multiple eurovoc/celex rows
    def first_non_null(series: pd.Series) -> Optional[str]:
        for v in series:
            if pd.notna(v) and v is not None and str(v) != "":
                return v
        return None

    agg = {
        "celex_uri": lambda s: first_non_null(s),
        "title": lambda s: first_non_null(s),
        "abstract": lambda s: first_non_null(s),
        "date": lambda s: first_non_null(s),
        "rtype": lambda s: first_non_null(s),
        "eurovoc": lambda s: join_semicolon([v for v in s if isinstance(v, str) and v]),
        "lang": lambda s: first_non_null(s),
    }

    df = df.groupby("work", as_index=False).agg(agg)

    # Extract CELEX code
    df["celex"] = df["celex_uri"].map(extract_celex_from_uri)

    # Derive links
    links = df.apply(lambda r: eurlex_links_from_celex(r.get("celex"), r.get("lang")), axis=1)
    df["link_html"] = [t[0] for t in links]
    df["link_pdf"] = [t[1] for t in links]

    # Rename columns to target schema
    df = df.rename(
        columns={
            "work": "work_uri",
            "date": "date_document",
            "rtype": "resource_type",
            "lang": "language",
        }
    )

    # Order columns
    cols = [
        "work_uri",
        "celex",
        "title",
        "abstract",
        "date_document",
        "resource_type",
        "eurovoc",
        "link_html",
        "link_pdf",
        "language",
    ]
    return df.reindex(columns=cols)


def fetch_all(
    endpoint: str,
    query_terms: List[str],
    eurovoc_uris: List[str],
    language: str,
    date_from: str,
    date_to: str,
    page_size: int,
    max_pages: int,
    timeout_s: int,
    retries: int,
    logger=None,
) -> pd.DataFrame:
    """Fetches all pages for the given parameters and returns a normalized DataFrame.

    Retries transient HTTP errors with exponential backoff.
    """
    all_pages: list[pd.DataFrame] = []
    for page in range(max_pages):
        offset = page * page_size
        q = build_query(
            query_terms=query_terms,
            eurovoc_uris=eurovoc_uris,
            language=language,
            date_from=date_from,
            date_to=date_to,
            limit=page_size,
            offset=offset,
        )
        if logger:
            snippet = q[:200].replace("\n", " ")
            logger.info(f"SPARQL page {page+1}: {snippet}...")

        try:
            for attempt in Retrying(
                stop=stop_after_attempt(max(1, int(retries))),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.HTTPError)),
                reraise=True,
            ):
                with attempt:
                    df_page = run_page(endpoint=endpoint, query=q, timeout_s=timeout_s)
        except RetryError as e:
            if logger:
                logger.warning(f"Page {page+1} failed after retries: {e}")
            break

        if df_page.empty:
            if logger:
                logger.info(f"No more results at page {page+1} (empty page). Stopping.")
            break

        all_pages.append(df_page)

        # If fewer than page_size rows, we've likely reached the end.
        if len(df_page) < page_size:
            if logger:
                logger.info(f"Last page reached at {page+1} (rows={len(df_page)}).")
            break

    if not all_pages:
        return _normalize(pd.DataFrame())

    concatenated = pd.concat(all_pages, ignore_index=True)
    result = _normalize(concatenated)
    return result

