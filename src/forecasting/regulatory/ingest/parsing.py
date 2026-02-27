"""Parsing and normalization helpers for regulatory ingestion.

Provides utilities to extract CELEX identifiers, normalize language codes,
join EuroVoc concepts, and clean text fields.
"""
from __future__ import annotations

from typing import Iterable, Optional


def extract_celex_from_uri(uri: Optional[str]) -> Optional[str]:
    """Extracts the CELEX code from a known CELEX URI.

    Examples of supported forms:
    - "http://publications.europa.eu/resource/celex/32019R1020" -> "32019R1020"

    Args:
        uri: A URI string or None.

    Returns:
        The CELEX code if extractable, else None.
    """
    if not uri:
        return None
    marker = "/resource/celex/"
    try:
        idx = uri.rfind(marker)
        if idx == -1:
            return None
        return uri[idx + len(marker) :]
    except Exception:
        return None


_LANG_3_TO_2 = {
    # Common CELLAR language codes mapped to EUR-Lex two-letter codes
    "BGU": "BG",  # Bulgarian (CELLAR may use BUL, older variants may exist)
    "BUL": "BG",
    "CES": "CS",
    "CZE": "CS",
    "DAN": "DA",
    "DEU": "DE",
    "GER": "DE",
    "ELL": "EL",
    "GRE": "EL",
    "ENG": "EN",
    "SPA": "ES",
    "EST": "ET",
    "FIN": "FI",
    "FRA": "FR",
    "FRE": "FR",
    "HRV": "HR",
    "HUN": "HU",
    "GLE": "GA",
    "GLE": "GA",
    "ITA": "IT",
    "LAV": "LV",
    "LIT": "LT",
    "MLT": "MT",
    "NLD": "NL",
    "DUT": "NL",
    "POL": "PL",
    "POR": "PT",
    "RON": "RO",
    "RUM": "RO",
    "SLK": "SK",
    "SLO": "SK",
    "SLV": "SL",
    "SWE": "SV",
}


def lang3_to_lang2(lang3: Optional[str]) -> str:
    """Converts a 3-letter CELLAR language code to a 2-letter EUR-Lex code.

    Falls back to "EN" if unknown or None.
    """
    if not lang3:
        return "EN"
    return _LANG_3_TO_2.get(lang3.upper(), "EN")


def join_semicolon(values: Iterable[Optional[str]]) -> str:
    """Joins unique, non-empty values with "; " preserving insertion order."""
    seen = []
    added = set()
    for v in values:
        if not v:
            continue
        if v not in added:
            added.add(v)
            seen.append(v)
    return "; ".join(seen)


def clean_text(text: Optional[str]) -> Optional[str]:
    """Normalizes text fields by stripping whitespace and collapsing blanks."""
    if text is None:
        return None
    t = text.strip()
    return t if t else None


def eurlex_links_from_celex(celex: Optional[str], lang3: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Builds EUR-Lex HTML and PDF links for a given CELEX and language.

    Args:
        celex: CELEX code like "32019R1020".
        lang3: CELLAR language code like "ENG" (mapped to EN for link).

    Returns:
        (link_html, link_pdf) where each may be None if celex is missing.
    """
    if not celex:
        return None, None
    lang2 = lang3_to_lang2(lang3)
    base = f"https://eur-lex.europa.eu/legal-content/{lang2}/TXT/"
    link_html = f"{base}?uri=CELEX:{celex}"
    link_pdf = f"{base}PDF/?uri=CELEX:{celex}"
    return link_html, link_pdf

