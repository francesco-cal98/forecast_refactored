"""Tests for regulatory.ingest.parsing module.

Tests parsing helpers: CELEX extraction, language conversion,
text cleaning, and link generation.
"""
from __future__ import annotations

import pytest

from src.forecasting.regulatory.ingest.parsing import (
    clean_text,
    eurlex_links_from_celex,
    extract_celex_from_uri,
    join_semicolon,
    lang3_to_lang2,
)
from tests.regulatory.fixtures.sample_responses import (
    TEST_CELEX_CODES,
    TEST_LANGUAGE_MAPPINGS,
)


class TestExtractCelexFromUri:
    """Tests for CELEX extraction from URIs."""

    @pytest.mark.parametrize("uri,expected", TEST_CELEX_CODES)
    def test_extract_celex_from_various_uris(self, uri, expected):
        """Test CELEX extraction from different URI formats."""
        result = extract_celex_from_uri(uri)
        assert result == expected

    def test_extract_celex_with_trailing_slash(self):
        """Test CELEX extraction handles trailing slashes."""
        uri = "http://publications.europa.eu/resource/celex/32019R1020/"
        result = extract_celex_from_uri(uri)
        assert result == "32019R1020/"

    def test_extract_celex_from_malformed_uri(self):
        """Test CELEX extraction returns None for malformed URIs."""
        uris = [
            "http://example.com/some/path",
            "celex/32019R1020",  # missing host
            "resource/celex/",  # no code
        ]
        for uri in uris:
            assert extract_celex_from_uri(uri) is None

    def test_extract_celex_handles_exception(self):
        """Test CELEX extraction handles exceptions gracefully."""
        # Pass an object that might cause issues
        result = extract_celex_from_uri(12345)
        assert result is None


class TestLang3ToLang2:
    """Tests for 3-letter to 2-letter language code conversion."""

    @pytest.mark.parametrize("lang3,expected", TEST_LANGUAGE_MAPPINGS)
    def test_language_conversion(self, lang3, expected):
        """Test conversion of various 3-letter language codes."""
        result = lang3_to_lang2(lang3)
        assert result == expected

    def test_language_conversion_case_insensitive(self):
        """Test language conversion is case-insensitive."""
        assert lang3_to_lang2("eng") == "EN"
        assert lang3_to_lang2("ENG") == "EN"
        assert lang3_to_lang2("Eng") == "EN"

    def test_unknown_language_defaults_to_english(self):
        """Test unknown language codes default to EN."""
        unknown_codes = ["XYZ", "ABC", "FOO"]
        for code in unknown_codes:
            assert lang3_to_lang2(code) == "EN"

    def test_empty_language_defaults_to_english(self):
        """Test empty/None language codes default to EN."""
        assert lang3_to_lang2(None) == "EN"
        assert lang3_to_lang2("") == "EN"

    def test_all_eu_languages_mapped(self):
        """Test all major EU language codes are mapped."""
        eu_langs = {
            "BUL": "BG",
            "CZE": "CS",
            "DAN": "DA",
            "DEU": "DE",
            "ELL": "EL",
            "ENG": "EN",
            "SPA": "ES",
            "EST": "ET",
            "FIN": "FI",
            "FRA": "FR",
            "HRV": "HR",
            "HUN": "HU",
            "GLE": "GA",
            "ITA": "IT",
            "LAV": "LV",
            "LIT": "LT",
            "MLT": "MT",
            "NLD": "NL",
            "POL": "PL",
            "POR": "PT",
            "RON": "RO",
            "SLK": "SK",
            "SLV": "SL",
            "SWE": "SV",
        }
        for lang3, expected_lang2 in eu_langs.items():
            assert lang3_to_lang2(lang3) == expected_lang2


class TestJoinSemicolon:
    """Tests for semicolon-separated value joining."""

    def test_join_simple_list(self):
        """Test joining simple list of strings."""
        values = ["apple", "banana", "cherry"]
        result = join_semicolon(values)
        assert result == "apple; banana; cherry"

    def test_join_removes_duplicates(self):
        """Test joining removes duplicates while preserving order."""
        values = ["apple", "banana", "apple", "cherry", "banana"]
        result = join_semicolon(values)
        assert result == "apple; banana; cherry"

    def test_join_filters_none_and_empty(self):
        """Test joining filters out None and empty strings."""
        values = ["apple", None, "", "banana", None, "cherry", ""]
        result = join_semicolon(values)
        assert result == "apple; banana; cherry"

    def test_join_empty_list(self):
        """Test joining empty list returns empty string."""
        assert join_semicolon([]) == ""
        assert join_semicolon([None, None, ""]) == ""

    def test_join_preserves_insertion_order(self):
        """Test joining preserves insertion order for unique values."""
        values = ["zebra", "apple", "mango", "banana"]
        result = join_semicolon(values)
        assert result == "zebra; apple; mango; banana"

    def test_join_handles_uris(self):
        """Test joining works with URIs."""
        uris = [
            "http://eurovoc.europa.eu/1234",
            "http://eurovoc.europa.eu/5678",
            "http://eurovoc.europa.eu/1234",  # duplicate
        ]
        result = join_semicolon(uris)
        assert result == "http://eurovoc.europa.eu/1234; http://eurovoc.europa.eu/5678"


class TestCleanText:
    """Tests for text cleaning and normalization."""

    def test_clean_text_strips_whitespace(self):
        """Test cleaning strips leading/trailing whitespace."""
        assert clean_text("  hello world  ") == "hello world"
        assert clean_text("\thello\n") == "hello"
        assert clean_text("  \n  text  \t  ") == "text"

    def test_clean_text_returns_none_for_empty(self):
        """Test cleaning returns None for empty/whitespace-only strings."""
        assert clean_text("") is None
        assert clean_text("   ") is None
        assert clean_text("\t\n  ") is None

    def test_clean_text_returns_none_for_none_input(self):
        """Test cleaning returns None for None input."""
        assert clean_text(None) is None

    def test_clean_text_preserves_internal_whitespace(self):
        """Test cleaning preserves internal whitespace structure."""
        assert clean_text("  hello   world  ") == "hello   world"
        assert clean_text("  first\nsecond  ") == "first\nsecond"


class TestEurlexLinksFromCelex:
    """Tests for EUR-Lex link generation from CELEX codes."""

    def test_eurlex_links_with_valid_celex(self):
        """Test link generation with valid CELEX and language."""
        celex = "32019R1020"
        lang3 = "ENG"
        html, pdf = eurlex_links_from_celex(celex, lang3)

        assert html == "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R1020"
        assert pdf == "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019R1020"

    def test_eurlex_links_with_different_languages(self):
        """Test link generation uses correct language code."""
        celex = "32020L0262"

        # French
        html, pdf = eurlex_links_from_celex(celex, "FRA")
        assert "FR/TXT/" in html
        assert "FR/TXT/PDF/" in pdf

        # German
        html, pdf = eurlex_links_from_celex(celex, "DEU")
        assert "DE/TXT/" in html
        assert "DE/TXT/PDF/" in pdf

        # Italian
        html, pdf = eurlex_links_from_celex(celex, "ITA")
        assert "IT/TXT/" in html
        assert "IT/TXT/PDF/" in pdf

    def test_eurlex_links_with_none_celex(self):
        """Test link generation returns None for missing CELEX."""
        html, pdf = eurlex_links_from_celex(None, "ENG")
        assert html is None
        assert pdf is None

    def test_eurlex_links_with_empty_celex(self):
        """Test link generation returns None for empty CELEX."""
        html, pdf = eurlex_links_from_celex("", "ENG")
        assert html is None
        assert pdf is None

    def test_eurlex_links_with_none_language(self):
        """Test link generation defaults to EN for None language."""
        celex = "32019R1020"
        html, pdf = eurlex_links_from_celex(celex, None)

        assert "EN/TXT/" in html
        assert "EN/TXT/PDF/" in pdf

    def test_eurlex_links_with_unknown_language(self):
        """Test link generation defaults to EN for unknown language."""
        celex = "32019R1020"
        html, pdf = eurlex_links_from_celex(celex, "UNKNOWN")

        assert "EN/TXT/" in html
        assert "EN/TXT/PDF/" in pdf

    def test_eurlex_links_format(self):
        """Test generated links have correct format."""
        celex = "52021PC0001"
        html, pdf = eurlex_links_from_celex(celex, "ENG")

        # Check URL structure
        assert html.startswith("https://eur-lex.europa.eu/legal-content/")
        assert "?uri=CELEX:" in html
        assert html.endswith(celex)

        assert pdf.startswith("https://eur-lex.europa.eu/legal-content/")
        assert "PDF/?uri=CELEX:" in pdf
        assert pdf.endswith(celex)
