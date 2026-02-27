"""Tests for regulatory.ingest.cellar_sparql module.

Tests SPARQL query building, execution, pagination, normalization,
and retry logic with mocked HTTP responses.
"""
from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import requests

from src.forecasting.regulatory.ingest.cellar_sparql import (
    PREFIXES,
    CellarQueryParams,
    _bindings_to_rows,
    _build_topic_filter,
    _normalize,
    build_query,
    fetch_all,
    run_page,
)
from tests.regulatory.fixtures.sample_responses import (
    EXPECTED_NORMALIZED_COLUMNS,
    SPARQL_RESPONSE_EMPTY,
    SPARQL_RESPONSE_MULTILANG,
    SPARQL_RESPONSE_SAMPLE,
    SPARQL_RESPONSE_WITH_NULLS,
    TEST_EUROVOC_URIS,
    TEST_QUERY_TERMS,
)


class TestBuildTopicFilter:
    """Tests for SPARQL topic filter construction."""

    def test_build_topic_filter_with_eurovoc_uris(self):
        """Test topic filter with EuroVoc URIs (takes precedence)."""
        eurovoc_uris = [
            "http://eurovoc.europa.eu/1234",
            "http://eurovoc.europa.eu/5678",
        ]
        query_terms = ["ignored", "terms"]

        result = _build_topic_filter(query_terms, eurovoc_uris)

        assert "VALUES ?eurovoc" in result
        assert "http://eurovoc.europa.eu/1234" in result
        assert "http://eurovoc.europa.eu/5678" in result
        # Query terms should be ignored when eurovoc_uris provided
        assert "CONTAINS" not in result

    def test_build_topic_filter_with_query_terms(self):
        """Test topic filter with query terms (when no eurovoc_uris)."""
        query_terms = ["artificial intelligence", "machine learning"]
        eurovoc_uris = []

        result = _build_topic_filter(query_terms, eurovoc_uris)

        assert "FILTER" in result
        assert "CONTAINS" in result
        assert "LCASE" in result
        assert "artificial intelligence" in result
        assert "machine learning" in result
        assert "?title" in result
        assert "?abstract" in result

    def test_build_topic_filter_with_no_filters(self):
        """Test topic filter returns empty string when no filters."""
        result = _build_topic_filter([], [])
        assert result == ""

    def test_build_topic_filter_strips_whitespace(self):
        """Test topic filter strips whitespace from query terms."""
        query_terms = ["  term1  ", "term2  ", "  term3"]
        result = _build_topic_filter(query_terms, [])

        assert "term1" in result
        assert "term2" in result
        assert "term3" in result

    def test_build_topic_filter_filters_empty_terms(self):
        """Test topic filter ignores empty/None query terms."""
        query_terms = ["valid", "", None, "  ", "another"]
        result = _build_topic_filter(query_terms, [])

        assert "valid" in result
        assert "another" in result
        # Should only have 2 CONTAINS clauses
        assert result.count("CONTAINS") == 4  # 2 terms * 2 fields (title + abstract)

    def test_build_topic_filter_handles_special_characters(self):
        """Test topic filter properly escapes special characters."""
        query_terms = ['term with "quotes"', "term's apostrophe"]
        result = _build_topic_filter(query_terms, [])

        # JSON dumps should properly escape
        assert "CONTAINS" in result
        assert "LCASE" in result


class TestBuildQuery:
    """Tests for SPARQL query construction."""

    def test_build_query_basic_structure(self):
        """Test built query has correct basic structure."""
        query = build_query(
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            limit=100,
            offset=0,
        )

        assert PREFIXES in query
        assert "SELECT DISTINCT" in query
        assert "?work" in query
        assert "?celex" in query
        assert "?title" in query
        assert "cdm:resource_legal" in query
        assert "LIMIT 100" in query
        assert "OFFSET 0" in query

    def test_build_query_with_eurovoc_filter(self):
        """Test query includes VALUES clause for eurovoc_uris."""
        query = build_query(
            query_terms=[],
            eurovoc_uris=TEST_EUROVOC_URIS,
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            limit=100,
            offset=0,
        )

        assert "VALUES ?eurovoc" in query
        for uri in TEST_EUROVOC_URIS:
            assert uri in query

    def test_build_query_with_query_terms_filter(self):
        """Test query includes FILTER clause for query_terms."""
        query = build_query(
            query_terms=TEST_QUERY_TERMS,
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            limit=100,
            offset=0,
        )

        assert "FILTER" in query
        assert "CONTAINS" in query
        for term in TEST_QUERY_TERMS:
            assert term in query

    def test_build_query_with_language_filter(self):
        """Test query includes language filter."""
        query = build_query(
            query_terms=[],
            eurovoc_uris=[],
            language="FRA",
            date_from="2020-01-01",
            date_to="2020-12-31",
            limit=100,
            offset=0,
        )

        assert '"FRA"' in query
        assert "?lang" in query

    def test_build_query_with_date_range(self):
        """Test query includes date range filter."""
        query = build_query(
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2019-01-01",
            date_to="2021-12-31",
            limit=50,
            offset=100,
        )

        assert "2019-01-01" in query
        assert "2021-12-31" in query
        assert "?date >=" in query
        assert "?date <=" in query
        assert "xsd:date" in query

    def test_build_query_pagination(self):
        """Test query uses correct LIMIT and OFFSET."""
        # Page 1
        query1 = build_query(
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            limit=100,
            offset=0,
        )
        assert "LIMIT 100" in query1
        assert "OFFSET 0" in query1

        # Page 2
        query2 = build_query(
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            limit=100,
            offset=100,
        )
        assert "LIMIT 100" in query2
        assert "OFFSET 100" in query2

    def test_build_query_orders_by_date(self):
        """Test query orders results by date descending."""
        query = build_query(
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            limit=100,
            offset=0,
        )

        assert "ORDER BY DESC(?date)" in query


class TestBindingsToRows:
    """Tests for SPARQL bindings conversion to rows."""

    def test_bindings_to_rows_converts_sample_response(self):
        """Test conversion of sample SPARQL bindings."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)

        assert len(rows) == 3
        assert all("work" in row for row in rows)
        assert all("celex_uri" in row for row in rows)

    def test_bindings_to_rows_extracts_values(self):
        """Test extraction of actual values from bindings."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"][:1]
        rows = _bindings_to_rows(bindings)

        row = rows[0]
        assert row["work"] == "http://publications.europa.eu/resource/cellar/abc123"
        assert row["celex_uri"] == "http://publications.europa.eu/resource/celex/32019R1020"
        assert row["title"] == "Regulation on Market Surveillance"
        assert row["abstract"] == "This regulation establishes rules..."
        assert row["lang"] == "ENG"

    def test_bindings_to_rows_handles_missing_fields(self):
        """Test conversion handles missing optional fields."""
        bindings = SPARQL_RESPONSE_WITH_NULLS["results"]["bindings"]
        rows = _bindings_to_rows(bindings)

        row = rows[0]
        assert row["work"] == "http://publications.europa.eu/resource/cellar/xyz789"
        assert row["celex_uri"] is None  # Missing
        assert row["abstract"] is None  # Missing
        assert row["rtype"] is None  # Missing
        assert row["eurovoc"] is None  # Missing

    def test_bindings_to_rows_handles_empty_bindings(self):
        """Test conversion handles empty bindings list."""
        rows = _bindings_to_rows([])
        assert rows == []

    def test_bindings_to_rows_handles_none_value(self):
        """Test conversion handles None values in bindings."""
        bindings = [
            {
                "work": {"type": "uri", "value": "http://example.com/work1"},
                "celex": {"type": "uri", "value": None},
                "title": None,
            }
        ]
        rows = _bindings_to_rows(bindings)

        assert len(rows) == 1
        assert rows[0]["celex_uri"] is None
        assert rows[0]["title"] is None


class TestRunPage:
    """Tests for single SPARQL page execution."""

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.requests.post")
    def test_run_page_success(self, mock_post):
        """Test successful SPARQL page execution."""
        mock_response = Mock()
        mock_response.json.return_value = SPARQL_RESPONSE_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        query = "SELECT * WHERE { ?s ?p ?o }"
        df = run_page(endpoint="http://example.com/sparql", query=query, timeout_s=30)

        assert not df.empty
        assert len(df) == 3
        assert "work" in df.columns
        assert "celex_uri" in df.columns
        mock_post.assert_called_once()

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.requests.post")
    def test_run_page_empty_results(self, mock_post):
        """Test SPARQL page execution with empty results."""
        mock_response = Mock()
        mock_response.json.return_value = SPARQL_RESPONSE_EMPTY
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        query = "SELECT * WHERE { ?s ?p ?o }"
        df = run_page(endpoint="http://example.com/sparql", query=query, timeout_s=30)

        assert df.empty

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.requests.post")
    def test_run_page_uses_correct_headers(self, mock_post):
        """Test run_page uses correct Accept header."""
        mock_response = Mock()
        mock_response.json.return_value = SPARQL_RESPONSE_EMPTY
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        query = "SELECT * WHERE { ?s ?p ?o }"
        run_page(endpoint="http://example.com/sparql", query=query, timeout_s=30)

        call_args = mock_post.call_args
        assert call_args[1]["headers"]["Accept"] == "application/sparql-results+json"

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.requests.post")
    def test_run_page_timeout_parameter(self, mock_post):
        """Test run_page passes timeout parameter."""
        mock_response = Mock()
        mock_response.json.return_value = SPARQL_RESPONSE_EMPTY
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        query = "SELECT * WHERE { ?s ?p ?o }"
        run_page(endpoint="http://example.com/sparql", query=query, timeout_s=60)

        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 60

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.requests.post")
    def test_run_page_handles_http_error(self, mock_post):
        """Test run_page raises on HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        query = "SELECT * WHERE { ?s ?p ?o }"

        with pytest.raises(requests.HTTPError):
            run_page(endpoint="http://example.com/sparql", query=query, timeout_s=30)


class TestNormalize:
    """Tests for DataFrame normalization."""

    def test_normalize_empty_dataframe(self):
        """Test normalize returns correct schema for empty DataFrame."""
        df = pd.DataFrame()
        result = _normalize(df)

        assert result.empty
        assert list(result.columns) == EXPECTED_NORMALIZED_COLUMNS

    def test_normalize_sample_data(self):
        """Test normalize processes sample data correctly."""
        # Create DataFrame from sample bindings
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df = pd.DataFrame(rows)

        result = _normalize(df)

        assert not result.empty
        assert list(result.columns) == EXPECTED_NORMALIZED_COLUMNS
        # Should have 2 unique works (abc123 appears twice)
        assert len(result) == 2

    def test_normalize_groups_by_work(self):
        """Test normalize groups multiple rows by work URI."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df = pd.DataFrame(rows)

        result = _normalize(df)

        # Work abc123 appears twice with different eurovoc
        work_abc123 = result[result["work_uri"] == "http://publications.europa.eu/resource/cellar/abc123"]
        assert len(work_abc123) == 1

        # Should have both eurovoc URIs joined
        eurovoc = work_abc123.iloc[0]["eurovoc"]
        assert "http://eurovoc.europa.eu/4512" in eurovoc
        assert "http://eurovoc.europa.eu/9876" in eurovoc
        assert ";" in eurovoc

    def test_normalize_extracts_celex(self):
        """Test normalize extracts CELEX code from URI."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"][:1]
        rows = _bindings_to_rows(bindings)
        df = pd.DataFrame(rows)

        result = _normalize(df)

        assert result.iloc[0]["celex"] == "32019R1020"

    def test_normalize_generates_links(self):
        """Test normalize generates EUR-Lex links."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"][:1]
        rows = _bindings_to_rows(bindings)
        df = pd.DataFrame(rows)

        result = _normalize(df)

        link_html = result.iloc[0]["link_html"]
        link_pdf = result.iloc[0]["link_pdf"]

        assert link_html is not None
        assert "32019R1020" in link_html
        assert "eur-lex.europa.eu" in link_html
        assert link_pdf is not None
        assert "PDF" in link_pdf

    def test_normalize_cleans_text(self):
        """Test normalize cleans text fields."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"][1:2]
        rows = _bindings_to_rows(bindings)
        df = pd.DataFrame(rows)

        result = _normalize(df)

        # Title has leading/trailing spaces in sample data
        title = result.iloc[0]["title"]
        assert title == "Directive on Tax Reporting"  # Stripped

    def test_normalize_handles_null_fields(self):
        """Test normalize handles NULL fields correctly."""
        bindings = SPARQL_RESPONSE_WITH_NULLS["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df = pd.DataFrame(rows)

        result = _normalize(df)

        assert len(result) == 1
        row = result.iloc[0]

        assert row["celex"] is None
        assert row["title"] is None
        assert row["abstract"] is None
        assert row["resource_type"] is None
        assert row["link_html"] is None  # No celex, no link
        assert row["link_pdf"] is None

    def test_normalize_renames_columns(self):
        """Test normalize renames columns to target schema."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"][:1]
        rows = _bindings_to_rows(bindings)
        df = pd.DataFrame(rows)

        result = _normalize(df)

        assert "work_uri" in result.columns
        assert "date_document" in result.columns
        assert "resource_type" in result.columns
        assert "language" in result.columns

        # Old names should not be present
        assert "work" not in result.columns
        assert "date" not in result.columns
        assert "rtype" not in result.columns
        assert "lang" not in result.columns


class TestFetchAll:
    """Tests for multi-page fetching with pagination and retries."""

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_fetch_all_single_page(self, mock_run_page):
        """Test fetch_all with single page of results."""
        # Return 50 rows (less than page_size)
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df_page = pd.DataFrame(rows)
        mock_run_page.return_value = df_page

        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=10,
            timeout_s=30,
            retries=3,
        )

        assert not result.empty
        mock_run_page.assert_called_once()

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_fetch_all_multiple_pages(self, mock_run_page):
        """Test fetch_all with multiple pages."""
        # First page: full page_size
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df_page1 = pd.DataFrame(rows * 34)  # 102 rows

        # Second page: partial
        df_page2 = pd.DataFrame(rows)

        mock_run_page.side_effect = [df_page1, df_page2]

        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=10,
            timeout_s=30,
            retries=3,
        )

        assert not result.empty
        assert mock_run_page.call_count == 2

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_fetch_all_stops_on_empty_page(self, mock_run_page):
        """Test fetch_all stops when encountering empty page."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df_page = pd.DataFrame(rows)
        df_empty = pd.DataFrame()

        mock_run_page.side_effect = [df_page, df_empty]

        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=10,
            timeout_s=30,
            retries=3,
        )

        assert not result.empty
        assert mock_run_page.call_count == 2

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_fetch_all_respects_max_pages(self, mock_run_page):
        """Test fetch_all respects max_pages limit."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df_page = pd.DataFrame(rows * 34)  # Always full page

        mock_run_page.return_value = df_page

        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=3,
            timeout_s=30,
            retries=3,
        )

        assert not result.empty
        assert mock_run_page.call_count == 3

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_fetch_all_handles_no_results(self, mock_run_page):
        """Test fetch_all handles no results gracefully."""
        df_empty = pd.DataFrame()
        mock_run_page.return_value = df_empty

        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=10,
            timeout_s=30,
            retries=3,
        )

        assert result.empty
        assert list(result.columns) == EXPECTED_NORMALIZED_COLUMNS

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_fetch_all_retries_on_timeout(self, mock_run_page):
        """Test fetch_all retries on timeout errors."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df_page = pd.DataFrame(rows)

        # Fail twice, then succeed
        mock_run_page.side_effect = [
            requests.Timeout("Timeout"),
            requests.Timeout("Timeout"),
            df_page,
        ]

        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=10,
            timeout_s=30,
            retries=3,
        )

        assert not result.empty
        assert mock_run_page.call_count == 3

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_fetch_all_stops_after_max_retries(self, mock_run_page):
        """Test fetch_all stops after exhausting retries."""
        # Always fail
        mock_run_page.side_effect = requests.Timeout("Timeout")

        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=10,
            timeout_s=30,
            retries=3,
        )

        # Should return empty normalized DataFrame
        assert result.empty
        assert list(result.columns) == EXPECTED_NORMALIZED_COLUMNS

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_fetch_all_uses_logger(self, mock_run_page):
        """Test fetch_all uses logger when provided."""
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df_page = pd.DataFrame(rows)
        mock_run_page.return_value = df_page

        mock_logger = MagicMock()

        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=[],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=10,
            timeout_s=30,
            retries=3,
            logger=mock_logger,
        )

        assert not result.empty
        # Logger should have been called
        assert mock_logger.info.called
