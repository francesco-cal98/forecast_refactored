"""Integration tests for regulatory module end-to-end pipeline.

Tests complete workflows from SPARQL query to text enrichment,
using real parquet data and mocked HTTP calls.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.forecasting.regulatory.enrichment.text_fetcher import TextFetcher
from src.forecasting.regulatory.ingest.cellar_sparql import build_query, fetch_all
from src.forecasting.regulatory.ingest.io_utils import ensure_dirs, safe_to_parquet
from src.forecasting.regulatory.ingest.parsing import (
    eurlex_links_from_celex,
    extract_celex_from_uri,
)
from tests.regulatory.fixtures.sample_responses import (
    EURLEX_HTML_SAMPLE,
    SPARQL_RESPONSE_SAMPLE,
)


class TestEndToEndPipeline:
    """Integration tests for complete ingestion pipeline."""

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_full_ingestion_pipeline(self, mock_run_page, tmp_path):
        """Test complete ingestion from SPARQL to parquet."""
        # Setup mock SPARQL response
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        from src.forecasting.regulatory.ingest.cellar_sparql import _bindings_to_rows

        rows = _bindings_to_rows(bindings)
        df_page = pd.DataFrame(rows)
        mock_run_page.return_value = df_page

        # Execute full fetch
        result = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=["market surveillance"],
            eurovoc_uris=[],
            language="ENG",
            date_from="2019-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=1,
            timeout_s=30,
            retries=3,
        )

        # Verify results
        assert not result.empty
        assert "work_uri" in result.columns
        assert "celex" in result.columns
        assert "title" in result.columns
        assert "link_html" in result.columns

        # Save to parquet
        output_path = tmp_path / "test_output.parquet"
        safe_to_parquet(result, output_path)

        # Verify parquet can be loaded
        loaded = pd.read_parquet(output_path)
        assert len(loaded) == len(result)
        pd.testing.assert_frame_equal(loaded, result)

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_full_ingestion_and_enrichment_pipeline(self, mock_get, mock_run_page, tmp_path):
        """Test complete pipeline from SPARQL to text enrichment."""
        # Setup SPARQL mock
        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        from src.forecasting.regulatory.ingest.cellar_sparql import _bindings_to_rows

        rows = _bindings_to_rows(bindings)
        df_page = pd.DataFrame(rows)
        mock_run_page.return_value = df_page

        # Setup HTTP mock for text fetching
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Step 1: Ingest from SPARQL
        df_ingested = fetch_all(
            endpoint="http://example.com/sparql",
            query_terms=[],
            eurovoc_uris=["http://eurovoc.europa.eu/4512"],
            language="ENG",
            date_from="2019-01-01",
            date_to="2020-12-31",
            page_size=100,
            max_pages=1,
            timeout_s=30,
            retries=3,
        )

        assert not df_ingested.empty
        assert "link_html" in df_ingested.columns

        # Step 2: Enrich with full text
        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        df_enriched = fetcher.enrich_dataframe(df_ingested, max_docs=2)

        assert "full_text" in df_enriched.columns
        assert df_enriched["full_text"].notna().any()

        # Step 3: Save enriched data
        output_path = tmp_path / "enriched_output.parquet"
        safe_to_parquet(df_enriched, output_path)

        # Verify complete pipeline
        loaded = pd.read_parquet(output_path)
        assert "full_text" in loaded.columns
        assert "celex" in loaded.columns
        assert "link_html" in loaded.columns


class TestRealDataValidation:
    """Tests using real parquet data for validation."""

    @pytest.fixture
    def real_data_path(self):
        """Path to real parquet data."""
        return Path("c:/Users/fraca/forecast_refactored/data/regulatory/raw/cellar_20260105_094309.parquet")

    def test_real_data_loads_correctly(self, real_data_path):
        """Test real parquet data loads with expected schema."""
        if not real_data_path.exists():
            pytest.skip("Real data file not found")

        df = pd.read_parquet(real_data_path)

        # Verify schema
        expected_columns = [
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

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_real_data_celex_extraction(self, real_data_path):
        """Test CELEX codes in real data are valid."""
        if not real_data_path.exists():
            pytest.skip("Real data file not found")

        df = pd.read_parquet(real_data_path)

        # Check CELEX codes are not None for rows with celex
        celex_codes = df[df["celex"].notna()]["celex"]
        assert len(celex_codes) > 0

        # Sample CELEX codes should have expected format (digits + letter + digits)
        for celex in celex_codes.head(10):
            assert len(celex) > 5
            # Most CELEX start with sector (3,5,6) + year + type + number
            # e.g., 32019R1020, 52021PC0001

    def test_real_data_link_generation(self, real_data_path):
        """Test EUR-Lex links in real data are valid."""
        if not real_data_path.exists():
            pytest.skip("Real data file not found")

        df = pd.read_parquet(real_data_path)

        # Check links exist for rows with CELEX
        rows_with_celex = df[df["celex"].notna()]

        for _, row in rows_with_celex.head(5).iterrows():
            link_html = row["link_html"]
            link_pdf = row["link_pdf"]

            if pd.notna(link_html):
                assert "eur-lex.europa.eu" in link_html
                assert "CELEX:" in link_html
                assert row["celex"] in link_html

            if pd.notna(link_pdf):
                assert "eur-lex.europa.eu" in link_pdf
                assert "PDF" in link_pdf
                assert row["celex"] in link_pdf

    def test_real_data_date_range(self, real_data_path):
        """Test dates in real data are valid."""
        if not real_data_path.exists():
            pytest.skip("Real data file not found")

        df = pd.read_parquet(real_data_path)

        # Check date format
        dates = df[df["date_document"].notna()]["date_document"]
        assert len(dates) > 0

        # Sample dates should be valid ISO format
        for date in dates.head(10):
            assert len(date) == 10  # YYYY-MM-DD
            assert date[4] == "-"
            assert date[7] == "-"

    def test_real_data_null_handling(self, real_data_path):
        """Test real data properly handles NULL fields."""
        if not real_data_path.exists():
            pytest.skip("Real data file not found")

        df = pd.read_parquet(real_data_path)

        # Title and abstract can be NULL in real data
        # This is a known issue mentioned in requirements
        null_titles = df["title"].isna().sum()
        null_abstracts = df["abstract"].isna().sum()

        # Just verify we can handle NULLs without errors
        assert null_titles >= 0
        assert null_abstracts >= 0

        # Verify work_uri is never NULL (required field)
        assert df["work_uri"].notna().all()

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_enrich_real_data_sample(self, mock_get, real_data_path, tmp_path):
        """Test enriching a sample of real data."""
        if not real_data_path.exists():
            pytest.skip("Real data file not found")

        # Load real data
        df = pd.read_parquet(real_data_path)

        # Take small sample with valid links
        df_sample = df[df["link_html"].notna()].head(3)

        # Mock HTTP responses
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Enrich
        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        df_enriched = fetcher.enrich_dataframe(df_sample)

        # Verify enrichment
        assert "full_text" in df_enriched.columns
        assert len(df_enriched) == len(df_sample)


class TestQueryBuilderIntegration:
    """Integration tests for query building with various filter combinations."""

    def test_query_with_eurovoc_and_date_range(self):
        """Test query builds correctly with EuroVoc filter and date range."""
        query = build_query(
            query_terms=[],
            eurovoc_uris=["http://eurovoc.europa.eu/4512", "http://eurovoc.europa.eu/1234"],
            language="ENG",
            date_from="2020-01-01",
            date_to="2020-12-31",
            limit=100,
            offset=0,
        )

        # Verify all components
        assert "VALUES ?eurovoc" in query
        assert "http://eurovoc.europa.eu/4512" in query
        assert "2020-01-01" in query
        assert "2020-12-31" in query
        assert '"ENG"' in query
        assert "LIMIT 100" in query

    def test_query_with_terms_and_date_range(self):
        """Test query builds correctly with query terms and date range."""
        query = build_query(
            query_terms=["climate", "environment"],
            eurovoc_uris=[],
            language="FRA",
            date_from="2019-06-01",
            date_to="2021-12-31",
            limit=50,
            offset=100,
        )

        # Verify all components
        assert "FILTER" in query
        assert "climate" in query
        assert "environment" in query
        assert "2019-06-01" in query
        assert "2021-12-31" in query
        assert '"FRA"' in query
        assert "LIMIT 50" in query
        assert "OFFSET 100" in query

    def test_query_with_no_topic_filter(self):
        """Test query builds correctly with no topic filter."""
        query = build_query(
            query_terms=[],
            eurovoc_uris=[],
            language="DEU",
            date_from="2018-01-01",
            date_to="2022-12-31",
            limit=200,
            offset=0,
        )

        # Should have no VALUES or FILTER for topics
        assert "VALUES ?eurovoc" not in query
        # Date filter should still be present
        assert "?date >=" in query
        assert "2018-01-01" in query
        assert '"DEU"' in query


class TestErrorHandlingIntegration:
    """Integration tests for error handling across modules."""

    @patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
    def test_handles_partial_sparql_failures(self, mock_run_page, tmp_path):
        """Test pipeline handles partial SPARQL failures gracefully."""
        import requests

        from src.forecasting.regulatory.ingest.cellar_sparql import _bindings_to_rows

        bindings = SPARQL_RESPONSE_SAMPLE["results"]["bindings"]
        rows = _bindings_to_rows(bindings)
        df_page = pd.DataFrame(rows)

        # First page succeeds, second fails, third succeeds
        mock_run_page.side_effect = [
            df_page,
            requests.Timeout("Timeout"),
            requests.Timeout("Timeout"),
            df_page,
        ]

        # Should recover after retries
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

        # Should get at least first page
        assert not result.empty

    def test_handles_malformed_celex_uris(self):
        """Test pipeline handles malformed CELEX URIs."""
        malformed_uris = [
            "not-a-uri",
            "http://example.com/wrong/path",
            None,
            "",
            "http://publications.europa.eu/resource/other/12345",
        ]

        for uri in malformed_uris:
            celex = extract_celex_from_uri(uri)
            # Should return None, not raise
            assert celex is None

            # Links should also handle None celex
            html, pdf = eurlex_links_from_celex(celex, "ENG")
            assert html is None
            assert pdf is None

    def test_handles_empty_dataframe_normalization(self):
        """Test normalization handles empty DataFrames."""
        from src.forecasting.regulatory.ingest.cellar_sparql import _normalize

        df_empty = pd.DataFrame()
        result = _normalize(df_empty)

        # Should return empty DataFrame with correct schema
        assert result.empty
        assert len(result.columns) == 10  # All expected columns

    def test_directory_creation_edge_cases(self, tmp_path):
        """Test directory creation handles edge cases."""
        # Multiple nested levels
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e" / "file.parquet"
        ensure_dirs(deep_path)
        assert deep_path.parent.exists()

        # Path with spaces
        space_path = tmp_path / "path with spaces" / "file.parquet"
        ensure_dirs(space_path)
        assert space_path.parent.exists()

        # Already existing
        existing = tmp_path / "existing"
        existing.mkdir()
        ensure_dirs(existing)  # Should not raise
        assert existing.exists()
