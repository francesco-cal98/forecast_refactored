"""Tests for regulatory.enrichment.text_fetcher module.

Tests text fetching, caching, rate limiting, HTML parsing,
and DataFrame enrichment.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from src.forecasting.regulatory.enrichment.text_fetcher import TextFetcher
from tests.regulatory.fixtures.sample_responses import (
    EURLEX_HTML_FRENCH,
    EURLEX_HTML_MINIMAL,
    EURLEX_HTML_SAMPLE,
)


class TestTextFetcherInit:
    """Tests for TextFetcher initialization."""

    def test_init_creates_cache_dir(self, tmp_path):
        """Test initialization creates cache directory."""
        cache_dir = tmp_path / "test_cache"
        assert not cache_dir.exists()

        fetcher = TextFetcher(cache_dir=cache_dir)

        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_init_default_parameters(self, tmp_path):
        """Test initialization with default parameters."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")

        assert fetcher.rate_limit_delay == 1.0
        assert fetcher.timeout == 30
        assert fetcher.retries == 3

    def test_init_custom_parameters(self, tmp_path):
        """Test initialization with custom parameters."""
        fetcher = TextFetcher(
            cache_dir=tmp_path / "cache",
            rate_limit_delay=2.5,
            timeout=60,
            retries=5,
        )

        assert fetcher.rate_limit_delay == 2.5
        assert fetcher.timeout == 60
        assert fetcher.retries == 5


class TestCachePath:
    """Tests for cache path generation."""

    def test_get_cache_path_generates_consistent_hash(self, tmp_path):
        """Test cache path is consistent for same URL."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        url = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R1020"

        path1 = fetcher._get_cache_path(url)
        path2 = fetcher._get_cache_path(url)

        assert path1 == path2

    def test_get_cache_path_different_for_different_urls(self, tmp_path):
        """Test cache path differs for different URLs."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        url1 = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R1020"
        url2 = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32020L0262"

        path1 = fetcher._get_cache_path(url1)
        path2 = fetcher._get_cache_path(url2)

        assert path1 != path2

    def test_get_cache_path_has_txt_extension(self, tmp_path):
        """Test cache path has .txt extension."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        url = "https://example.com/test"

        path = fetcher._get_cache_path(url)

        assert path.suffix == ".txt"


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_delays_requests(self, tmp_path):
        """Test rate limiter enforces delay between requests."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0.5)

        # First call should not delay
        start = time.time()
        fetcher._rate_limit()
        first_duration = time.time() - start

        # Second call should delay
        start = time.time()
        fetcher._rate_limit()
        second_duration = time.time() - start

        assert first_duration < 0.1  # First call is fast
        assert second_duration >= 0.4  # Second call waits ~0.5s

    def test_rate_limit_updates_last_request_time(self, tmp_path):
        """Test rate limiter updates last request time."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")

        initial_time = fetcher.last_request_time
        fetcher._rate_limit()

        assert fetcher.last_request_time > initial_time


class TestFetchHtml:
    """Tests for HTML fetching with retries."""

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_html_success(self, mock_get, tmp_path):
        """Test successful HTML fetch."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        html = fetcher._fetch_html("https://example.com")

        assert html == EURLEX_HTML_SAMPLE
        mock_get.assert_called_once()

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_html_uses_correct_headers(self, mock_get, tmp_path):
        """Test fetch uses correct User-Agent and Accept headers."""
        mock_response = Mock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        fetcher._fetch_html("https://example.com")

        call_args = mock_get.call_args
        headers = call_args[1]["headers"]

        assert "User-Agent" in headers
        assert "Accept" in headers

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_html_uses_timeout(self, mock_get, tmp_path):
        """Test fetch uses configured timeout."""
        mock_response = Mock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", timeout=60)
        fetcher._fetch_html("https://example.com")

        call_args = mock_get.call_args
        assert call_args[1]["timeout"] == 60

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_html_handles_http_error(self, mock_get, tmp_path):
        """Test fetch raises on HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache")

        with pytest.raises(requests.HTTPError):
            fetcher._fetch_html("https://example.com")


class TestExtractTextFromHtml:
    """Tests for text extraction from EUR-Lex HTML."""

    def test_extract_text_from_standard_html(self, tmp_path):
        """Test extraction from standard EUR-Lex HTML structure."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        text = fetcher._extract_text_from_html(EURLEX_HTML_SAMPLE)

        assert text is not None
        assert "Regulation (EU) 2019/1020" in text
        assert "Article 1" in text
        assert "Article 2" in text
        assert "market surveillance" in text

    def test_extract_text_from_french_html(self, tmp_path):
        """Test extraction from French EUR-Lex HTML."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        text = fetcher._extract_text_from_html(EURLEX_HTML_FRENCH)

        assert text is not None
        assert "règlement" in text
        assert "Article 1" in text

    def test_extract_text_from_minimal_html(self, tmp_path):
        """Test extraction falls back to body for minimal HTML."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        text = fetcher._extract_text_from_html(EURLEX_HTML_MINIMAL)

        assert text is not None
        assert "Some text" in text

    def test_extract_text_preserves_paragraph_structure(self, tmp_path):
        """Test extraction preserves paragraph structure with newlines."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        text = fetcher._extract_text_from_html(EURLEX_HTML_SAMPLE)

        # Should have multiple lines/paragraphs
        assert "\n" in text

    def test_extract_text_minimum_length_check(self, tmp_path):
        """Test extraction rejects too-short content."""
        html = "<html><body><div id='text'>Short</div></body></html>"

        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        text = fetcher._extract_text_from_html(html)

        # Less than 100 chars, should be rejected
        assert text is None

    def test_extract_text_handles_empty_html(self, tmp_path):
        """Test extraction handles empty HTML gracefully."""
        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        text = fetcher._extract_text_from_html("")

        assert text is None

    def test_extract_text_handles_malformed_html(self, tmp_path):
        """Test extraction handles malformed HTML gracefully."""
        html = "<html><body><div id='text'><p>Unclosed paragraph</body></html>"

        fetcher = TextFetcher(cache_dir=tmp_path / "cache")
        # Should not raise, BeautifulSoup handles malformed HTML
        text = fetcher._extract_text_from_html(html)

        assert text is None or isinstance(text, str)


class TestFetchText:
    """Tests for full text fetching with caching."""

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_text_success(self, mock_get, tmp_path):
        """Test successful text fetch and cache."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        url = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R1020"

        text = fetcher.fetch_text(url)

        assert text is not None
        assert "Regulation (EU) 2019/1020" in text

        # Check cache was written
        cache_path = fetcher._get_cache_path(url)
        assert cache_path.exists()

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_text_uses_cache(self, mock_get, tmp_path):
        """Test fetch uses cache on subsequent calls."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        url = "https://example.com/test"

        # First call - fetches from web
        text1 = fetcher.fetch_text(url)
        assert mock_get.call_count == 1

        # Second call - uses cache
        text2 = fetcher.fetch_text(url, use_cache=True)
        assert mock_get.call_count == 1  # Still 1, no new request

        assert text1 == text2

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_text_bypasses_cache_when_disabled(self, mock_get, tmp_path):
        """Test fetch bypasses cache when use_cache=False."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        url = "https://example.com/test"

        # First call
        text1 = fetcher.fetch_text(url, use_cache=False)
        assert mock_get.call_count == 1

        # Second call with use_cache=False
        text2 = fetcher.fetch_text(url, use_cache=False)
        assert mock_get.call_count == 2  # New request made

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_text_handles_request_error(self, mock_get, tmp_path):
        """Test fetch handles request errors gracefully."""
        mock_get.side_effect = requests.RequestException("Connection error")

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        text = fetcher.fetch_text("https://example.com")

        assert text is None

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_text_handles_extraction_failure(self, mock_get, tmp_path):
        """Test fetch handles text extraction failure."""
        mock_response = Mock()
        mock_response.text = "<html><body>Too short</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        text = fetcher.fetch_text("https://example.com")

        assert text is None

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_fetch_text_enforces_rate_limit(self, mock_get, tmp_path):
        """Test fetch enforces rate limiting."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0.3)

        # First request
        start = time.time()
        fetcher.fetch_text("https://example.com/1", use_cache=False)

        # Second request should be delayed
        start2 = time.time()
        fetcher.fetch_text("https://example.com/2", use_cache=False)
        duration = time.time() - start2

        assert duration >= 0.2  # Should wait ~0.3s


class TestEnrichDataframe:
    """Tests for DataFrame enrichment with full text."""

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_enrich_dataframe_adds_text_column(self, mock_get, tmp_path):
        """Test enrich_dataframe adds full_text column."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = pd.DataFrame(
            {
                "celex": ["32019R1020", "32020L0262"],
                "link_html": [
                    "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R1020",
                    "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32020L0262",
                ],
            }
        )

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        result = fetcher.enrich_dataframe(df)

        assert "full_text" in result.columns
        assert len(result) == 2
        assert result.iloc[0]["full_text"] is not None

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_enrich_dataframe_handles_missing_urls(self, mock_get, tmp_path):
        """Test enrich_dataframe handles missing URLs."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = pd.DataFrame(
            {
                "celex": ["32019R1020", "32020L0262", "32021R0001"],
                "link_html": [
                    "https://example.com/1",
                    None,  # Missing URL
                    "",  # Empty URL
                ],
            }
        )

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        result = fetcher.enrich_dataframe(df)

        assert result.iloc[0]["full_text"] is not None
        assert pd.isna(result.iloc[1]["full_text"])
        assert pd.isna(result.iloc[2]["full_text"])

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_enrich_dataframe_respects_max_docs(self, mock_get, tmp_path):
        """Test enrich_dataframe respects max_docs limit."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = pd.DataFrame(
            {
                "link_html": [
                    "https://example.com/1",
                    "https://example.com/2",
                    "https://example.com/3",
                    "https://example.com/4",
                ]
            }
        )

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        result = fetcher.enrich_dataframe(df, max_docs=2)

        assert len(result) == 2
        assert mock_get.call_count == 2

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_enrich_dataframe_custom_column_names(self, mock_get, tmp_path):
        """Test enrich_dataframe with custom column names."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = pd.DataFrame({"url": ["https://example.com"]})

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        result = fetcher.enrich_dataframe(df, link_column="url", text_column="extracted_text")

        assert "extracted_text" in result.columns
        assert result.iloc[0]["extracted_text"] is not None

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_enrich_dataframe_handles_errors_gracefully(self, mock_get, tmp_path):
        """Test enrich_dataframe continues on errors."""
        # First URL succeeds, second fails, third succeeds
        success_response = Mock()
        success_response.text = EURLEX_HTML_SAMPLE
        success_response.raise_for_status = Mock()

        mock_get.side_effect = [
            success_response,
            requests.RequestException("Error"),
            success_response,
        ]

        df = pd.DataFrame(
            {
                "link_html": [
                    "https://example.com/1",
                    "https://example.com/2",
                    "https://example.com/3",
                ]
            }
        )

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        result = fetcher.enrich_dataframe(df)

        assert len(result) == 3
        assert result.iloc[0]["full_text"] is not None
        assert pd.isna(result.iloc[1]["full_text"])  # Failed
        assert result.iloc[2]["full_text"] is not None

    @patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
    def test_enrich_dataframe_preserves_original_columns(self, mock_get, tmp_path):
        """Test enrich_dataframe preserves original DataFrame columns."""
        mock_response = Mock()
        mock_response.text = EURLEX_HTML_SAMPLE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = pd.DataFrame(
            {
                "celex": ["32019R1020"],
                "title": ["Test Title"],
                "link_html": ["https://example.com"],
            }
        )

        fetcher = TextFetcher(cache_dir=tmp_path / "cache", rate_limit_delay=0)
        result = fetcher.enrich_dataframe(df)

        assert "celex" in result.columns
        assert "title" in result.columns
        assert "link_html" in result.columns
        assert "full_text" in result.columns
