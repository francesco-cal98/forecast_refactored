"""Pytest configuration and shared fixtures for regulatory tests.

Provides common fixtures, mock data, and test utilities used across
all regulatory test modules.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from tests.regulatory.fixtures.sample_responses import (
    SPARQL_RESPONSE_SAMPLE,
    TEST_EUROVOC_URIS,
    TEST_QUERY_TERMS,
)


@pytest.fixture
def tmp_cache_dir() -> Generator[Path, None, None]:
    """Provides a temporary directory for cache testing.

    Automatically cleaned up after test completion.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_sparql_response():
    """Provides sample SPARQL JSON response."""
    return SPARQL_RESPONSE_SAMPLE


@pytest.fixture
def sample_eurovoc_uris():
    """Provides sample EuroVoc URIs for testing."""
    return TEST_EUROVOC_URIS.copy()


@pytest.fixture
def sample_query_terms():
    """Provides sample query terms for testing."""
    return TEST_QUERY_TERMS.copy()


@pytest.fixture
def sample_cellar_dataframe():
    """Provides a sample DataFrame in CELLAR format (pre-normalization).

    Simulates raw data from SPARQL bindings conversion.
    """
    return pd.DataFrame(
        [
            {
                "work": "http://publications.europa.eu/resource/cellar/abc123",
                "celex_uri": "http://publications.europa.eu/resource/celex/32019R1020",
                "title": "Regulation on Market Surveillance",
                "abstract": "This regulation establishes rules...",
                "date": "2019-06-20",
                "rtype": "http://publications.europa.eu/resource/type/REG",
                "eurovoc": "http://eurovoc.europa.eu/4512",
                "lang": "ENG",
            },
            {
                "work": "http://publications.europa.eu/resource/cellar/def456",
                "celex_uri": "http://publications.europa.eu/resource/celex/32020L0262",
                "title": "Directive on Tax Reporting",
                "abstract": None,
                "date": "2020-02-18",
                "rtype": "http://publications.europa.eu/resource/type/DIR",
                "eurovoc": "http://eurovoc.europa.eu/1234",
                "lang": "ENG",
            },
        ]
    )


@pytest.fixture
def sample_normalized_dataframe():
    """Provides a sample normalized DataFrame (post-processing).

    Simulates data ready for parquet output.
    """
    return pd.DataFrame(
        [
            {
                "work_uri": "http://publications.europa.eu/resource/cellar/abc123",
                "celex": "32019R1020",
                "title": "Regulation on Market Surveillance",
                "abstract": "This regulation establishes rules...",
                "date_document": "2019-06-20",
                "resource_type": "REG",
                "eurovoc": "http://eurovoc.europa.eu/4512",
                "link_html": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R1020",
                "link_pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019R1020",
                "language": "ENG",
            },
            {
                "work_uri": "http://publications.europa.eu/resource/cellar/def456",
                "celex": "32020L0262",
                "title": "Directive on Tax Reporting",
                "abstract": None,
                "date_document": "2020-02-18",
                "resource_type": "DIR",
                "eurovoc": "http://eurovoc.europa.eu/1234",
                "link_html": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32020L0262",
                "link_pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32020L0262",
                "language": "ENG",
            },
        ]
    )


@pytest.fixture
def sample_enriched_dataframe(sample_normalized_dataframe):
    """Provides a sample enriched DataFrame with full text.

    Simulates data after text fetching enrichment.
    """
    df = sample_normalized_dataframe.copy()
    df["full_text"] = [
        "Full text of regulation on market surveillance...",
        "Full text of directive on tax reporting...",
    ]
    return df


@pytest.fixture
def mock_sparql_endpoint():
    """Provides a mock SPARQL endpoint URL for testing."""
    return "http://publications.europa.eu/webapi/rdf/sparql"


@pytest.fixture
def sample_query_params():
    """Provides sample query parameters for CELLAR queries."""
    return {
        "query_terms": ["artificial intelligence"],
        "eurovoc_uris": [],
        "language": "ENG",
        "date_from": "2020-01-01",
        "date_to": "2024-12-31",
        "page_size": 100,
        "max_pages": 10,
        "timeout_s": 30,
        "retries": 3,
    }


@pytest.fixture
def real_parquet_path():
    """Provides path to real parquet data file if it exists."""
    path = Path("c:/Users/fraca/forecast_refactored/data/regulatory/raw/cellar_20260105_094309.parquet")
    return path if path.exists() else None


# Custom assertions for regulatory data


def assert_valid_cellar_schema(df: pd.DataFrame) -> None:
    """Assert DataFrame has valid CELLAR schema (pre-normalization)."""
    required_cols = ["work", "celex_uri", "title", "abstract", "date", "rtype", "eurovoc", "lang"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def assert_valid_normalized_schema(df: pd.DataFrame) -> None:
    """Assert DataFrame has valid normalized schema."""
    required_cols = [
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
    assert list(df.columns) == required_cols, f"Schema mismatch. Expected {required_cols}, got {list(df.columns)}"


def assert_valid_enriched_schema(df: pd.DataFrame) -> None:
    """Assert DataFrame has valid enriched schema."""
    assert_valid_normalized_schema(df)
    assert "full_text" in df.columns, "Missing full_text column in enriched data"


# Pytest configuration


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test (slower)")
    config.addinivalue_line("markers", "unit: mark test as unit test (fast)")
    config.addinivalue_line("markers", "real_data: mark test as requiring real data file")
    config.addinivalue_line("markers", "slow: mark test as slow running")
