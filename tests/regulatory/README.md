# Regulatory Module Test Suite

Comprehensive test suite for the `forecasting.regulatory` module, covering ingestion, enrichment, and data processing workflows.

## Structure

```
tests/regulatory/
├── __init__.py
├── conftest.py                 # Shared pytest fixtures and configuration
├── README.md                   # This file
│
├── fixtures/
│   ├── __init__.py
│   └── sample_responses.py     # Mock SPARQL responses and test data
│
├── test_parsing.py             # Unit tests for parsing helpers
├── test_io_utils.py            # Unit tests for I/O utilities
├── test_cellar_sparql.py       # Unit tests for SPARQL client
├── test_text_fetcher.py        # Unit tests for text enrichment
└── test_integration.py         # Integration tests for end-to-end pipeline
```

## Test Coverage

### Unit Tests

#### `test_parsing.py`
- CELEX code extraction from URIs
- Language code conversion (3-letter to 2-letter)
- Text cleaning and normalization
- Semicolon-separated value joining
- EUR-Lex link generation

#### `test_io_utils.py`
- Logger creation and configuration
- Directory creation (with edge cases)
- Safe Parquet I/O
- Path handling (strings vs Path objects)

#### `test_cellar_sparql.py`
- SPARQL query building
  - Topic filters (EuroVoc URIs vs query terms)
  - Date range filters
  - Language filters
  - Pagination (LIMIT/OFFSET)
- SPARQL response parsing
- DataFrame normalization
  - Grouping by work URI
  - EuroVoc aggregation
  - Link generation
  - NULL handling
- Multi-page fetching with retry logic
- Error handling (timeouts, HTTP errors)

#### `test_text_fetcher.py`
- TextFetcher initialization and configuration
- Cache path generation (MD5 hashing)
- Rate limiting enforcement
- HTML fetching with retries
- Text extraction from EUR-Lex HTML
  - Multiple div selectors
  - Language variants (EN, FR, etc.)
  - Fallback strategies
  - Minimum length validation
- Full text fetching with caching
- DataFrame enrichment
  - Custom column names
  - max_docs limit
  - Error handling
  - Progress tracking

### Integration Tests

#### `test_integration.py`
- End-to-end ingestion pipeline (SPARQL → DataFrame → Parquet)
- Full enrichment pipeline (SPARQL → Text fetching → Parquet)
- Real data validation
  - Schema validation
  - CELEX code format validation
  - Link generation validation
  - Date range validation
  - NULL handling
- Query builder with various filter combinations
- Error handling across modules

## Running Tests

### Run all tests
```bash
pytest tests/regulatory/
```

### Run specific test file
```bash
pytest tests/regulatory/test_parsing.py
pytest tests/regulatory/test_cellar_sparql.py
```

### Run with coverage
```bash
pytest tests/regulatory/ --cov=src/forecasting/regulatory --cov-report=html
```

### Run only unit tests (fast)
```bash
pytest tests/regulatory/ -m unit
```

### Run only integration tests
```bash
pytest tests/regulatory/ -m integration
```

### Run tests requiring real data
```bash
pytest tests/regulatory/ -m real_data
```

### Run with verbose output
```bash
pytest tests/regulatory/ -v
```

### Run specific test class or method
```bash
pytest tests/regulatory/test_parsing.py::TestExtractCelexFromUri
pytest tests/regulatory/test_parsing.py::TestExtractCelexFromUri::test_extract_celex_from_various_uris
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `tmp_cache_dir`: Temporary directory for cache testing
- `sample_sparql_response`: Mock SPARQL JSON response
- `sample_eurovoc_uris`: Sample EuroVoc concept URIs
- `sample_query_terms`: Sample keyword search terms
- `sample_cellar_dataframe`: Pre-normalization CELLAR data
- `sample_normalized_dataframe`: Post-normalization data
- `sample_enriched_dataframe`: Data with full text
- `mock_sparql_endpoint`: Mock SPARQL endpoint URL
- `sample_query_params`: Sample query parameters
- `real_parquet_path`: Path to real data file (if exists)

## Test Data

### Mock Responses (`fixtures/sample_responses.py`)

- `SPARQL_RESPONSE_SAMPLE`: Sample SPARQL response with 3 documents
- `SPARQL_RESPONSE_EMPTY`: Empty SPARQL response
- `SPARQL_RESPONSE_WITH_NULLS`: Response with NULL/missing fields
- `SPARQL_RESPONSE_MULTILANG`: Response with multiple languages
- `EURLEX_HTML_SAMPLE`: Sample EUR-Lex HTML document
- `EURLEX_HTML_MINIMAL`: Minimal HTML (fallback testing)
- `EURLEX_HTML_FRENCH`: French language HTML
- `TEST_CELEX_CODES`: CELEX URI test cases
- `TEST_LANGUAGE_MAPPINGS`: Language code mappings
- `TEST_EUROVOC_URIS`: Sample EuroVoc URIs
- `TEST_QUERY_TERMS`: Sample query terms

### Real Data

Tests can optionally use real parquet data:
- `data/regulatory/raw/cellar_20260105_094309.parquet`

Tests using real data are marked with `@pytest.mark.real_data` and will be skipped if the file doesn't exist.

## Mocking Strategy

### HTTP Calls

All external HTTP calls are mocked using `unittest.mock.patch`:

```python
@patch("src.forecasting.regulatory.ingest.cellar_sparql.requests.post")
def test_run_page_success(self, mock_post):
    mock_response = Mock()
    mock_response.json.return_value = SPARQL_RESPONSE_SAMPLE
    mock_post.return_value = mock_response
    # ... test code
```

### SPARQL Endpoint

SPARQL queries are tested by mocking `requests.post` at the module level:

```python
@patch("src.forecasting.regulatory.ingest.cellar_sparql.run_page")
def test_fetch_all_multiple_pages(self, mock_run_page):
    mock_run_page.side_effect = [df_page1, df_page2]
    # ... test code
```

### Text Fetching

EUR-Lex HTML fetching is mocked at the `requests.get` level:

```python
@patch("src.forecasting.regulatory.enrichment.text_fetcher.requests.get")
def test_fetch_text_success(self, mock_get, tmp_path):
    mock_response = Mock()
    mock_response.text = EURLEX_HTML_SAMPLE
    mock_get.return_value = mock_response
    # ... test code
```

## Critical Test Cases

### NULL Field Handling
- Tests verify correct handling of NULL/missing fields (known issue in real data)
- See: `test_normalize_handles_null_fields`, `test_real_data_null_handling`

### Retry Logic
- Tests verify exponential backoff retries for transient errors
- See: `test_fetch_all_retries_on_timeout`, `test_fetch_all_stops_after_max_retries`

### Pagination
- Tests verify correct LIMIT/OFFSET calculation and stopping conditions
- See: `test_fetch_all_multiple_pages`, `test_fetch_all_stops_on_empty_page`

### Rate Limiting
- Tests verify rate limiting enforces delays between requests
- See: `test_rate_limit_delays_requests`, `test_fetch_text_enforces_rate_limit`

### Text Extraction
- Tests verify multiple HTML selectors and fallback strategies
- See: `test_extract_text_from_standard_html`, `test_extract_text_from_minimal_html`

### CELEX Parsing
- Tests verify robust CELEX extraction from various URI formats
- See: `test_extract_celex_from_various_uris`, `test_real_data_celex_extraction`

## Dependencies

Required test dependencies:
- pytest >= 7.0
- pytest-cov (for coverage reports)
- pandas
- requests
- beautifulsoup4
- tenacity
- pyarrow (for parquet I/O)

## Known Issues

1. **NULL Titles/Abstracts**: Real data may have NULL title/abstract fields. Tests verify this is handled gracefully.

2. **Language Field**: Language field may be empty in real data. Tests verify fallback to "EN".

3. **Rate Limiting**: Tests with rate limiting may take longer to run. Use `rate_limit_delay=0` in test fixtures.

## Future Enhancements

- [ ] Add performance benchmarks
- [ ] Add tests for concurrent fetching
- [ ] Add tests for incremental updates
- [ ] Add tests for data validation rules
- [ ] Add tests for error reporting and logging
