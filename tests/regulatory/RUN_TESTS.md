# Running Regulatory Tests

Quick reference for running the regulatory module test suite.

## Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Basic Commands

### Run all regulatory tests
```bash
pytest tests/regulatory/
```

### Run all tests with coverage report
```bash
pytest tests/regulatory/ --cov=src/forecasting/regulatory --cov-report=html
```
Coverage report will be in `htmlcov/index.html`

### Run specific test file
```bash
# Parsing tests
pytest tests/regulatory/test_parsing.py

# SPARQL tests
pytest tests/regulatory/test_cellar_sparql.py

# Text fetcher tests
pytest tests/regulatory/test_text_fetcher.py

# I/O utilities tests
pytest tests/regulatory/test_io_utils.py

# Integration tests
pytest tests/regulatory/test_integration.py
```

### Run specific test class
```bash
pytest tests/regulatory/test_parsing.py::TestExtractCelexFromUri
pytest tests/regulatory/test_cellar_sparql.py::TestBuildQuery
```

### Run specific test method
```bash
pytest tests/regulatory/test_parsing.py::TestExtractCelexFromUri::test_extract_celex_from_various_uris
```

## Test Organization

### Unit tests only (fast)
```bash
pytest tests/regulatory/ -m unit
```

### Integration tests only
```bash
pytest tests/regulatory/ -m integration
```

### Tests with real data
```bash
pytest tests/regulatory/ -m real_data
```

### Exclude slow tests
```bash
pytest tests/regulatory/ -m "not slow"
```

## Output Options

### Verbose output
```bash
pytest tests/regulatory/ -v
```

### Very verbose (show all test names)
```bash
pytest tests/regulatory/ -vv
```

### Show print statements
```bash
pytest tests/regulatory/ -s
```

### Show summary of all outcomes
```bash
pytest tests/regulatory/ -ra
```

### Stop on first failure
```bash
pytest tests/regulatory/ -x
```

### Run last failed tests only
```bash
pytest tests/regulatory/ --lf
```

### Run failed tests first
```bash
pytest tests/regulatory/ --ff
```

## Coverage Reports

### Terminal coverage report
```bash
pytest tests/regulatory/ --cov=src/forecasting/regulatory --cov-report=term
```

### HTML coverage report
```bash
pytest tests/regulatory/ --cov=src/forecasting/regulatory --cov-report=html
# Open htmlcov/index.html in browser
```

### Coverage report with missing lines
```bash
pytest tests/regulatory/ --cov=src/forecasting/regulatory --cov-report=term-missing
```

### Coverage for specific module
```bash
# Only cellar_sparql module
pytest tests/regulatory/test_cellar_sparql.py --cov=src/forecasting/regulatory/ingest/cellar_sparql

# Only text_fetcher module
pytest tests/regulatory/test_text_fetcher.py --cov=src/forecasting/regulatory/enrichment/text_fetcher
```

## Parallel Execution

Install pytest-xdist:
```bash
pip install pytest-xdist
```

Run tests in parallel:
```bash
# Auto-detect CPU count
pytest tests/regulatory/ -n auto

# Use specific number of workers
pytest tests/regulatory/ -n 4
```

## Filtering Tests

### Run tests matching pattern
```bash
# All tests with "celex" in name
pytest tests/regulatory/ -k celex

# All tests with "fetch" or "enrich" in name
pytest tests/regulatory/ -k "fetch or enrich"

# Exclude specific tests
pytest tests/regulatory/ -k "not slow"
```

## Debugging

### Drop into debugger on failure
```bash
pytest tests/regulatory/ --pdb
```

### Start debugger at beginning of test
Add to test:
```python
import pytest; pytest.set_trace()
```

### Show local variables on failure
```bash
pytest tests/regulatory/ -l
```

## CI/CD Integration

### Generate JUnit XML report
```bash
pytest tests/regulatory/ --junit-xml=test-results.xml
```

### Combined coverage and JUnit
```bash
pytest tests/regulatory/ \
    --cov=src/forecasting/regulatory \
    --cov-report=xml \
    --cov-report=html \
    --junit-xml=test-results.xml
```

## Common Workflows

### Quick check (unit tests only, no coverage)
```bash
pytest tests/regulatory/ -m unit -v
```

### Full validation (all tests with coverage)
```bash
pytest tests/regulatory/ \
    --cov=src/forecasting/regulatory \
    --cov-report=html \
    --cov-report=term-missing \
    -v
```

### Development workflow (fast feedback)
```bash
# Run only tests related to changes
pytest tests/regulatory/ --lf -x -v

# Or watch mode (requires pytest-watch)
pip install pytest-watch
ptw tests/regulatory/
```

### Pre-commit check
```bash
# Run all tests with coverage
pytest tests/regulatory/ \
    --cov=src/forecasting/regulatory \
    --cov-report=term-missing \
    --cov-fail-under=80 \
    -v
```

## Expected Results

### Test Counts
- **Unit tests**: ~150+ tests
  - test_parsing.py: ~30 tests
  - test_io_utils.py: ~25 tests
  - test_cellar_sparql.py: ~50 tests
  - test_text_fetcher.py: ~40 tests
- **Integration tests**: ~15 tests
  - test_integration.py: ~15 tests

### Coverage Targets
- Overall coverage: >90%
- Per-module coverage:
  - parsing.py: >95%
  - io_utils.py: >90%
  - cellar_sparql.py: >85%
  - text_fetcher.py: >85%

### Typical Run Time
- Unit tests only: ~5-10 seconds
- All tests (with mocks): ~15-30 seconds
- Real data tests: ~30-60 seconds (if data available)

## Troubleshooting

### Import errors
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/forecast_refactored"

# Or use pytest with src layout
pip install -e .
```

### Cache issues
```bash
# Clear pytest cache
pytest --cache-clear tests/regulatory/

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

### Temporary file cleanup
```bash
# Pytest automatically cleans tmp_path fixtures
# Manual cleanup if needed:
rm -rf /tmp/pytest-of-$USER/
```

## Tips

1. **Use fixtures**: Leverage shared fixtures in `conftest.py` for common test data

2. **Parametrize tests**: Use `@pytest.mark.parametrize` for testing multiple inputs

3. **Mock external calls**: Always mock HTTP requests (SPARQL, EUR-Lex)

4. **Test isolation**: Each test should be independent and not rely on test order

5. **Fast feedback**: Run unit tests frequently during development

6. **Coverage as guide**: Use coverage to find untested code paths, not as a goal

7. **Real data tests**: Mark tests using real data with `@pytest.mark.real_data`

8. **Network tests**: Mark tests requiring network with `@pytest.mark.network` (should be mocked)
