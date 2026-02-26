# EU Regulatory Data Extraction Guide

## Table of Contents
- [Overview](#overview)
- [Problem Solved](#problem-solved)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Output Data Schema](#output-data-schema)
- [Viewing Results](#viewing-results)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Overview

This system extracts EU legal documents from the **CELLAR** repository (Publications Office of the European Union) via SPARQL queries. It retrieves metadata including:
- Document titles and abstracts
- CELEX identifiers
- Publication dates
- EuroVoc subject classifications
- Direct links to HTML and PDF versions on EUR-Lex

**Architecture:**
```
run_ingest.py (CLI) → cellar_sparql.py (SPARQL client) → CELLAR endpoint
                                                             ↓
                                                    Parquet output
```

**Key Files:**
- `src/forecasting/regulatory/ingest/run_ingest.py` - Hydra CLI entrypoint
- `src/forecasting/regulatory/ingest/cellar_sparql.py` - SPARQL query builder and executor
- `conf/regulatory/ingest.yaml` - Main configuration
- `conf/regulatory/sources/cellar.yaml` - Endpoint settings

---

## Problem Solved

### Critical Bug Fixed

The original code used an incorrect SPARQL ontology class name:
```sparql
# ❌ WRONG (returns 0 results)
?work a cdm:legal-resource .

# ✅ CORRECT
GRAPH ?g {
  ?work a cdm:resource_legal .
}
```

**What was wrong:**
1. Class name was `cdm:legal-resource` instead of `cdm:resource_legal`
2. Query didn't include `GRAPH ?g` wrapper to access named graphs
3. CELLAR stores data in named graphs with pattern `http://publications.europa.eu/resource/cellar/{uuid}`

**No API keys or authentication required** - the endpoint is fully public.

---

## Prerequisites

### System Requirements
- Python 3.11+
- Internet connection to `https://publications.europa.eu/webapi/rdf/sparql`

### Required Dependencies
```bash
pip install hydra-core pandas pyarrow requests tenacity
```

Or install all project dependencies:
```bash
pip install -r requirements.txt
```

**Note:** The file is currently named `requirments.txt` (typo) - this should be renamed to `requirements.txt`.

### Verify Installation
```bash
python -c "import hydra; import pandas; import pyarrow; import tenacity; import requests; print('All dependencies OK')"
```

---

## Quick Start

### 1. Basic Extraction (No Filters)

Extract all legal documents from 2024:
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    date_from=2024-01-01 \
    date_to=2024-12-31 \
    max_pages=5 \
    page_size=100
```

**Output location:** `data/regulatory/raw/cellar_YYYYMMDD_HHMMSS.parquet`

### 2. View Results

Using the notebook:
```bash
jupyter notebook notebooks/sanity_ingest.ipynb
```

Or using the CLI viewer (see [Viewing Results](#viewing-results)):
```bash
python scripts/view_parquet.py data/regulatory/raw/cellar_*.parquet
```

---

## Configuration

### Main Config File: `conf/regulatory/ingest.yaml`

```yaml
source: cellar  # or eurlex_soap (not implemented)

# Keyword filtering (case-insensitive search in title/abstract)
query_terms: ["cosmetics", "textile", "retail"]

# Alternative: EuroVoc concept URIs (overrides query_terms if provided)
eurovoc_uris: []

# Language code (CELLAR 3-letter codes)
language: "ENG"  # ENG, ITA, FRA, DEU, SPA, etc.

# Date range (inclusive)
date_from: "2020-01-01"
date_to: "2025-10-19"

# Pagination
page_size: 5000  # SPARQL LIMIT per page
max_pages: 50    # Maximum pages to fetch

# Output
outfile: "${paths.raw_dir}/cellar_${now:%Y%m%d}_${now:%H%M%S}.parquet"
log_level: "INFO"  # or DEBUG
```

### Endpoint Config: `conf/regulatory/sources/cellar.yaml`

```yaml
endpoint: "https://publications.europa.eu/webapi/rdf/sparql"
timeout_s: 60   # Request timeout
retries: 3      # Retry attempts with exponential backoff
```

### Override Config via CLI

All parameters can be overridden without editing YAML:
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    language=ITA \
    date_from=2023-01-01 \
    max_pages=10
```

---

## Usage Examples

### Example 1: Extract All 2024 Documents

```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    date_from=2024-01-01 \
    date_to=2024-12-31 \
    page_size=1000 \
    max_pages=100
```

**Expected output:** ~100,000 documents (if 100 pages × 1000 rows)

### Example 2: Filter by Keywords

```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=['product','safety'] \
    date_from=2020-01-01 \
    date_to=2025-12-31
```

**Note:** Many documents lack `title` or `abstract`, so keyword filtering may yield 0 results. Use EuroVoc URIs instead for better coverage.

### Example 3: Filter by EuroVoc Topics

```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    eurovoc_uris=['http://eurovoc.europa.eu/2326'] \
    date_from=2023-01-01
```

Find EuroVoc concept URIs at: https://op.europa.eu/en/web/eu-vocabularies

### Example 4: Extract Italian Documents

```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    language=ITA \
    query_terms=[] \
    date_from=2024-01-01 \
    date_to=2024-12-31
```

**Supported language codes:**
- `ENG` - English
- `ITA` - Italian
- `FRA` - French
- `DEU` - German
- `SPA` - Spanish
- `POL` - Polish
- etc. (all EU official languages)

### Example 5: Quick Test (10 Documents)

```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    max_pages=1 \
    page_size=10 \
    log_level=DEBUG
```

### Example 6: Increase Timeout for Slow Networks

```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    regulatory.sources.timeout_s=120 \
    regulatory.sources.retries=5
```

---

## Output Data Schema

### Parquet Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `work_uri` | string | CELLAR work URI | `http://publications.europa.eu/resource/cellar/...` |
| `celex` | string | CELEX identifier | `32019R1020` |
| `title` | string | Document title (may be null) | `Regulation (EU) 2019/1020` |
| `abstract` | string | Document abstract (may be null) | Summary text |
| `date_document` | date | Publication date | `2019-06-20` |
| `resource_type` | string | Document type code | `R`, `L`, `C`, `D` |
| `eurovoc` | string | EuroVoc topics (semicolon-separated) | `http://eurovoc.europa.eu/123; http://...` |
| `link_html` | string | EUR-Lex HTML link | `https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R1020` |
| `link_pdf` | string | EUR-Lex PDF link | `https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019R1020` |
| `language` | string | Language code (may be null) | `EN` |

### Resource Type Codes

- `R` - Regulation (Regolamento)
- `L` - Directive (Direttiva)
- `C` - Communication (Comunicazione)
- `D` - Decision (Decisione)
- `O` - Opinion (Parere)

### CELEX Identifier Format

CELEX codes follow the pattern: `{sector}{year}{type}{number}`

**Examples:**
- `32019R1020` - Regulation 1020 from 2019 (sector 3 = EU law)
- `72022L2555ROU_202500257` - Romanian implementation of Directive 2022/2555

**Learn more:** https://eur-lex.europa.eu/content/tools/TableOfSectors/types_of_documents_in_eurlex.html

---

## Viewing Results

### Method 1: Jupyter Notebook

```bash
jupyter notebook notebooks/sanity_ingest.ipynb
```

The notebook:
- Loads the latest parquet file
- Displays date range statistics
- Shows language distribution
- Previews title and links

### Method 2: Python Script

Use the CLI viewer script:

```bash
# View latest file
python scripts/view_parquet.py

# View specific file
python scripts/view_parquet.py data/regulatory/raw/cellar_20260105_091959.parquet

# Summary only
python scripts/view_parquet.py --summary

# Filter by resource type
python scripts/view_parquet.py --filter resource_type=R

# Export to CSV
python scripts/view_parquet.py --output results.csv
```

### Method 3: Python Code

```python
import pandas as pd
from pathlib import Path

# Load latest parquet
raw_dir = Path('data/regulatory/raw')
files = sorted(raw_dir.glob('*.parquet'))
df = pd.read_parquet(files[-1])

# Basic stats
print(f"Total documents: {len(df)}")
print(f"Date range: {df['date_document'].min()} to {df['date_document'].max()}")
print(f"\nResource types:\n{df['resource_type'].value_counts()}")

# View sample
print(df[['celex', 'title', 'date_document', 'resource_type']].head(10))

# Filter Regulations only
regulations = df[df['resource_type'] == 'R']
print(f"\nRegulations: {len(regulations)}")
```

---

## Troubleshooting

### Issue: 0 Results with Keywords

**Symptom:** Query with `query_terms=['cosmetics', 'textile']` returns 0 rows.

**Cause:** Many CELLAR documents have `null` values for `title` and `abstract`.

**Solutions:**
1. Remove keyword filters:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest query_terms=[]
   ```

2. Use EuroVoc URIs instead:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest \
       eurovoc_uris=['http://eurovoc.europa.eu/2326']
   ```

3. Filter the parquet file after extraction:
   ```python
   df = pd.read_parquet('data/regulatory/raw/cellar_*.parquet')
   cosmetics_docs = df[df['eurovoc'].str.contains('cosmetic', case=False, na=False)]
   ```

### Issue: Timeout Errors

**Symptom:** `requests.exceptions.Timeout` or `RetryError`

**Solutions:**
1. Increase timeout:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest \
       regulatory.sources.timeout_s=120
   ```

2. Reduce page size:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest page_size=100
   ```

3. Increase retry attempts:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest \
       regulatory.sources.retries=5
   ```

### Issue: `ModuleNotFoundError: No module named 'hydra'`

**Solution:** Install missing dependencies:
```bash
pip install hydra-core pyarrow
```

### Issue: Empty DataFrame Despite Query Success

**Symptom:** Logs show "Rows: 0" but no errors.

**Diagnosis:** Check if filters are too restrictive.

**Solutions:**
1. Enable DEBUG logging:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest log_level=DEBUG
   ```

2. Test without filters:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest \
       query_terms=[] \
       max_pages=1 \
       page_size=10
   ```

### Issue: Too Much Data / Long Execution Time

**Solutions:**
1. Limit date range:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest \
       date_from=2024-01-01 \
       date_to=2024-03-31
   ```

2. Limit pages:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest max_pages=5
   ```

3. Run incrementally by year:
   ```bash
   for year in 2020 2021 2022 2023 2024; do
       python -m src.forecasting.regulatory.ingest.run_ingest \
           date_from=${year}-01-01 \
           date_to=${year}-12-31
   done
   ```

---

## Advanced Topics

### Understanding SPARQL Query Structure

The system builds queries like this (simplified):

```sparql
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>

SELECT DISTINCT ?work ?celex ?title ?date
WHERE {
  GRAPH ?g {
    ?work a cdm:resource_legal .
    ?work cdm:work_date_document ?date .

    OPTIONAL {
      ?work owl:sameAs ?celex .
      FILTER(CONTAINS(STR(?celex), "resource/celex/"))
    }

    OPTIONAL {
      ?expr cdm:expression_belongs_to_work ?work ;
            cdm:expression_title ?title .
    }

    FILTER(?date >= "2024-01-01"^^xsd:date && ?date <= "2024-12-31"^^xsd:date)
  }
}
ORDER BY DESC(?date)
LIMIT 1000 OFFSET 0
```

**Key components:**
- `GRAPH ?g` - Query across all named graphs
- `cdm:resource_legal` - Correct class for legal documents
- `OPTIONAL` - Fields that may be missing
- `FILTER` - Date range and keyword constraints

### Pagination Logic

The system fetches data in pages:
1. Start at `OFFSET 0`, fetch `page_size` rows
2. If fewer than `page_size` rows returned → stop (last page)
3. If exactly `page_size` rows → fetch next page at `OFFSET += page_size`
4. Stop after `max_pages` or when page is empty

### Retry Strategy

Uses `tenacity` library with exponential backoff:
- Initial wait: 1 second
- Max wait: 10 seconds
- Retries: 3 attempts (configurable)
- Retry on: `Timeout`, `ConnectionError`, `HTTPError`

### Data Normalization

After fetching raw SPARQL results, the code:
1. Groups by `work_uri` to deduplicate
2. Aggregates multiple EuroVoc topics into semicolon-separated string
3. Extracts CELEX code from URI
4. Maps 3-letter language codes to 2-letter codes (ENG → EN)
5. Generates EUR-Lex links from CELEX + language

### Extending the System

#### Add New Data Sources

1. Create config in `conf/regulatory/sources/new_source.yaml`
2. Implement fetcher in `src/forecasting/regulatory/ingest/new_source.py`
3. Add routing in `run_ingest.py`:
   ```python
   elif source == "new_source":
       df = fetch_from_new_source(cfg)
   ```

#### Add Custom Filters

Modify `_build_topic_filter()` in `cellar_sparql.py`:
```python
def _build_topic_filter(query_terms, eurovoc_uris, custom_filter):
    if custom_filter:
        return f"FILTER({custom_filter})"
    # ... existing logic
```

#### Export to Different Format

Modify `run_ingest.py`:
```python
outfile = cfg.outfile
if outfile.endswith('.csv'):
    df.to_csv(outfile, index=False)
elif outfile.endswith('.json'):
    df.to_json(outfile, orient='records')
else:
    safe_to_parquet(df, outfile)
```

---

## References

### Official Documentation
- [CELLAR SPARQL Endpoint](https://publications.europa.eu/webapi/rdf/sparql)
- [EUR-Lex Portal](https://eur-lex.europa.eu)
- [EU Vocabularies - EuroVoc](https://op.europa.eu/en/web/eu-vocabularies)
- [CELEX Number Information](https://eur-lex.europa.eu/content/tools/TableOfSectors/types_of_documents_in_eurlex.html)

### Technical Resources
- [Hydra Configuration Framework](https://hydra.cc/)
- [SPARQL 1.1 Query Language](https://www.w3.org/TR/sparql11-query/)
- [Apache Parquet Format](https://parquet.apache.org/)

### Related Tools
- [eurlex R package](https://github.com/michalovadek/eurlex) - Alternative implementation in R
- [IPS-X SPARQL Explorer](https://ipsoeu.github.io/ips-explorer/)

---

## Support

For issues or questions:
1. Check logs with `log_level=DEBUG`
2. Verify dependencies with `pip list | grep -E "(hydra|pandas|pyarrow)"`
3. Test CELLAR endpoint directly: https://publications.europa.eu/webapi/rdf/sparql
4. Review error messages in console output

---

**Last Updated:** 2026-01-05
**Version:** 1.0
**Author:** Regulatory Extraction System
