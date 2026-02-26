# Troubleshooting Guide - Regulatory Extraction

## Common Issues and Solutions

### Issue: Notebook Shows "DataFrame is empty"

**Symptom:** When you open `notebooks/sanity_ingest.ipynb`, it shows:
```
DataFrame is empty
```

**Causes:**
1. No extraction has been run yet
2. Latest extraction returned 0 results (filters too restrictive)
3. The parquet files exist but contain 0 rows

**Diagnosis:**
Check which files exist and their row counts:
```bash
python scripts/view_parquet.py --summary
```

Or manually:
```bash
ls -lh data/regulatory/raw/*.parquet
```

**Solutions:**

#### Solution 1: Run a New Extraction (Recommended)

Run extraction without keyword filters to ensure results:
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    date_from=2024-01-01 \
    date_to=2024-12-31 \
    max_pages=10 \
    page_size=100
```

This should extract ~1000 documents (10 pages × 100 rows).

#### Solution 2: Use Broader Date Range

If you got 0 results, the date range might be too narrow:
```bash
# Extract last 3 months
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    date_from=2024-10-01 \
    date_to=2024-12-31 \
    max_pages=5
```

#### Solution 3: Check Latest File Manually

The notebook loads the **most recent file by modification time**:
```python
files = sorted(raw_dir.glob('*.parquet'), key=lambda f: f.stat().st_mtime, reverse=True)
latest = files[0]
```

You can check all files with:
```bash
python -c "
import pandas as pd
from pathlib import Path

for f in sorted(Path('data/regulatory/raw').glob('*.parquet'),
                key=lambda x: x.stat().st_mtime, reverse=True):
    df = pd.read_parquet(f)
    print(f'{f.name}: {len(df)} rows')
"
```

#### Solution 4: Clean Up Empty Files

If you have many empty parquet files cluttering the directory:
```bash
# List empty files
python -c "
import pandas as pd
from pathlib import Path

for f in Path('data/regulatory/raw').glob('*.parquet'):
    df = pd.read_parquet(f)
    if len(df) == 0:
        print(f'{f.name}: EMPTY')
"

# Optional: Delete empty files (be careful!)
# python -c "
# import pandas as pd
# from pathlib import Path
# for f in Path('data/regulatory/raw').glob('*.parquet'):
#     df = pd.read_parquet(f)
#     if len(df) == 0:
#         f.unlink()
#         print(f'Deleted {f.name}')
# "
```

---

### Issue: 0 Results Despite Successful Query

**Symptom:** Logs show "Rows: 0" but no errors occurred.

**Causes:**
1. **Keyword filters too restrictive** - `query_terms=['cosmetics']` finds nothing because most documents lack `title` or `abstract`
2. **Date range has no data** - e.g., future dates or very old dates
3. **Language filter** - Some languages have fewer documents

**Solutions:**

#### Remove Keyword Filters
```bash
python -m src.forecasting.regulatory.ingest.run_ingest query_terms=[]
```

#### Use EuroVoc URIs Instead of Keywords
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    eurovoc_uris=['http://eurovoc.europa.eu/2326']
```

Find EuroVoc concepts at: https://op.europa.eu/en/web/eu-vocabularies

#### Test with Debug Mode
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    date_from=2024-12-01 \
    date_to=2024-12-31 \
    max_pages=1 \
    page_size=10 \
    log_level=DEBUG
```

This will show you the exact SPARQL query being sent.

---

### Issue: Why Are title/abstract/language Always NULL?

**Symptom:** When viewing data, columns show:
```
title: 143 (100.0% null)
abstract: 143 (100.0% null)
language: 143 (100.0% null)
```

**Explanation:**

This is **NORMAL** for many CELLAR documents. The SPARQL query has:
```sparql
OPTIONAL {
  ?expr cdm:expression_title ?title .
  ?expr cdm:expression_uses_language ?langRes .
}
```

The `OPTIONAL` keyword means these fields are not required. Many legal documents in CELLAR:
- Have no English title/abstract
- Are in multiple languages (title varies by language)
- Are stored only as CELEX identifiers with links

**Impact:**
- ❌ Keyword filtering on `title`/`abstract` won't work
- ✅ CELEX identifiers are always available
- ✅ Links to EUR-Lex HTML/PDF are generated correctly
- ✅ EuroVoc topic filtering still works

**Workaround:**

Use the generated links to access the full document:
```python
import pandas as pd

df = pd.read_parquet('data/regulatory/raw/latest.parquet')

# Print clickable links
for _, row in df.head(10).iterrows():
    print(f"{row['celex']}: {row['link_html']}")
```

Or filter by EuroVoc topics instead of title keywords.

---

### Issue: Can't Install Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'hydra'
```

**Solutions:**

#### Install Missing Package
```bash
pip install hydra-core
```

#### Install All Requirements
```bash
pip install -r requirements.txt
```

**Note:** The file is named `requirments.txt` (typo), should be `requirements.txt`.

#### Verify Installation
```bash
python -c "import hydra; import pandas; import pyarrow; import tenacity; print('OK')"
```

---

### Issue: SPARQL Timeout

**Symptom:**
```
requests.exceptions.Timeout: HTTPSConnectionPool...
```

**Solutions:**

#### Increase Timeout
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    regulatory.sources.timeout_s=120
```

#### Reduce Page Size
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    page_size=100
```

#### Increase Retries
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    regulatory.sources.retries=5
```

---

### Issue: Too Many Results / Long Execution

**Symptom:** Query takes forever or generates huge files.

**Solutions:**

#### Limit Pages
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    max_pages=5
```

#### Narrow Date Range
```bash
python -m src.forecasting.regulatory.ingest.run_ingest \
    date_from=2024-12-01 \
    date_to=2024-12-31
```

#### Run Incrementally by Month
```bash
for month in {01..12}; do
    python -m src.forecasting.regulatory.ingest.run_ingest \
        date_from=2024-${month}-01 \
        date_to=2024-${month}-31 \
        query_terms=[]
done
```

---

### Issue: How to Filter Results After Extraction?

**Best Practice:** Extract everything first, then filter in Python/pandas.

**Reason:** CELLAR documents lack `title`/`abstract`, so pre-filtering gives 0 results.

**Workflow:**

1. Extract all documents:
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest \
       query_terms=[] \
       date_from=2020-01-01 \
       date_to=2024-12-31 \
       max_pages=50
   ```

2. Filter in pandas:
   ```python
   import pandas as pd

   df = pd.read_parquet('data/regulatory/raw/latest.parquet')

   # Filter by resource type
   regulations = df[df['resource_type'] == 'R']

   # Filter by date range
   recent = df[df['date_document'] >= '2024-01-01']

   # Filter by EuroVoc topic (contains cosmetics)
   cosmetics = df[df['eurovoc'].str.contains('cosmetic', case=False, na=False)]

   # Save filtered results
   regulations.to_parquet('data/regulatory/processed/regulations_only.parquet')
   ```

3. Use the CLI viewer with filters:
   ```bash
   python scripts/view_parquet.py --filter resource_type=R --output regulations.csv
   ```

---

## Quick Diagnostic Checklist

When something doesn't work:

- [ ] Check dependencies: `pip list | grep -E "(hydra|pandas|pyarrow)"`
- [ ] Check file existence: `ls data/regulatory/raw/*.parquet`
- [ ] Check row counts: `python scripts/view_parquet.py --summary`
- [ ] Test with minimal query: `query_terms=[] max_pages=1 page_size=10`
- [ ] Enable debug logging: `log_level=DEBUG`
- [ ] Check CELLAR endpoint: `https://publications.europa.eu/webapi/rdf/sparql`

---

## Getting Help

If you're still stuck:

1. Run with DEBUG logging and share the output
2. Check the SPARQL query being generated
3. Test the endpoint directly in browser
4. Review [REGULATORY_EXTRACTION_GUIDE.md](REGULATORY_EXTRACTION_GUIDE.md)

---

**Last Updated:** 2026-01-05
