# Documentation

This directory contains documentation for the Forecast Refactored project.

## Available Guides

### [REGULATORY_EXTRACTION_GUIDE.md](REGULATORY_EXTRACTION_GUIDE.md)

Comprehensive guide for the EU Regulatory Data Extraction system.

### [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

Common issues and solutions, including:
- Empty DataFrame in notebook (most common issue)
- 0 results despite successful query
- Why title/abstract/language are NULL
- Filtering results after extraction

**What's covered in the main guide:**
- How the CELLAR SPARQL extraction works
- Bug fix that was applied (ontology class name correction)
- Installation and setup
- Configuration options
- Usage examples and CLI commands
- Output data schema
- Viewing and analyzing results
- Troubleshooting common issues
- Advanced topics (SPARQL queries, pagination, etc.)

**Quick Links:**
- [Quick Start](REGULATORY_EXTRACTION_GUIDE.md#quick-start)
- [Usage Examples](REGULATORY_EXTRACTION_GUIDE.md#usage-examples)
- [Viewing Results](REGULATORY_EXTRACTION_GUIDE.md#viewing-results)
- [Troubleshooting](REGULATORY_EXTRACTION_GUIDE.md#troubleshooting)

## Related Tools

### Scripts

- `scripts/view_parquet.py` - CLI tool to view, filter, and export parquet files
  ```bash
  python scripts/view_parquet.py --help
  ```

### Notebooks

- `notebooks/sanity_ingest.ipynb` - Interactive notebook to inspect extraction results

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run first extraction:**
   ```bash
   python -m src.forecasting.regulatory.ingest.run_ingest \
       query_terms=[] \
       date_from=2024-01-01 \
       date_to=2024-12-31 \
       max_pages=5
   ```

3. **View results:**
   ```bash
   python scripts/view_parquet.py
   ```

## Project Structure

```
forecast_refactored/
├── doc/                              # Documentation (you are here)
│   ├── README.md
│   ├── REGULATORY_EXTRACTION_GUIDE.md
│   └── TROUBLESHOOTING.md
├── src/forecasting/regulatory/ingest/  # Extraction code
│   ├── run_ingest.py                 # CLI entrypoint
│   ├── cellar_sparql.py              # SPARQL client
│   ├── parsing.py                    # Data normalization
│   └── io_utils.py                   # I/O helpers
├── conf/regulatory/                  # Configuration
│   ├── ingest.yaml                   # Main config
│   └── sources/
│       └── cellar.yaml               # Endpoint settings
├── scripts/                          # Utility scripts
│   └── view_parquet.py               # Parquet viewer
├── notebooks/                        # Jupyter notebooks
│   └── sanity_ingest.ipynb           # Quick data inspection
└── data/regulatory/raw/              # Output directory
    └── cellar_*.parquet              # Extracted data
```

## Support

For questions or issues:
1. **Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first** - covers most common problems
2. Review the [main guide's Troubleshooting section](REGULATORY_EXTRACTION_GUIDE.md#troubleshooting)
3. Check logs with `log_level=DEBUG`
4. Verify installation: `pip list | grep -E "(hydra|pandas|pyarrow)"`
