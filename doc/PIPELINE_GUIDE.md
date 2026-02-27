# Complete Pipeline Guide: From SPARQL to BERT Fine-tuning

This guide covers the **end-to-end pipeline** for extracting EU regulatory documents, enriching them with full text, and preparing them for BERT fine-tuning.

## Table of Contents
- [Pipeline Overview](#pipeline-overview)
- [Directory Structure](#directory-structure)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│           REGULATORY DATA → BERT FINE-TUNING PIPELINE         │
└──────────────────────────────────────────────────────────────┘

Step 1: METADATA EXTRACTION ✅
  Input:  None
  Tool:   CELLAR SPARQL
  Output: data/regulatory/raw/cellar_*.parquet
  Time:   ~2 minutes for 1000 docs
  Data:   CELEX, dates, links, eurovoc topics

Step 2: TEXT ENRICHMENT ✅
  Input:  cellar_*.parquet
  Tool:   HTML scraping with caching
  Output: data/regulatory/enriched/documents_with_text_*.parquet
  Time:   ~1 second per document (with rate limiting)
  Data:   + full_text column

Step 3: PREPROCESSING 🚧 (Coming soon)
  Input:  documents_with_text_*.parquet
  Tool:   spaCy / NLTK
  Output: data/regulatory/processed/documents_for_annotation.parquet
  Time:   ~10 seconds per 1000 docs
  Data:   Cleaned, sentence-split, language-detected

Step 4: ANNOTATION PREP 🚧 (Coming soon)
  Input:  documents_for_annotation.parquet
  Tool:   Custom chunking (512 tokens)
  Output: data/regulatory/annotation/samples_for_labeling.jsonl
  Time:   Instant
  Data:   BERT-ready chunks with metadata

Step 5: ANNOTATION 👤 (Manual)
  Input:  samples_for_labeling.jsonl
  Tool:   Label Studio / Prodigy / Custom
  Output: data/regulatory/annotation/labeled_data.jsonl
  Time:   Depends on annotation speed
  Data:   + labels

Step 6: BERT DATASET 🚧 (Coming soon)
  Input:  labeled_data.jsonl
  Tool:   HuggingFace datasets
  Output: data/regulatory/finetuning/bert_dataset/
  Time:   Instant
  Data:   Train/val/test splits, DatasetDict
```

---

## Directory Structure

```
forecast_refactored/
├── data/regulatory/
│   ├── raw/                          # Step 1: SPARQL metadata
│   │   └── cellar_YYYYMMDD_HHMMSS.parquet
│   │
│   ├── cache/                        # Cached HTML/text
│   │   └── {url_hash}.txt
│   │
│   ├── enriched/                     # Step 2: + full text
│   │   └── documents_with_text_*.parquet
│   │
│   ├── processed/                    # Step 3: cleaned for NLP
│   │   └── documents_for_annotation.parquet
│   │
│   ├── annotation/                   # Step 4-5: labeling
│   │   ├── samples_for_labeling.jsonl
│   │   └── labeled_data.jsonl
│   │
│   └── finetuning/                   # Step 6: BERT-ready
│       ├── bert_dataset/
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   └── test.parquet
│       └── metadata.json
│
├── src/forecasting/regulatory/
│   ├── ingest/                       # Step 1 ✅
│   ├── enrichment/                   # Step 2 ✅
│   ├── preprocessing/                # Step 3 🚧
│   └── annotation/                   # Step 4 🚧
│
└── scripts/
    ├── pipeline_runner.py            # Orchestration ✅
    └── prepare_for_bert.py           # Final conversion 🚧
```

---

## Step-by-Step Workflow

### **Step 1: Extract Metadata (Already Working)**

```bash
# Extract all documents from 2024
python -m src.forecasting.regulatory.ingest.run_ingest \
    query_terms=[] \
    date_from=2024-01-01 \
    date_to=2024-12-31 \
    max_pages=20 \
    page_size=100

# Output: data/regulatory/raw/cellar_20260105_094309.parquet
# Columns: work_uri, celex, title, abstract, date_document, resource_type,
#          eurovoc, link_html, link_pdf, language
```

**What you get:**
- CELEX identifiers for all documents
- Publication dates
- Document types (R=Regulation, L=Directive, etc.)
- EUR-Lex links (HTML + PDF)
- EuroVoc topic classifications

**Time:** ~2 minutes for 2000 documents

---

### **Step 2: Enrich with Full Text (New!)**

```bash
# Test with 10 documents first
python -m src.forecasting.regulatory.enrichment.run_enrichment \
    max_docs=10 \
    log_level=INFO

# Full enrichment (all documents)
python -m src.forecasting.regulatory.enrichment.run_enrichment \
    input_file=data/regulatory/raw/cellar_20260105_094309.parquet

# Output: data/regulatory/enriched/documents_with_text_20260105_103045.parquet
# New column: full_text (string, 1000-50000 characters)
```

**What you get:**
- Everything from Step 1
- `full_text` column with complete document text
- Cached downloads (reusable for future runs)

**Features:**
- ✅ Automatic rate limiting (1 req/sec default)
- ✅ Retry with exponential backoff
- ✅ Caching (avoid re-downloading)
- ✅ Progress logging every 10 documents

**Time:** ~1 second per document (600 docs = 10 minutes)

**Configuration:**

```yaml
# conf/regulatory/enrichment.yaml

# Input/output
input_file: "data/regulatory/raw/cellar_*.parquet"
output_file: "data/regulatory/enriched/documents_with_text_${now}.parquet"

# Rate limiting (be respectful!)
rate_limit_delay: 1.0  # Seconds between requests
timeout: 30
retries: 3

# Caching
cache_dir: "data/regulatory/cache"
use_cache: true

# Testing
max_docs: null  # Set to 10 for testing
```

---

### **Step 3: Preprocess for NLP (Coming Soon)**

```bash
# Will implement:
python -m src.forecasting.regulatory.preprocessing.run_preprocessing \
    input_file=data/regulatory/enriched/documents_with_text_*.parquet
```

**What it will do:**
- Sentence segmentation
- Language detection (filter non-English if needed)
- Remove boilerplate (headers, footers, page numbers)
- Normalize whitespace
- Basic quality filtering (min/max length)
- Token counting (for BERT chunking)

**Output schema:**
```python
{
    'celex': str,
    'full_text': str,
    'cleaned_text': str,
    'sentences': List[str],
    'language': str,  # 'en', 'it', 'fr', etc.
    'token_count': int,
    'quality_score': float  # 0-1
}
```

---

### **Step 4: Prepare for Annotation (Coming Soon)**

```bash
# Will implement:
python -m src.forecasting.regulatory.annotation.export_for_labeling \
    input_file=data/regulatory/processed/documents_for_annotation.parquet \
    chunk_size=512 \
    overlap=50 \
    output_format=jsonl
```

**What it will do:**
- Split long documents into 512-token chunks (BERT max length)
- Add overlap between chunks (50 tokens) for context
- Stratified sampling (ensure balanced representation)
- Export to annotation-friendly format

**Output format (JSONL):**
```json
{
  "id": "32019R1020_chunk_0",
  "celex": "32019R1020",
  "text": "Article 1. This regulation applies to...",
  "chunk_id": 0,
  "total_chunks": 5,
  "metadata": {
    "date_document": "2019-06-20",
    "resource_type": "R",
    "eurovoc": ["consumer protection", "product safety"]
  }
}
```

---

### **Step 5: Annotation (Manual Work)**

**Option A: Label Studio** (Recommended)
```bash
# Install
pip install label-studio

# Start server
label-studio start

# Import samples_for_labeling.jsonl
# Annotate via web UI
# Export as labeled_data.jsonl
```

**Option B: Prodigy**
```bash
# Requires license
prodigy textcat.manual regulatory_labels \
    data/regulatory/annotation/samples_for_labeling.jsonl \
    --label COMPLIANCE,SAFETY,ENVIRONMENTAL,OTHER
```

**Option C: Custom annotation tool**
- Build with Streamlit / Gradio
- Store annotations in database
- Export to JSONL

**Annotation schema example:**
```json
{
  "id": "32019R1020_chunk_0",
  "text": "Article 1...",
  "labels": ["COMPLIANCE", "SAFETY"],
  "annotator": "user123",
  "timestamp": "2026-01-05T10:30:00"
}
```

---

### **Step 6: Convert to BERT Dataset (Coming Soon)**

```bash
# Will implement:
python scripts/prepare_for_bert.py \
    --input data/regulatory/annotation/labeled_data.jsonl \
    --output data/regulatory/finetuning/bert_dataset \
    --model bert-base-uncased \
    --task classification
```

**What it will do:**
- Load annotations
- Tokenize with BERT tokenizer
- Create train/val/test splits (80/10/10)
- Save as HuggingFace Dataset
- Generate label mappings
- Compute class weights (for imbalanced data)

**Output structure:**
```
data/regulatory/finetuning/bert_dataset/
├── dataset_dict.json           # HuggingFace DatasetDict
├── train.parquet
├── val.parquet
├── test.parquet
├── label_mapping.json          # {"COMPLIANCE": 0, "SAFETY": 1, ...}
├── class_weights.json          # For loss balancing
└── metadata.json               # Stats, tokenizer info
```

**Usage in training:**
```python
from datasets import load_from_disk

dataset = load_from_disk('data/regulatory/finetuning/bert_dataset')

# dataset['train']: 1600 samples
# dataset['val']: 200 samples
# dataset['test']: 200 samples

# Columns: input_ids, attention_mask, labels
```

---

## Quick Start

### **Run Complete Pipeline (Test Mode)**

```bash
# Test with 10 documents (fast)
python scripts/pipeline_runner.py --all --test

# Output:
# - data/regulatory/raw/cellar_*.parquet (10 docs)
# - data/regulatory/enriched/documents_with_text_*.parquet (10 docs with text)
```

### **Run Individual Steps**

```bash
# Step 1 only: Extract metadata
python scripts/pipeline_runner.py --ingest --date-from 2024-01-01 --date-to 2024-12-31

# Step 2 only: Enrich with text
python scripts/pipeline_runner.py --enrich

# Steps 1+2 together
python scripts/pipeline_runner.py --ingest --enrich --max-docs 100
```

### **Production Run (1000 documents)**

```bash
python scripts/pipeline_runner.py \
    --ingest \
    --enrich \
    --date-from 2024-01-01 \
    --date-to 2024-12-31 \
    --max-docs 1000

# Time estimate:
# - Ingestion: ~2 minutes
# - Enrichment: ~17 minutes (1000 docs × 1 sec + overhead)
# Total: ~20 minutes
```

---

## Configuration

### **Global Paths** (`conf/paths/default.yaml`)

```yaml
data_dir: data/regulatory
raw_dir: ${paths.data_dir}/raw
enriched_dir: ${paths.data_dir}/enriched
processed_dir: ${paths.data_dir}/processed
annotation_dir: ${paths.data_dir}/annotation
finetuning_dir: ${paths.data_dir}/finetuning
cache_dir: ${paths.data_dir}/cache
```

### **Ingestion Config** (`conf/regulatory/ingest.yaml`)

```yaml
source: cellar
query_terms: []  # Empty for all documents
date_from: "2024-01-01"
date_to: "2024-12-31"
page_size: 100
max_pages: 20
```

### **Enrichment Config** (`conf/regulatory/enrichment.yaml`)

```yaml
# Adjust these based on your needs
rate_limit_delay: 1.0  # Increase to 2.0 if you get blocked
timeout: 30            # Increase for slow connections
max_docs: null         # Set to 10-50 for testing
use_cache: true        # Always keep this true!
```

---

## Best Practices

### **1. Always Test First**

```bash
# Test with small dataset before full run
python scripts/pipeline_runner.py --all --test  # 10 docs
python scripts/pipeline_runner.py --enrich --max-docs 50  # 50 docs
```

### **2. Use Caching**

```yaml
# enrichment.yaml
use_cache: true  # Saves money and time!
```

The cache stores downloaded text at `data/regulatory/cache/{hash}.txt`. If you re-run enrichment, it will reuse cached files.

### **3. Monitor Success Rate**

```bash
# Check enrichment logs
grep "success rate" logs/regulatory/*.log

# Expected: 80-95% success rate
# If <80%: increase timeout or check EUR-Lex availability
```

### **4. Rate Limiting**

```yaml
# Be respectful to EUR-Lex servers
rate_limit_delay: 1.0  # 1 request per second = 3600 req/hour

# If you get HTTP 429 (Too Many Requests):
rate_limit_delay: 2.0  # Slow down
```

### **5. Incremental Processing**

```bash
# Process by month to avoid long-running jobs
for month in {01..12}; do
    python scripts/pipeline_runner.py \
        --ingest --enrich \
        --date-from 2024-${month}-01 \
        --date-to 2024-${month}-31
done
```

### **6. Data Quality Checks**

```python
import pandas as pd

df = pd.read_parquet('data/regulatory/enriched/documents_with_text_*.parquet')

# Check text quality
text_lengths = df['full_text'].str.len()
print(f"Avg length: {text_lengths.mean():.0f} chars")
print(f"Min length: {text_lengths.min()}")
print(f"Max length: {text_lengths.max()}")

# Check for missing texts
missing = df['full_text'].isna().sum()
print(f"Missing texts: {missing} ({missing/len(df)*100:.1f}%)")

# View sample
print(df['full_text'].iloc[0][:500])  # First 500 chars
```

---

## Troubleshooting

### **Issue: Enrichment fails with timeout**

**Solution:**
```yaml
# enrichment.yaml
timeout: 60  # Increase from 30
retries: 5   # Increase from 3
```

### **Issue: Low success rate (<80%)**

**Possible causes:**
1. EUR-Lex servers are slow/down
2. URLs are malformed
3. Document structure changed

**Debug:**
```bash
# Run with DEBUG logging
python -m src.forecasting.regulatory.enrichment.run_enrichment \
    log_level=DEBUG \
    max_docs=5
```

### **Issue: Out of disk space**

**Solution:**
```bash
# Cache can grow large, clean old files
find data/regulatory/cache -type f -mtime +30 -delete  # Remove files older than 30 days
```

### **Issue: "No input file found"**

**Solution:**
```bash
# Run ingestion first
python scripts/pipeline_runner.py --ingest --test

# Or specify input file manually
python -m src.forecasting.regulatory.enrichment.run_enrichment \
    input_file=data/regulatory/raw/cellar_SPECIFIC_FILE.parquet
```

---

## Next Steps

1. **Run test pipeline:**
   ```bash
   python scripts/pipeline_runner.py --all --test
   ```

2. **Explore enriched data:**
   ```bash
   python scripts/view_parquet.py data/regulatory/enriched/
   ```

3. **Check text quality:**
   Open `notebooks/explore_enriched.ipynb` (to be created)

4. **Scale up:**
   ```bash
   python scripts/pipeline_runner.py --all --max-docs 1000
   ```

5. **Prepare for annotation:**
   Wait for Step 3-4 implementation, or manually:
   - Filter documents by topic (EuroVoc)
   - Sample subset for annotation
   - Export to your annotation tool

---

**Last Updated:** 2026-01-05
**Version:** 1.0
**Status:** Steps 1-2 implemented ✅, Steps 3-6 coming soon 🚧
