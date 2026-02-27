#!/usr/bin/env python
"""
Orchestration script for the complete regulatory data pipeline.

Runs all steps from metadata extraction to BERT-ready dataset preparation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 80}")
    print(f"STEP: {description}")
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run regulatory data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/pipeline_runner.py --all

  # Run specific steps
  python scripts/pipeline_runner.py --ingest --enrich

  # Test mode (10 documents only)
  python scripts/pipeline_runner.py --enrich --test

  # Custom date range
  python scripts/pipeline_runner.py --ingest --date-from 2024-01-01 --date-to 2024-12-31
        """
    )

    # Step selection
    parser.add_argument("--all", action="store_true", help="Run all pipeline steps")
    parser.add_argument("--ingest", action="store_true", help="Step 1: Metadata ingestion (SPARQL)")
    parser.add_argument("--enrich", action="store_true", help="Step 2: Text enrichment (HTML scraping)")
    parser.add_argument("--preprocess", action="store_true", help="Step 3: Preprocessing for NLP")
    parser.add_argument("--prepare-annotation", action="store_true", help="Step 4: Prepare for annotation")

    # Options
    parser.add_argument("--test", action="store_true", help="Test mode (limited documents)")
    parser.add_argument("--date-from", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--date-to", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-docs", type=int, help="Max documents to process")

    args = parser.parse_args()

    # If --all, enable all steps
    if args.all:
        args.ingest = True
        args.enrich = True
        args.preprocess = True
        args.prepare_annotation = True

    # Check if at least one step is selected
    if not any([args.ingest, args.enrich, args.preprocess, args.prepare_annotation]):
        print("Error: No pipeline steps selected. Use --all or specify individual steps.")
        parser.print_help()
        sys.exit(1)

    success = True

    # Step 1: Metadata Ingestion
    if args.ingest:
        cmd = [
            "python", "-m", "src.forecasting.regulatory.ingest.run_ingest",
            f"date_from={args.date_from}",
            f"date_to={args.date_to}",
            "query_terms=[]",
        ]

        if args.test:
            cmd.extend(["max_pages=1", "page_size=10"])
        else:
            cmd.extend(["max_pages=20", "page_size=100"])

        if args.max_docs:
            # Calculate pages needed
            page_size = 10 if args.test else 100
            pages = (args.max_docs + page_size - 1) // page_size
            cmd.extend([f"max_pages={pages}", f"page_size={page_size}"])

        success = run_command(cmd, "Metadata Ingestion (CELLAR SPARQL)")
        if not success:
            print("\n⚠️ Ingestion failed. Stopping pipeline.")
            sys.exit(1)

    # Step 2: Text Enrichment
    if args.enrich:
        cmd = [
            "python", "-m", "src.forecasting.regulatory.enrichment.run_enrichment",
        ]

        if args.test or args.max_docs:
            max_docs = args.max_docs if args.max_docs else 10
            cmd.append(f"max_docs={max_docs}")

        success = run_command(cmd, "Text Enrichment (HTML Scraping)")
        if not success:
            print("\n⚠️ Enrichment failed. You can continue with other steps if metadata exists.")

    # Step 3: Preprocessing (placeholder)
    if args.preprocess:
        print("\n⚠️ Preprocessing step not yet implemented")
        print("Coming soon: sentence splitting, tokenization, cleaning")

    # Step 4: Annotation Preparation (placeholder)
    if args.prepare_annotation:
        print("\n⚠️ Annotation preparation not yet implemented")
        print("Coming soon: chunking, export to JSONL, stratified sampling")

    print(f"\n{'=' * 80}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 80}")

    if success:
        print("✅ Pipeline completed successfully!")
        print("\nNext steps:")
        if args.enrich:
            print("  1. View enriched data: python scripts/view_parquet.py data/regulatory/enriched/")
            print("  2. Check text quality in notebook: notebooks/explore_enriched.ipynb")
            print("  3. Run preprocessing: python scripts/pipeline_runner.py --preprocess")
    else:
        print("❌ Pipeline completed with errors. Check logs above.")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
