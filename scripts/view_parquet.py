#!/usr/bin/env python
"""
Parquet Viewer for Regulatory Data Extraction

Quick viewer for CELLAR parquet files with filtering, summary stats, and export options.

Usage:
    python scripts/view_parquet.py                                    # View latest file
    python scripts/view_parquet.py data/regulatory/raw/file.parquet  # View specific file
    python scripts/view_parquet.py --summary                          # Show only summary
    python scripts/view_parquet.py --filter resource_type=R           # Filter Regulations
    python scripts/view_parquet.py --output results.csv               # Export to CSV
    python scripts/view_parquet.py --head 20                          # Show first 20 rows
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed. Run: pip install pandas pyarrow")
    sys.exit(1)


def find_latest_parquet(directory: Path = Path("data/regulatory/raw")) -> Optional[Path]:
    """Find the most recent parquet file in the directory."""
    if not directory.exists():
        return None
    files = sorted(directory.glob("*.parquet"), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None


def print_summary(df: pd.DataFrame, filename: str) -> None:
    """Print summary statistics for the DataFrame."""
    print("\n" + "=" * 80)
    print(f"PARQUET FILE SUMMARY: {filename}")
    print("=" * 80)

    print(f"\nTotal documents: {len(df):,}")

    if df.empty:
        print("DataFrame is empty")
        return

    # Date range
    if "date_document" in df.columns:
        date_min = df["date_document"].min()
        date_max = df["date_document"].max()
        print(f"Date range: {date_min} to {date_max}")

    # Resource types
    if "resource_type" in df.columns:
        print("\nResource types:")
        type_counts = df["resource_type"].value_counts()
        for rtype, count in type_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {rtype}: {count:>6,} ({pct:>5.1f}%)")

    # Language distribution
    if "language" in df.columns:
        lang_counts = df["language"].value_counts().head(10)
        if not lang_counts.empty:
            print("\nTop languages:")
            for lang, count in lang_counts.items():
                if pd.notna(lang):
                    print(f"  {lang}: {count:>6,}")

    # Null value analysis
    print("\nNull values:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {col}: {count:>6,} ({pct:>5.1f}%)")

    # CELEX coverage
    if "celex" in df.columns:
        celex_coverage = df["celex"].notna().sum()
        pct = (celex_coverage / len(df)) * 100
        print(f"\nCELEX identifiers: {celex_coverage:,} ({pct:.1f}%)")

    print("\n" + "=" * 80)


def print_dataframe_preview(df: pd.DataFrame, n_rows: int = 10) -> None:
    """Print a formatted preview of the DataFrame."""
    if df.empty:
        print("\nNo data to display")
        return

    print(f"\nFirst {min(n_rows, len(df))} rows:")
    print("-" * 80)

    # Select columns to display
    display_cols = ["celex", "title", "date_document", "resource_type"]
    available_cols = [col for col in display_cols if col in df.columns]

    if not available_cols:
        available_cols = list(df.columns[:4])  # Fallback to first 4 columns

    preview = df[available_cols].head(n_rows)

    # Format and print
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 120)
    print(preview.to_string(index=False))
    print("-" * 80)

    # Show sample links if available
    if "link_html" in df.columns:
        print("\nSample links:")
        for i, row in df.head(3).iterrows():
            if pd.notna(row.get("link_html")):
                celex = row.get("celex", "N/A")
                print(f"  {celex}: {row['link_html']}")


def apply_filter(df: pd.DataFrame, filter_str: str) -> pd.DataFrame:
    """Apply a filter to the DataFrame.

    Args:
        df: DataFrame to filter
        filter_str: Filter string in format "column=value" or "column!=value"

    Returns:
        Filtered DataFrame
    """
    if "=" not in filter_str:
        print(f"Warning: Invalid filter format '{filter_str}'. Expected 'column=value'")
        return df

    if "!=" in filter_str:
        col, value = filter_str.split("!=", 1)
        col = col.strip()
        value = value.strip()
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found")
            return df
        return df[df[col] != value]
    else:
        col, value = filter_str.split("=", 1)
        col = col.strip()
        value = value.strip()
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found")
            return df
        return df[df[col] == value]


def export_data(df: pd.DataFrame, output_path: str) -> None:
    """Export DataFrame to various formats based on file extension."""
    output_file = Path(output_path)

    try:
        if output_file.suffix == ".csv":
            df.to_csv(output_file, index=False)
            print(f"\nExported {len(df):,} rows to CSV: {output_file}")
        elif output_file.suffix == ".json":
            df.to_json(output_file, orient="records", indent=2)
            print(f"\nExported {len(df):,} rows to JSON: {output_file}")
        elif output_file.suffix == ".xlsx":
            df.to_excel(output_file, index=False)
            print(f"\nExported {len(df):,} rows to Excel: {output_file}")
        elif output_file.suffix == ".parquet":
            df.to_parquet(output_file, index=False)
            print(f"\nExported {len(df):,} rows to Parquet: {output_file}")
        else:
            print(f"Warning: Unsupported format '{output_file.suffix}'. Exporting as CSV.")
            df.to_csv(output_file.with_suffix(".csv"), index=False)
    except Exception as e:
        print(f"Error exporting data: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze CELLAR parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                           # View latest file with summary
  %(prog)s data/regulatory/raw/cellar_20240101.parquet  # View specific file
  %(prog)s --summary                                 # Show only summary stats
  %(prog)s --filter resource_type=R                  # Show only Regulations
  %(prog)s --filter "title!=None"                    # Show only docs with titles
  %(prog)s --output results.csv                      # Export to CSV
  %(prog)s --head 20                                 # Show first 20 rows
  %(prog)s --no-preview                              # Summary only, no data preview
        """
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Path to parquet file (default: latest file in data/regulatory/raw)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary statistics, no data preview"
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter data (format: column=value or column!=value)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Export filtered data to file (supports .csv, .json, .xlsx, .parquet)"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Number of rows to preview (default: 10)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip data preview, show summary only"
    )

    args = parser.parse_args()

    # Determine file to load
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    else:
        file_path = find_latest_parquet()
        if file_path is None:
            print("Error: No parquet files found in data/regulatory/raw/")
            print("Run the ingestion first:")
            print("  python -m src.forecasting.regulatory.ingest.run_ingest")
            sys.exit(1)
        print(f"Loading latest file: {file_path.name}")

    # Load data
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

    # Apply filter if specified
    if args.filter:
        original_len = len(df)
        df = apply_filter(df, args.filter)
        print(f"\nFilter applied: {args.filter}")
        print(f"Rows after filter: {len(df):,} (from {original_len:,})")

    # Print summary
    print_summary(df, file_path.name)

    # Print preview unless disabled
    if not args.summary and not args.no_preview:
        print_dataframe_preview(df, n_rows=args.head)

    # Export if requested
    if args.output:
        export_data(df, args.output)

    # Print column info
    if not df.empty:
        print(f"\nAvailable columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
